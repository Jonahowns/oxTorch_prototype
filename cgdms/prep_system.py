# Differentiable molecular simulation of proteins with a coarse-grained potential
# Author: Joe G Greener

# biopython, PeptideBuilder and colorama are also imported in functions
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.functional import normalize

from itertools import count
from math import pi
import os
from random import choices, gauss, random, randrange, shuffle

# Read an input data file
# The protein sequence is read from the file but will overrule the file if provided
def read_input_file(fp, seq="", device="cpu"):
    with open(fp) as f:
        lines = f.readlines()
        if seq == "":
            seq = lines[0].rstrip()
        ss_pred = lines[1].rstrip()
        assert len(seq) == len(ss_pred), f"Sequence length is {len(seq)} but SS prediction length is {len(ss_pred)}"
    seq_info = []
    for i in range(len(seq)):
        for atom in atoms:
            seq_info.append((i, atom))
    n_atoms = len(seq_info)
    native_coords = torch.tensor(np.loadtxt(fp, skiprows=2), dtype=torch.float,
                                    device=device).view(n_atoms, 3)

    inters = torch.ones(n_atoms, n_atoms, dtype=torch.long, device=device) * -1
    for i in range(n_atoms):
        inters[i, i] = len(interactions) - 1 # Placeholder for same atom
        for j in range(i):
            res_sep = abs(seq_info[i][0] - seq_info[j][0])
            if 1 <= res_sep <= n_adjacent:
                # Due to known ordering we know that the order of residues is j->i
                info_1, info_2 = seq_info[j], seq_info[i]
            else:
                # Sort by amino acid index then by atom
                info_1, info_2 = sorted([seq_info[i], seq_info[j]],
                                        key=lambda x : (aas.index(seq[x[0]]), atoms.index(x[1])))
            inter = f"{seq[info_1[0]]}_{info_1[1]}_{seq[info_2[0]]}_{info_2[1]}"
            if res_sep == 0:
                inter += "_same"
            elif res_sep <= n_adjacent:
                inter += f"_adj{res_sep}"
            else:
                inter += "_other"
            inter_i = interactions.index(inter)
            inters[i, j] = inter_i
            inters[j, i] = inter_i
    inters_flat = inters.view(n_atoms * n_atoms)

    masses = []
    for i, r in enumerate(seq):
        mass_CA = 13.0 # Includes H
        mass_N = 15.0 # Includes amide H
        if i == 0:
            mass_N += 2.0 # Add charged N-terminus
        mass_C = 28.0 # Includes carbonyl O
        if i == len(seq) - 1:
            mass_C += 16.0 # Add charged C-terminus
        mass_cent = aa_masses[r] - 74.0 # Subtract non-centroid section
        if r == "G":
            mass_cent += 10.0 # Make glycine artificially heavier
        masses.append(mass_N)
        masses.append(mass_CA)
        masses.append(mass_C)
        masses.append(mass_cent)
    masses = torch.tensor(masses, device=device)

    # Different angle potentials for each residue
    inters_ang = torch.tensor([aas.index(r) for r in seq], dtype=torch.long, device=device)

    # Different dihedral potentials for each residue and predicted secondary structure type
    inters_dih = torch.tensor([aas.index(r) * len(ss_types) + ss_types.index(s) for r, s in zip(seq, ss_pred)],
                                dtype=torch.long, device=device)

    return native_coords, inters_flat, inters_ang, inters_dih, masses, seq

# Read an input data file and thread a new sequence onto it
def read_input_file_threaded(fp, seq, device="cpu"):
    coords, inters_flat, inters_ang, inters_dih, masses, seq = read_input_file(fp, seq, device=device)

    # Move centroids out to minimum distances for that sequence
    ind_ca, ind_cent = atoms.index("CA"), atoms.index("cent")
    for i, r in enumerate(seq):
        ca_cent_diff = coords[i * len(atoms) + ind_cent] - coords[i * len(atoms) + ind_ca]
        ca_cent_unitvec = ca_cent_diff / ca_cent_diff.norm()
        coords[i * len(atoms) + ind_cent] = coords[i * len(atoms) + ind_ca] + centroid_dists[r] * ca_cent_unitvec

    return coords, inters_flat, inters_ang, inters_dih, masses, seq

# Read a dataset of input files
class ProteinDataset(Dataset):
    def __init__(self, pdbids, coord_dir, device="cpu"):
        self.pdbids = pdbids
        self.coord_dir = coord_dir
        self.set_size = len(pdbids)
        self.device = device

    def __len__(self):
        return self.set_size

    def __getitem__(self, index):
        fp = os.path.join(self.coord_dir, self.pdbids[index] + ".txt")
        return read_input_file(fp, device=self.device)



# RMSD between two sets of coordinates with shape (n_atoms, 3) using the Kabsch algorithm
# Returns the RMSD and whether convergence was reached
def rmsd(c1, c2):
    device = c1.device
    r1 = c1.transpose(0, 1)
    r2 = c2.transpose(0, 1)
    P = r1 - r1.mean(1).view(3, 1)
    Q = r2 - r2.mean(1).view(3, 1)
    cov = torch.matmul(P, Q.transpose(0, 1))
    try:
        U, S, V = torch.svd(cov)
    except RuntimeError:
        report("  SVD failed to converge", 0)
        return torch.tensor([20.0], device=device), False
    d = torch.tensor([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, torch.det(torch.matmul(V, U.transpose(0, 1)))]
    ], device=device)
    rot = torch.matmul(torch.matmul(V, d), U.transpose(0, 1))
    rot_P = torch.matmul(rot, P)
    diffs = rot_P - Q
    msd = (diffs ** 2).sum() / diffs.size(1)
    return msd.sqrt(), True


# trajectory shape [n_confs, n_atoms, 3]
# returns bfactors shape [n_atoms]
def bfactors(traj, box, masses):
    # get aligned trajectory
    aligned_trajectory = align_trajectory(traj, box, masses)
    # calculate mean
    mean_conf = torch.mean(aligned_trajectory, dim=0)
    # calculate deviations
    deviations = aligned_trajectory - mean_conf.unsqueeze_(2)
    # squared deviations
    squared_deviations = torch.square(deviations)
    # mean of squared deviations
    msd = torch.mean(squared_deviations, dim=0)
    # B factors
    b_factors = 8*pi**2 / 3 * msd
    return b_factors


def align_trajectory(traj, box, masses):
    aligned_traj = torch.zeros_like(traj)

    # random config index
    rand_config = torch.randperm(traj.shape[0])[0]
    # choose random starting configuration
    conf_to_align_to = traj[rand_config]
    # center_in_box
    centered_conf_to_align_to = box.center_system(conf_to_align_to, masses)
    centered_conf_to_align_to_transpose = centered_conf_to_align_to.transpose(0, 1)

    for i in range(traj.shape[0]):
        if i == rand_config:
            aligned_traj[i] = centered_conf_to_align_to
            continue
        # center configuation
        centered_conf = box.center_system(traj[i], masses)
        # svd for rotation
        cov = torch.matmul(centered_conf, centered_conf_to_align_to_transpose)
        try:
            U, S, V = torch.svd(cov)
        except RuntimeError:
            print("SVD failed to converge")
        # rotation matrix
        R = torch.dot(V.T, U.T)
        # rotate config
        aligned_conf = torch.dot(R, centered_conf)
        aligned_traj[i] = aligned_conf

    return aligned_traj


def loss_bfactors(traj, box, masses, target_bfactors):
    bfacts_from_traj = bfactors(traj, box, masses)
    # loss will be the sum of the squared differences of the bfactors
    loss = torch.sum(torch.square(target_bfactors-bfacts_from_traj))
    return loss





# Generate starting coordinates
# conformation is extended/predss/random/helix
# def starting_coords(seq, conformation="extended", input_file="", device="cpu"):
#     import PeptideBuilder
#
#     coords = torch.zeros(len(seq) * len(atoms), 3, device=device)
#     backbone_atoms = ("N", "CA", "C", "O")
#     ss_phis = {"C": -120.0, "H": -60.0, "E": -120.0}
#     ss_psis = {"C":  140.0, "H": -60.0, "E":  140.0}
#
#     if conformation == "predss":
#         with open(input_file) as f:
#             ss_pred = f.readlines()[1].rstrip()
#     for i, r in enumerate(seq):
#         r_to_use = "A" if r == "G" else r
#         if i == 0:
#             structure = PeptideBuilder.initialize_res(r_to_use)
#         elif conformation == "predss":
#             structure = PeptideBuilder.add_residue(structure, r_to_use, ss_phis[ss_pred[i]], ss_psis[ss_pred[i]])
#         elif conformation == "random":
#             # ϕ can be -180° -> -30°, ψ can be anything
#             phi = -180 + random() * 150
#             psi = -180 + random() * 360
#             structure = PeptideBuilder.add_residue(structure, r_to_use, phi, psi)
#         elif conformation == "helix":
#             structure = PeptideBuilder.add_residue(structure, r_to_use, ss_phis["H"], ss_psis["H"])
#         elif conformation == "extended":
#             coil_level = 30.0
#             phi = -120.0 + gauss(0.0, coil_level)
#             psi =  140.0 + gauss(0.0, coil_level)
#             structure = PeptideBuilder.add_residue(structure, r_to_use, phi, psi)
#         else:
#             raise(AssertionError(f"Invalid conformation {conformation}"))
#         for ai, atom in enumerate(atoms):
#             if atom == "cent":
#                 coords[len(atoms) * i + ai] = torch.tensor(
#                     [at.coord for at in structure[0]["A"][i + 1] if at.name not in backbone_atoms],
#                     dtype=torch.float, device=device).mean(dim=0)
#             else:
#                 coords[len(atoms) * i + ai] = torch.tensor(structure[0]["A"][i + 1][atom].coord,
#                                                             dtype=torch.float, device=device)
#     return coords

# Print a protein data file from a PDB/mmCIF file and an optional PSIPRED ss2 file
# def print_input_file(structure_file, ss2_file=None):
#     extension = os.path.basename(structure_file).rsplit(".", 1)[-1].lower()
#     if extension in ("cif", "mmcif"):
#         from Bio.PDB import MMCIFParser
#         parser = MMCIFParser()
#     else:
#         from Bio.PDB import PDBParser
#         parser = PDBParser()
#     struc = parser.get_structure("", structure_file)
#
#     seq = ""
#     coords = []
#     for chain in struc[0]:
#         for res in chain:
#             # Skip hetero and water residues
#             if res.id[0] != " ":
#                 continue
#             seq += three_to_one_aas[res.get_resname()]
#             if res.get_resname() == "GLY":
#                 # Extend vector of length 1 Å from Cα to act as fake centroid
#                 d = res["CA"].get_coord() - res["C"].get_coord() + res["CA"].get_coord() - res["N"].get_coord()
#                 coord_cent = res["CA"].get_coord() + d / np.linalg.norm(d)
#             else:
#                 # Centroid coordinates of sidechain heavy atoms
#                 atom_coords = []
#                 for atom in res:
#                     if atom.get_name() not in ("N", "CA", "C", "O") and atom.element != "H":
#                         atom_coords.append(atom.get_coord())
#                 coord_cent = np.array(atom_coords).mean(0)
#             coords.append([res["N"].get_coord(), res["CA"].get_coord(), res["C"].get_coord(), coord_cent])
#
#     print(seq)
#     if ss2_file:
#         # Extract 3-state secondary structure prediction from PSIPRED ss2 output file
#         ss_pred = ""
#         with open(ss2_file) as f:
#             for line in f:
#                 if len(line.rstrip()) > 0 and not line.startswith("#"):
#                     ss_pred += line.split()[2]
#         assert len(seq) == len(ss_pred), f"Sequence length is {len(seq)} but SS prediction length is {len(ss_pred)}"
#         print(ss_pred)
#     else:
#         print("C" * len(seq))
#
#     def coord_str(coord):
#         return " ".join([str(round(c, 3)) for c in coord])
#
#     for coord_n, coord_ca, coord_c, coord_cent in coords:
#         print(f"{coord_str(coord_n)} {coord_str(coord_ca)} {coord_str(coord_c)} {coord_str(coord_cent)}")
#
# def fixed_backbone_design(input_file, simulator, n_mutations=2_000, n_min_steps=100,
#                             print_color=True, device="cpu", verbosity=0):
#     if print_color:
#         from colorama import Fore, Style
#         highlight_open = Fore.RED
#         highlight_close = Style.RESET_ALL
#     else:
#         highlight_open = ""
#         highlight_close = ""
#
#     coords, inters_flat, inters_ang, inters_dih, masses, native_seq = read_input_file(input_file, device=device)
#     energy_native_min = simulator(coords.unsqueeze(0), inters_flat.unsqueeze(0),
#                                     inters_ang.unsqueeze(0), inters_dih.unsqueeze(0),
#                                     masses.unsqueeze(0), native_seq, coords.unsqueeze(0),
#                                     n_min_steps, integrator="min", energy=True,
#                                     verbosity=verbosity).item()
#     print(f"Native score is {energy_native_min:6.1f}")
#
#     aa_weights = [pdb_aa_frequencies[aa] for aa in aas]
#     seq = "".join(choices(aas, weights=aa_weights, k=len(native_seq)))
#     coords, inters_flat, inters_ang, inters_dih, masses, seq = read_input_file_threaded(
#                                                                     input_file, seq, device=device)
#     energy_min = simulator(coords.unsqueeze(0), inters_flat.unsqueeze(0),
#                             inters_ang.unsqueeze(0), inters_dih.unsqueeze(0),
#                             masses.unsqueeze(0), seq, coords.unsqueeze(0),
#                             n_min_steps, integrator="min", energy=True, verbosity=verbosity).item()
#
#     for mi in range(n_mutations):
#         mutate_i = randrange(len(seq))
#         rand_aa = choices(aas, weights=aa_weights, k=1)[0]
#         # Ensure we don't randomly choose the same residue
#         while rand_aa == seq[mutate_i]:
#             rand_aa = choices(aas, weights=aa_weights, k=1)[0]
#         new_seq = seq[:mutate_i] + rand_aa + seq[(mutate_i + 1):]
#         coords, inters_flat, inters_ang, inters_dih, masses, new_seq = read_input_file_threaded(
#                                                                 input_file, new_seq, device=device)
#         new_energy_min = simulator(coords.unsqueeze(0), inters_flat.unsqueeze(0),
#                                     inters_ang.unsqueeze(0), inters_dih.unsqueeze(0),
#                                     masses.unsqueeze(0), new_seq, coords.unsqueeze(0),
#                                     n_min_steps, integrator="min", energy=True,
#                                     verbosity=verbosity).item()
#
#         if new_energy_min < energy_min:
#             decision = "accept_lower"
#         elif new_energy_min - energy_min < 10.0 and random() < -0.25 + 0.5 * (n_mutations - mi) / n_mutations:
#             decision = "accept_chance"
#         else:
#             decision = "reject"
#         print("{:5} / {:5} | {:6.1f} | {:13} | {:5.3f} | {}".format(mi + 1, n_mutations, new_energy_min,
#             decision, sum(1 for r1, r2 in zip(new_seq, native_seq) if r1 == r2) / len(native_seq),
#             "".join([f"{highlight_open}{r1}{highlight_close}" if r1 == r2 else r1 for r1, r2 in zip(new_seq, native_seq)])))
#         if decision.startswith("accept"):
#             seq = new_seq
#             energy_min = new_energy_min
#
#     print("        final | {:6.1f} | {:13} | {:5.3f} | {}".format(energy_min,
#             "-", sum(1 for r1, r2 in zip(seq, native_seq) if r1 == r2) / len(native_seq),
#             "".join([f"{highlight_open}{r1}{highlight_close}" if r1 == r2 else r1 for r1, r2 in zip(seq, native_seq)])))

# def train(model_filepath, device="cpu", verbosity=0):
#     max_n_steps = 2_000
#     learning_rate = 1e-4
#     n_accumulate = 100
#
#     simulator = Simulator(
#         torch.zeros(len(interactions), n_bins_pot, device=device),
#         torch.zeros(len(angles), n_aas, n_bins_pot, device=device),
#         torch.zeros(len(dihedrals), n_aas * len(ss_types), n_bins_pot + 2, device=device)
#     )
#
#     train_set = ProteinDataset(train_proteins, train_val_dir, device=device)
#     val_set   = ProteinDataset(val_proteins  , train_val_dir, device=device)
#
#     optimizer = torch.optim.Adam(simulator.parameters(), lr=learning_rate)
#
#     report("Starting training", 0, verbosity)
#     for ei in count(start=0, step=1):
#         # After 37 epochs reset the optimiser with a lower learning rate
#         if ei == 37:
#             optimizer = torch.optim.Adam(simulator.parameters(), lr=learning_rate / 2)
#
#         train_rmsds, val_rmsds = [], []
#         n_steps = min(250 * ((ei // 5) + 1), max_n_steps) # Scale up n_steps over epochs
#         train_inds = list(range(len(train_set)))
#         val_inds   = list(range(len(val_set)))
#         shuffle(train_inds)
#         shuffle(val_inds)
#         simulator.train()
#         optimizer.zero_grad()
#         for i, ni in enumerate(train_inds):
#             native_coords, inters_flat, inters_ang, inters_dih, masses, seq = train_set[ni]
#             coords = simulator(native_coords.unsqueeze(0), inters_flat.unsqueeze(0),
#                                 inters_ang.unsqueeze(0), inters_dih.unsqueeze(0), masses.unsqueeze(0),
#                                 seq, native_coords.unsqueeze(0), n_steps, verbosity=verbosity)
#             loss, passed = rmsd(coords[0], native_coords)
#             train_rmsds.append(loss.item())
#             if passed:
#                 loss_log = torch.log(1.0 + loss)
#                 loss_log.backward()
#             report("  Training   {:4} / {:4} - RMSD {:6.2f} over {:4} steps and {:3} residues".format(
#                     i + 1, len(train_set), loss.item(), n_steps, len(seq)), 1, verbosity)
#             if (i + 1) % n_accumulate == 0:
#                 optimizer.step()
#                 optimizer.zero_grad()
#         simulator.eval()
#         with torch.no_grad():
#             for i, ni in enumerate(val_inds):
#                 native_coords, inters_flat, inters_ang, inters_dih, masses, seq = val_set[ni]
#                 coords = simulator(native_coords.unsqueeze(0), inters_flat.unsqueeze(0),
#                                     inters_ang.unsqueeze(0), inters_dih.unsqueeze(0), masses.unsqueeze(0),
#                                     seq, native_coords.unsqueeze(0), n_steps, verbosity=verbosity)
#                 loss, passed = rmsd(coords[0], native_coords)
#                 val_rmsds.append(loss.item())
#                 report("  Validation {:4} / {:4} - RMSD {:6.2f} over {:4} steps and {:3} residues".format(
#                         i + 1, len(val_set), loss.item(), n_steps, len(seq)), 1, verbosity)
#         torch.save({"distances": simulator.ff_distances.data,
#                     "angles"   : simulator.ff_angles.data,
#                     "dihedrals": simulator.ff_dihedrals.data,
#                     "optimizer": optimizer.state_dict()},
#                     model_filepath)
#         report("Epoch {:4} - med train/val RMSD {:6.3f} / {:6.3f} over {:4} steps".format(
#                 ei + 1, np.median(train_rmsds), np.median(val_rmsds), n_steps), 0, verbosity)
