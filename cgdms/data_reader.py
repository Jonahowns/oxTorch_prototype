import torch
import Bio.PDB
from statistics import mean
import json

# Import two helpful objects
from cg_mapper import conv
import units
import os
from cg_mapper import oneletter

# make
# def import_pdb(pdb_file):
#     if "/" in pdb_file:
#         pdbid = pdb_file.rsplit('/', 1)[1].split('.')[0]
#     else:
#         pdbid = pdb_file.split('.')[0]
#     structure = Bio.PDB.PDBParser().get_structure(pdbid, pdb_file)
#
#     # Get all chains in file
#     model = Bio.PDB.Selection.unfold_entities(structure, 'C')
#
#     chains = [x for x in model]
#     return chains


# takes in mapping and pdb chain and produces a cg system
# chain,
def apply_mapping(chain, residue_mapping):
    cg_sites = []
    topology = []

    # iterate through residues
    for residue in chain.get_residues():
        tags = residue.get_full_id()
        # print(tags)
        if tags[3][0] == " ":  # Denotes a protein Residue
            # Get Atoms
            atoms = residue.atoms()

            # Get Residues one letter code
            onelettercode = conv[residue.get_resname()]

            # Fetch Our Residue Mapping
            res_mapping = residue_mapping[onelettercode]

            # temporary storage of coordinates
            atom_pos = {}
            for atom in atoms:
                atom_pos[atom.get_id()] = atom.get_coord()

            cg_site_id = 0
            # main chain sites
            for x in res_mapping["mainchain"]:
                # id is just an index, residue id,
                cg_sites.append({"id": onelettercode + "m" + str(cg_site_id), "coords": mean([atom_pos[y] for y in x])})
                cg_site_id += 1

            cg_site_id = 0
            # side chain sites
            for x in res_mapping["sidechain"]:
                cg_sites.append({"id": onelettercode + "s" + str(cg_site_id), "coords": mean([atom_pos[y] for y in x])})
                cg_site_id += 1

    return cg_sites


# structure file is either a pdb or cif/mmcif
# cgsitedict is our particle definitions
def generate_input_file(structure_file, cgsitedict, residue_mapping, outdir=None):
    extension = os.path.basename(structure_file).rsplit(".", 1)[-1].lower()
    if extension in ("cif", "mmcif"):
        from Bio.PDB import MMCIFParser
        parser = MMCIFParser()
    else:
        from Bio.PDB import PDBParser
        parser = PDBParser()

    # Get the structure id (usually the rcsb id)
    if "/" in structure_file:
        structure_id = structure_file.rsplit('/', 1)[1].split('.')[0]
    else:
        structure_id = structure_file.split('.')[0]

    # Specify output file in directory specified by outdir
    if outdir != None:
        # Adds slash if none in outdir specification
        if outdir[-1] != '/':
            outdir += '/'
        outfile = outdir + structure_id + ".json"
    else:
        outfile = structure_id + ".json"

    # Get Available CG types from the generated cgsitedict
    # generated from define_particles
    global_subtypes = cgsitedict["cg_map"]
    mc_subtypes = cgsitedict["mc_map"]
    sc_subtypes = cgsitedict["sc_map"]

    N = cgsitedict["cg_num"]
    M = cgsitedict["mc_num"]
    S = cgsitedict["sc_num"]

    # read in structure
    struc = parser.get_structure("", structure_file)

    seq = ""
    coords = []
    cg_system = []
    for chain in struc[0]:
        cg_sites = apply_mapping(chain, residue_mapping)
        cg_system += cg_sites

    all_coords = [x['coords'] for x in cg_system]
    mc_coords = [x['coords'] for x in cg_system if 'm' in x['id']]
    sc_coords = [x['coords'] for x in cg_system if 's' in x['id']]

    #Convert to oxdna length units
    all_coords /= units.oxdna_length_to_A   # 1 oxdna = 8.518 Angstroms
    mc_coords /= units.oxdna_length_to_A   # 1 oxdna = 8.518 Angstroms
    sc_coords /= units.oxdna_length_to_A   # 1 oxdna = 8.518 Angstroms

    # Separate Subtypes for the goal of decreasing force calc times
    # Get List of system specific subtypes
    subtypes = [global_subtypes.index(x["id"]) for x in cg_system]
    # All maincahin ids contain the letter hence
    mc_subtypes = [mc_subtypes.index(x["id"]) for x in cg_system if 'm' in x['id']]
    sc_subtypes = [sc_subtypes.index(x["id"]) for x in cg_system if 's' in x['id']]


    filedata = {'all_coords': all_coords, "mc_coords": mc_coords, "sc_coords": sc_coords, "system_subtypes":subtypes,
                "mc_subtypes":mc_subtypes, "sc_subtypes": sc_subtypes}

    # write file
    out = open(outfile, 'w+')
    json.dump(filedata, out)
    out.close()









        for res in chain:
            # Skip hetero and water residues
            if res.id[0] != " ":
                continue
            seq += three_to_one_aas[res.get_resname()]
            if res.get_resname() == "GLY":
                # Extend vector of length 1 Å from Cα to act as fake centroid
                d = res["CA"].get_coord() - res["C"].get_coord() + res["CA"].get_coord() - res["N"].get_coord()
                coord_cent = res["CA"].get_coord() + d / np.linalg.norm(d)
            else:
                # Centroid coordinates of sidechain heavy atoms
                atom_coords = []
                for atom in res:
                    if atom.get_name() not in ("N", "CA", "C", "O") and atom.element != "H":
                        atom_coords.append(atom.get_coord())
                coord_cent = np.array(atom_coords).mean(0)
            coords.append([res["N"].get_coord(), res["CA"].get_coord(), res["C"].get_coord(), coord_cent])

    print(seq)
    if ss2_file:
        # Extract 3-state secondary structure prediction from PSIPRED ss2 output file
        ss_pred = ""
        with open(ss2_file) as f:
            for line in f:
                if len(line.rstrip()) > 0 and not line.startswith("#"):
                    ss_pred += line.split()[2]
        assert len(seq) == len(ss_pred), f"Sequence length is {len(seq)} but SS prediction length is {len(ss_pred)}"
        print(ss_pred)
    else:
        print("C" * len(seq))

    def coord_str(coord):
        return " ".join([str(round(c, 3)) for c in coord])

    for coord_n, coord_ca, coord_c, coord_cent in coords:
        print(f"{coord_str(coord_n)} {coord_str(coord_ca)} {coord_str(coord_c)} {coord_str(coord_cent)}")


# Print a protein data file from a PDB/mmCIF file and an optional PSIPRED ss2 file
def print_input_file(structure_file, ss2_file=None):
    extension = os.path.basename(structure_file).rsplit(".", 1)[-1].lower()
    if extension in ("cif", "mmcif"):
        from Bio.PDB import MMCIFParser
        parser = MMCIFParser()
    else:
        from Bio.PDB import PDBParser
        parser = PDBParser()

    struc = parser.get_structure("", structure_file)



    seq = ""
    coords = []
    for chain in struc[0]:
        for res in chain:
            # Skip hetero and water residues
            if res.id[0] != " ":
                continue
            seq += three_to_one_aas[res.get_resname()]
            if res.get_resname() == "GLY":
                # Extend vector of length 1 Å from Cα to act as fake centroid
                d = res["CA"].get_coord() - res["C"].get_coord() + res["CA"].get_coord() - res["N"].get_coord()
                coord_cent = res["CA"].get_coord() + d / np.linalg.norm(d)
            else:
                # Centroid coordinates of sidechain heavy atoms
                atom_coords = []
                for atom in res:
                    if atom.get_name() not in ("N", "CA", "C", "O") and atom.element != "H":
                        atom_coords.append(atom.get_coord())
                coord_cent = np.array(atom_coords).mean(0)
            coords.append([res["N"].get_coord(), res["CA"].get_coord(), res["C"].get_coord(), coord_cent])

    print(seq)
    if ss2_file:
        # Extract 3-state secondary structure prediction from PSIPRED ss2 output file
        ss_pred = ""
        with open(ss2_file) as f:
            for line in f:
                if len(line.rstrip()) > 0 and not line.startswith("#"):
                    ss_pred += line.split()[2]
        assert len(seq) == len(ss_pred), f"Sequence length is {len(seq)} but SS prediction length is {len(ss_pred)}"
        print(ss_pred)
    else:
        print("C" * len(seq))

    def coord_str(coord):
        return " ".join([str(round(c, 3)) for c in coord])

    for coord_n, coord_ca, coord_c, coord_cent in coords:
        print(f"{coord_str(coord_n)} {coord_str(coord_ca)} {coord_str(coord_c)} {coord_str(coord_cent)}")