import Bio
import Bio.PDB
import numpy as np
from statistics import mean

# Site of Main Chain sites (No sidechain atoms)
# Sites
import torch
import torch.nn.functional as F


# generates our cg site definitions from our residue mappings
# basically
def define_cg_particles(residue_mapping):
    # C includes carbonyl O
    # N includes amide H
    # Terms are charged
    # CA weight includes H
    atom_masses = {"CA": 13.0, "N": 15.0, "C": 28.0, "Nterm": 17.0, "Cterm": 16.0}
    mc_sites, sc_sites, vc_sites = [], [], []
    mc_mass, sc_mass, vc_mass = [], [], []
    # Fill Masses and cg_sites for subtyping
    for o in oneletter:
        cid = 0
        for mc_site in residue_mapping[o]["mainchain"]:
            mc_sites.append(o + "m" + str(cid))
            mc_site.append(sum([atom_masses[x] for x in mc_site]))
            # make cterm version
            if "C" in mc_site:
                mc_sites.append(o + "CT")
                mc_site.append(sum([atom_masses[x] for x in mc_site if x != 'C'])+atom_masses["Cterm"])
            # make nterm version
            if "N" in mc_site:
                mc_sites.append(o + "NT")
                mc_site.append(sum([atom_masses[x] for x in mc_site if x != 'N']) + atom_masses["Nterm"])
            cid += 1

        cid = 0
        for sc_site in residue_mapping[o]["sidechain"]:
            sc_sites.append(o + "s" + str(cid))
            sc_site.append(sum([atom_masses[x] for x in sc_site]))
            cid += 1

        cid = 0
        for vc_site in residue_mapping[o]["virtual"]:
            vc_sites.append(o + "v" + str(cid))
            vc_mass.append(sum([atom_masses[x] for x in vc_site]))
            cid += 1

    # all
    all_cg_sites = mc_sites + sc_sites + vc_sites
    all_cg_mass = mc_mass + sc_mass + vc_mass

    cgdict = {"cg_map": all_cg_sites, "mc_map": mc_sites, "sc_map": sc_sites, "vc_map": vc_sites,
              "masses": all_cg_mass, "mc_num": len(mc_sites), "sc_num": len(sc_sites),
              "vc_num": len(vc_sites), "cg_num": len(all_cg_sites)}

    return cgdict



conv = {'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K', 'ILE': 'I', 'PRO': 'P', 'THR': 'T', 'PHE': 'F',
        'ASN': 'N', 'GLY': 'G', 'HIS': 'H', 'LEU': 'L', 'ARG': 'R', 'TRP': 'W', 'ALA': 'A', 'VAL': 'V', 'GLU': 'E',
        'TYR': 'Y', 'MET': 'M'}

oneletter = ['C', 'D', 'S', 'Q', 'K', 'I', 'P', 'T', 'F', 'N', 'G', 'H', 'L', 'R', 'W', 'A', 'V', 'E', 'Y', 'M']

# Easy examples, just the CA site is shown
default_residue_mapping = {"mainchain": [["CA"]], "sidechain": [], "virtual": []}


# more complicated ex.
# for everything but glycine which lacks a CB atom, the glycine entry would have the sidechain as []
# ex_residue_mapping = {"mainchain": [["CA", "N"], ["CA", "O"]], "sidechain": ["CB"], "virtual": ["CA"], "topology": [vc0, sc0], "vdef": ?]}
# "mainchain" mappings are along the backbone of provided pdb
# ex_residue mappings specifies two main chain sites: 1) is located at the mean of the "CA" and "N"  atoms
# 2) is located at the mean of the "CA" and "O" atoms
residue_mapping = {}

for l in oneletter:
    residue_mapping[l] = default_residue_mapping


# map specifically defines subtypes of each particle
# necessary for creating parameter maps
# masses contains the mass in amu, will probs change to oxDNA units eventually
# This is NOT the particledict created for each system
proto_cgdict = define_cg_particles(residue_mapping)




# Must specify the specific type of application
# Supported Types
# Main Chain, these forces can be calculated in a straightforward fashion
# Side Chain, these forces can be calculated in a slightly different fashion


# Bonded Forces
# First parameter is # of particles, Second is Type of potential,
# Third is the location where to apply the , this will impact how forces are calculated
# The fourth parameter is whether the parameters should be learnable
# ['2', 'harmonic', 'all', True], ['pair', 'harmonic', 'mainchain'],

#{"name": "bonds", "n_body": 2, "potential": "harmonic", "application": "all", "learnable": True}
#{"name": "angles", "n_body": 3, "potential": "angle", "application": "mainchain", "learnable": True}
#{"name": "angles","n_body": 3, "potential": "angle", "application": "sidechain", "learnable": True}
#{"name": "dihedrals", "n_body": 4, "potential": "dihedral", "application": "mainchain", "learnable": True}

# Nonbonded Forces
# ['pair', 'electrostatics', 'all'], ['pair', 'hydrophobic', 'all]
#{"name": "charge", "n_body": 2, "potential": "electrostatics", "application": "all", "learnable": True}
#{"name": "hydro", "n_body": 2, "potential": "hydrophobic", "application": "all", "learnable": True}
#{"name": "XV", "n_body": 2, "potential": "excluded_volume  ", "application": "all", "learnable": True}


forcefield_specification_CA = {
    "Bonded Forces": [{"name": "bonds", "n_body": 2, "potential": "harmonic", "application": "all", "learnable": True, "param_number": 2, "param_key": ["k", "r0"]}],
    "Nonbonded Forces": [['2', 'excluded_volume', 'all']]
}

forcefield_spec_example = {
    "Bonded Forces": [{"name": "bonds", "n_body": 2, "potential": "harmonic", "application": "all", "learnable": True}],
    "Nonbonded Forces": [{"name": "XV", "n_body": 2, "potential": "excluded_volume", "application": "all", "learnable": True}]
}



# Forces

# Creates Parameter definitions and applications for use by simulator
# Also generates the Observables needed for use by the loss function
# Primarily driven by force field design

# Needs to contain all cg_sites from entire system
# def generate_parameters(cgsitedict, forcefield_specification, device="cpu"):
#     # get number of particles to determine number of parameters we need
#     # cgsitedict is the cgparticle dictionary generates
#     total_cg_types = cgsitedict["cg_num"]
#     vc_num = cgsitedict["vc_num"]
#     mc_num = cgsitedict["mc_num"]
#     sc_num = cgsitedict["sc_num"]
#     cg_mapping = cgsitedict["cg_map"]
#     mc_mapping = cgsitedict["mc_map"]
#     sc_mapping = cgsitedict["sc_map"]
#     vc_mapping = cgsitedict["vc_map"]
#
#     # Create Parameter Matrices
#     bonded = forcefield_specification['Bonded Forces']
#     nonbonded = forcefield_specification['Nonbonded Forces']
#     allforces = bonded + nonbonded
#     parameterdict = {}
#     for force in allforces:
#         application = force["application"]
#         n_term = force["n_body"]
#         n_params = force["param_number"]
#         param_key = force["param_key"]
#         if application == 'all':
#             # parameter matrix, key is defined by the total number of subtypes
#             param_shape = [n_params]
#             param_shape += [total_cg_types for x in range(n_term)]
#             param_shape = tuple(param_shape)
#             parameterdict[force["potential"]] = {'params': torch.tensor(param_shape, device=device), 'key': cg_mapping}
#         elif application == "mainchain":
#             param_shape = [n_params]
#             param_shape += [mc_num for x in range(n_term)]
#             param_shape = tuple(param_shape)
#             parameterdict[force["potential"]] = {'params': torch.tensor(param_shape, device=device), "key": mc_mapping}
#         # elif application == "sidechain":
#         #     ## sidechain application will need extra work for sure
#         #     param_shape = tuple([mc_num for x in range(n_term)])
#         #     parameterdict[force["potential"]] = {'params': torch.tensor(param_shape, device=device), "key": sc_mapping}
#
#     return parameterdict


# Must be Intialized for each new system,
# initalizes bond, angle, and dihedral lists
class ForceField(torch.nn.module):
    def __init__(self, forcefield_specification, cgsitedict, device='cpu'):
        self.global_subtypes = cgsitedict["cg_map"]
        self.mc_subtypes = cgsitedict["mc_map"]
        self.sc_subtypes = cgsitedict["sc_map"]

        self.N = cgsitedict["cg_num"]
        self.M = cgsitedict["mc_num"]
        self.S = cgsitedict["sc_num"]

        self.mc_to_global_map = [self.global_subtypes.index(x) for x in self.mc_subtypes]

        self.device = device

        self.bonded_forces = forcefield_specification['Bonded Forces']
        self.nonbonded_forces = forcefield_specification['Nonbonded Forces']

        self.forces_list = []




        for x in nonbonded:
            if x["name"] == "excluded_volume":
                self.init_xv(x['application'], default_eps=1.0, default_sigma=0.5)

    # Initialize Bonded Forces
    def init_bonded_forces(self, distances, cutoff=1.0):
        for x in self.bonded_forces:
            if x["name"] == "harmonic":
                self.init_harmonic(x["application"])
            elif x["name"] == "angle":
                self.init_angles(x["application"])
            elif x["name"] == "anm":
                self.init_anm(distances, cutoff=cutoff)
            elif x["name"] == "hanm":
                self.init_hanm(distances, cutoff=cutoff, default_k=1.0)




    # Enforces parameters always be called with the smallest values first ex. subtypes 12 15 and 2 is returned as 2, 12, 15
    def param_access(self, subtypelist):
        subtypelist.sort()
        return subtypelist

    def init_harmonic(self, application):
        if application == "all":
            self.params.harmonic.k = torch.tensor((self.N, self.N), device=self.device)
            self.params.harmonic.r0 = torch.tensor((self.N, self.N), device=self.device)
        elif application == "mainchain":
            self.params.harmonic.k = torch.tensor((self.M, self.M), device=self.device)
            self.params.harmonic.r0 = torch.tensor((self.M, self.M), device=self.device)

    def init_angles(self, application):
        if application == "all":
            self.params.angles.k = torch.tensor((self.N, self.N, self.N), device=self.device)
            self.params.angles.theta0 = torch.tensor((self.N, self.N, self.N), device=self.device)
        elif application == "mainchain":
            self.params.angles.k = torch.tensor((self.M, self.M, self.M), device=self.device)
            self.params.angles.theta0 = torch.tensor((self.M, self.M, self.M), device=self.device)

    def init_xv(self, application, default_eps=1.0, default_sigma=0.5):
        if application == "all":
            sigma = torch.tensor(self.N, device=self.device).fill_(default_sigma)
            self.params.xv.sigma = torch.triu(sigma)
            eps = torch.tensor(self.N, device=self.device).fill_(default_eps)
            self.params.xv.epsilon = torch.triu(eps)

    def init_anm(self, distances, cutoff, default_k=1.0):
        bond_matrix = torch.triu(torch.ones_like(distances))
        self.params.anm.bond_matrix = bond_matrix
        dist_triu =  torch.triu(distances)
        self.params.anm.r0 = torch.where(dist_triu <= cutoff, dist_triu, 0.)
        self.params.anm.globalk = torch.tensor(default_k)

    def init_hanm(self, distances, default_k=1.0):
        bond_matrix = torch.triu(torch.ones_like(distances).fill_(default_k))
        self.params.hanm.bond_matrix = bond_matrix
        k_matrix = torch.tensor((self.N, self.N), device=self.device).fill_(default_k)
        self.params.hanm.k_matrix = torch.triu(k_matrix)

    def load_system(self, systemObj, system_subtypes, system_mc_subtypes):

        mc_coord_map = ["m" in x for x in system_subtypes]
        sc_coord_map = ["s" in x for x in system_subtypes]
        self.system_subtypes = [self.global_subtypes.index(x) for x in system_subtypes] # system specific subtypes ex. [0, 3, 5, 3, 3, 2, 1]
        self.N = len(system_subtypes)
        self.mc_system_subtypes = [self.mc_subtypes.index(x) for x in system_mc_subtypes]
        self.mc_mask = mc_coord_map  # Gets Coordinates for Main chain entities

    def angles_mainchain(self, vectors_mc, min_image_mc):
        angle_forces = torch.zeros((vectors_mc.shape[0], 3))  # Per site forces from angles
        # Angle = acos (ba * bc) with particles a, b, b
        for i in range(self.M - 2):   # Number of angles
            # Fetch Params
            p1, p2, p3 = self.param_access(self.mc_system_subtypes[i:i+3])
            theta_knot = self.params.angles.theta0[p1][p2][p3]
            k = self.params.angles.theta0[p1][p2][p3]
            # vectors are already normalized
            ba = vectors_mc[i+1][i]
            bc = vectors_mc[i+1][i+2]
            ang = torch.acos((ba * bc))
            mag_F = 2*k*(ang-theta_knot)
            mag_E = k*(ang-theta_knot)**2

            ba_u = min_image_mc[i+1][i]
            bc_u = min_image_mc[i+1][i+2]
            ba_norms = ba_u.norm(dim=2)
            bc_norms = bc_u.norm(dim=2)

            cross_ba_bc = torch.cross(ba_u, bc_u, dim=2)
            fa = mag_F * F.normalize(torch.cross(ba_u, cross_ba_bc, dim=2), dim=2) / ba_norms.unsqueeze(2)
            fc = mag_F * F.normalize(torch.cross(-bc_u, cross_ba_bc, dim=2), dim=2) / bc_norms.unsqueeze(2)

            angle_forces[i] += fa
            angle_forces[i+2] += fc
            angle_forces[i+1] += -fa - fc


    def pairwise_mc(self, distances_mc, vectors_mc):
        pair_forces = torch.zeros((self.M, 3))  # per particle forces

        pair_dists = torch.diagonal(distances_mc, offset=-1)
        pair_vecs = torch.diagonal(vectors_mc, offset=-1)

        for i in range(self.M-1):  # Along Main Chain
            p1, p2 = self.param_access(self.mc_system_subtypes[i:i + 2])
            k = self.params.harmonic.k[p1][p2]
            r0 = self.params.harmonic.r0[p1][p2]
            force = - 2 * k * (pair_dists[i] - r0) * pair_vecs[i]
            pair_forces[i] -= force
            pair_forces[i+1] += force

    def lennard_jones(self, distances, vectors, cutoff):
        # Use Lorenz-Berthelot Mixing Rules
        forces = torch.tensor((self.N, 3))
        dist_tri = torch.triu(distances)
        lj = torch.where(dist_tri < cutoff, dist_tri, torch.zeros_like(dist_tri))
        interactions = torch.nonzero(lj)
        for x in interactions:
            # Grabbing the triangular matrix removes this case
            # if x[0] >= x[1]:
            #     continue
            # else:
            rij = lj[x[0]][x[1]]
            e, f = [self.system_subtypes[x[0]], self.system_subtypes[x[1]]]
            eps = torch.sqrt(self.params.xv.epsilon[e] * self.params.xv.epsilon[f])
            sigma = (self.params.xv.sigma[e] + self.params.xv.sigma[f]) / 2.
            force = vectors[x[0]][x[1]] * 4. * eps * (2 * (sigma / rij) ** 12 - (sigma / rij) ** 6)
            energy = vectors[x[0]][x[1]] * 4. * eps * ((sigma / rij) ** 12 - (sigma / rij) ** 6)
            forces[x[0]] -= force
            forces[x[1]] += force


    def harmonic_anm(self, distances, vectors):
        forces = torch.tensor((self.N, 3))
        bond_matrix = self.params.anm.bond_matrix
        interactions = torch.nonzero(bond_matrix)
        for x in interactions:
            p1, p2 = self.param_access([self.global_subtypes[x[0]], self.global_subtype[x[1]]])
            r0 = self.params.anm.r0[p1][p2]
            k = self.params.anm.globalk
            force = - 2 * k * (distances[x[0]][x[1]] - r0) * vectors[x[0]][x[1]]
            forces[x[0]] -= force
            forces[x[1]] += force






    def forces(self, distances_all, distances_mc, vectors_all, vectors_mc):
        total_force = torch.zeros_like(coords)
        total_U = torch.zeroslike(self.subtypes)
        for force in self.forcefunctions:
            F, U = force(coords)
            total_force += F
            total_U += U


class System(torch.nn.Module):
    def __init__():




