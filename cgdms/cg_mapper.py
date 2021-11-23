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
# Main Chain, these forces.py can be calculated in a straightforward fashion
# Side Chain, these forces.py can be calculated in a slightly different fashion


# Bonded Forces
# First parameter is # of particles, Second is Type of potential,
# Third is the location where to apply the , this will impact how forces.py are calculated
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








