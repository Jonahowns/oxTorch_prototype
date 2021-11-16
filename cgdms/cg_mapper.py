import Bio
import Bio.PDB
import numpy as np
from statistics import mean

# Site of Main Chain sites (No sidechain atoms)
# Sites
conv = {'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K', 'ILE': 'I', 'PRO': 'P', 'THR': 'T', 'PHE': 'F',
        'ASN': 'N',
        'GLY': 'G', 'HIS': 'H', 'LEU': 'L', 'ARG': 'R', 'TRP': 'W', 'ALA': 'A', 'VAL': 'V', 'GLU': 'E', 'TYR': 'Y',
        'MET': 'M'}

# Easy examples, just the CA site is shown
default_residue_mapping = {"mainchain": [["CA"]], "sidechain": [], "virtual": []}


# more complicated ex.
# for everything but glycine which lacks a CB atom, the glycine entry would have the sidechain as []
# ex_residue_mapping = {"mainchain": [["CA", "N"], ["CA", "O"]], "sidechain": ["CB"], "virtual": ["CA"]}
# "mainchain" mappings are along the backbone of provided pdb
# ex_residue mappings specifies two main chain sites: 1) is located at the mean of the "CA" and "N"  atoms
# 2) is located at the mean of the "CA" and "O" atoms
residue_mapping = {}

for key, item in conv:
    residue_mapping[item] = default_residue_mapping






# make
def import_pdb(pdb_file):
    if "/" in pdb_file:
        pdbid = pdb_file.rsplit('/', 1)[1].split('.')[0]
    else:
        pdbid = pdb_file.split('.')[0]
    structure = Bio.PDB.PDBParser().get_structure(pdbid, pdb_file)

    # Get all chains in file
    model = Bio.PDB.Selection.unfold_entities(structure, 'C')

    chains = [x for x in model]
    return chains

# takes in mapping and pdb chain and produces a cg system
# chain,

def apply_mapping(chain, mapping):
    cg_sites = []

    cg_site_id = 0
    res_id = 0
    # iterate through residues
    for residue in chain.get_residues():
        tags = residue.get_full_id()
        # print(tags)
        if tags[3][0] == " ":
            # Get Residues one letter code
            onelettercode = conv[residue.get_resname()]

            # Fetch Residue Mapping
            res_mapping = mapping[onelettercode]

            #temporary storage of coordinates
            atom_pos = {}
            for atom in atoms:
                atom_pos[atom.get_id()] = atom.get_coord()

            # main chain sites
            for x in res_mapping["mainchain"]:
                # id is just an index, residue id,
                cg_sites.append({"id": cg_site_id, "res_id": res_id, "type": "mc", "coords": mean([atom_pos[y] for y in x])})
                cg_site_id += 1

            # side chain sites
            for x in res_mapping["sidechain"]:
                cg_sites.append({"id": cg_site_id, "res_id": res_id, "type": "mc", "coords": mean([atom_pos[y] for y in x])})
                cg_site_id += 1



            mc_sites = [ x for x in res_mapping["mainchain"]]





            # get residue number and identity per chain
            chainseqtmp.append((tags[2], onelettercode))
            chainboundstmp.append(int(tags[3][1]))
            atoms = residue.get_atoms()
            # Center of Mass Used only if Orientation=True
            com = np.full(3, 0.0)
            count = 0

