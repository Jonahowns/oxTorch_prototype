import torch
import Bio.PDB
from statistics import mean

# Import two helpful objects
from cg_mapper import conv
from cg_mapper import oneletter

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

def apply_mapping(chain, residue_mapping):
    cg_sites = []

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

            # side chain sites
            for x in res_mapping["sidechain"]:
                cg_sites.append({"id": onelettercode + "s" + str(cg_site_id), "coords": mean([atom_pos[y] for y in x])})
                cg_site_id += 1

    return cg_sites