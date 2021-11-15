import Bio
import Bio.PDB
import numpy as np

# Site of Main Chain sites (No sidechain atoms)
# Sites
default_residue_mapping = {"mainchain": [["CA"]], "sidechain": [], "virtual": []}

for a in aa:
    residue_mapping

    residue_maps





def import_pdb(pdb_file):

    structure = Bio.PDB.PDBParser().get_structure(pdbid, pdb_file)