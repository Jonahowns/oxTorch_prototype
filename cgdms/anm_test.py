import cg_mapper as cgm

default_residue_mapping = {"mainchain": [["CA"]], "sidechain": [], "virtual": []}

residue_mapping = {}
for l in cgm.oneletter:
    residue_mapping[l] = default_residue_mapping
    
cgdefs = cgm.define_cg_particles(residue_mapping)

forcefield_spec_example = {
    "Bonded Forces": [{"name": "bonds", "n_body": 2, "potential": "harmonic", "application": "all", "learnable": True}],
    "Nonbonded Forces": [{"name": "XV", "n_body": 2, "potential": "excluded_volume", "application": "all", "learnable": True}]
}

