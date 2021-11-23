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

def training_anm(model_filepath, atom_ff_definitions, device="cpu", verbosity=0):
    max_n_steps = 2_000
    learning_rate = 1e-4
    n_accumulate = 100


    parameters = atom_ff_definitions.parameters
    applications = atom_ff_definitions.applications

    simulator = Simulator(parameters, applications)

    train_set = ProteinDataset(train_proteins, train_val_dir, device=device)
    val_set = ProteinDataset(val_proteins, train_val_dir, device=device)

    optimizer = torch.optim.Adam(simulator.parameters(), lr=learning_rate)

    report("Starting training", 0, verbosity)
    for ei in count(start=0, step=1):
        # After 37 epochs reset the optimiser with a lower learning rate
        if ei == 37:
            optimizer = torch.optim.Adam(simulator.parameters(), lr=learning_rate / 2)

        train_rmsds, val_rmsds = [], []
        n_steps = min(250 * ((ei // 5) + 1), max_n_steps)  # Scale up n_steps over epochs
        train_inds = list(range(len(train_set)))
        val_inds = list(range(len(val_set)))
        shuffle(train_inds)
        shuffle(val_inds)
        simulator.train()
        optimizer.zero_grad()

        for i, ni in enumerate(train_inds):
            # basically need to get observables from starting info

            # then
            native_coords, inters_flat, inters_ang, inters_dih, masses, seq = train_set[ni]
            coords = simulator(native_coords.unsqueeze(0), inters_flat.unsqueeze(0),
                               inters_ang.unsqueeze(0), inters_dih.unsqueeze(0), masses.unsqueeze(0),
                               seq, native_coords.unsqueeze(0), n_steps, verbosity=verbosity)
            loss, passed = rmsd(coords[0], native_coords)
            train_rmsds.append(loss.item())
            if passed:
                loss_log = torch.log(1.0 + loss)
                loss_log.backward()
            report("  Training   {:4} / {:4} - RMSD {:6.2f} over {:4} steps and {:3} residues".format(
                i + 1, len(train_set), loss.item(), n_steps, len(seq)), 1, verbosity)
            if (i + 1) % n_accumulate == 0:
                optimizer.step()
                optimizer.zero_grad()
        simulator.eval()
        with torch.no_grad():
            for i, ni in enumerate(val_inds):
                native_coords, inters_flat, inters_ang, inters_dih, masses, seq = val_set[ni]
                coords = simulator(native_coords.unsqueeze(0), inters_flat.unsqueeze(0),
                                   inters_ang.unsqueeze(0), inters_dih.unsqueeze(0), masses.unsqueeze(0),
                                   seq, native_coords.unsqueeze(0), n_steps, verbosity=verbosity)
                loss, passed = rmsd(coords[0], native_coords)
                val_rmsds.append(loss.item())
                report("  Validation {:4} / {:4} - RMSD {:6.2f} over {:4} steps and {:3} residues".format(
                    i + 1, len(val_set), loss.item(), n_steps, len(seq)), 1, verbosity)
        torch.save({"distances": simulator.ff_distances.data,
                    "angles": simulator.ff_angles.data,
                    "dihedrals": simulator.ff_dihedrals.data,
                    "optimizer": optimizer.state_dict()},
                   model_filepath)
        report("Epoch {:4} - med train/val RMSD {:6.3f} / {:6.3f} over {:4} steps".format(
            ei + 1, np.median(train_rmsds), np.median(val_rmsds), n_steps), 0, verbosity)