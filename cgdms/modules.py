import torch
import numpy as np
import torch.nn.functional as F

from force import ForceField

class Thermostat(torch.nn.Module):
    def __init__(self, SimObj, type, timestep, temperature, device="cpu", **thrmst_params):
        if type is "vel":
            self.type = type
            self.apply = self.apply_no_vel
            neccesary_parameters = ["thermostat_cnst"]
        elif type is "no_vel":
            self.type = type
            self.apply = self.apply_vel
            neccesary_parameters = ["thermostat_cnst"]
        else:
            print(f"Type {type} is not supported")

        self.device = device
        self.temperature = temperature
        self.timestep = timestep
        self.therm_params = {}
        for param, val in thrmst_params:
            self.therm_params[param] = val

        self.sim = SimObj

    def apply_vel(self):
        thermostat_prob = self.timestep / self.therm_params["thermostat_cnst"]
        rndm_numbers = torch.random(self.sim.vels.shape[0]) < thermostat_prob
        self.sim.vels[rndm_numbers] = torch.randn(3, device=self.device)
        self.sim.vels[rndm_numbers] /= self.sim.masses[rndm_numbers]

    def apply_no_vel(self):
        thermostat_prob = self.timestep / self.therm_params["thermostat_cnst"]
        rndm_numbers = torch.random(self.sim.coords.shape[0]) < thermostat_prob
        self.sim.coords[rndm_numbers] = torch.randn(3, device=self.device) * self.temperature * self.timestep

    # for ai in range(n_atoms):
    #     if torch.random(vels)
    #     if random() < thermostat_prob:
    #
    #             # Actually this should be divided by the mass
    #             new_vel = torch.randn(3, device=device) * temperature
    #             vels[0, ai] = new_vel
    #         elif self.type == "no_vel":
    #             new_diff = torch.randn(3, device=device) * temperature * self.timestep
    #             coords_last[0, ai] = coords[0, ai] - new_diff

class Integrator(torch.nn.Module):
    def __init__(self, Simobj, type, timestep, temperature, masses, device="cpu", otherparams=None):
        supported_types = ['vel', 'langevin', 'langevin_simple']
        if type in supported_types:
            self.type = type
        else:
            print(f"Integrator type {type} not supported")

        self.sim = Simobj # lets us access the Simulation Tensors

        if otherparams != None:
            self.otherparams = otherparams

        if self.type == "vel":
            self.first_step = self.first_step_vel
            self.second_step = self.second_step_vel
        if self.type == "no_vel":
            self.first_step = None
            self.second_step = self.second_step_no_vel
        elif self.type == "langevin":
            self.first_step = self.first_step_langevin
            self.second_step = self.second_step_langevin
        elif self.type == "langevin_simple":
            self.first_step = self.first_step_langevin_simple
            self.second_step = self.second_step_langevin_simple

        self.temp = temperature
        self.masses = masses
        self.timestep = timestep
        self.device = device

    def first_step_vel(self):
        self.sim.coords = self.sim.coords + self.sim.vels * self.timestep + 0.5 * self.sim.accs_last * self.timestep * self.timestep

    def first_step_langevin(self):
        alpha, twokbT = self.otherparams['thermostat_const'], self.otherparams['temperature']
        beta = np.sqrt(twokbT * alpha * self.timestep) * torch.randn(self.sim.vels.shape, device=self.device)
        b = 1.0 / (1.0 + (alpha * self.timestep) / (2 * self.masses.unsqueeze(2)))
        self.sim.coords_last = self.sim.coords # ?
        self.sim.coords = self.sim.coords + b * self.timestep * self.sim.vels + 0.5 * b * (self.timestep ** 2) * self.sim.accs_last + 0.5 * b * self.timestep * beta / self.sim.masses.unsqueeze(2)

    def first_step_langevin_simple(self):
        self.sim.coords = self.sim.coords + self.sim.vels * self.timestep + 0.5 * self.sim.accs_last * self.timestep * self.timestep

    def second_step_vel(self):
        self.sim.vels = self.sim.vels + 0.5 * (self.sim.accs_last + self.sim.accs) * self.timestep
        self.sim.accs_last = self.sim.accs

    def second_step_no_vel(self):
        coords_next = 2 * self.sim.coords - self.sim.coords_last + self.sim.accs * self.timestep * self.timestep
        self.sim.coords_last = self.sim.coords
        self.sim.coords = coords_next

    def second_step_langevin(self):
        # From Gronbech-Jensen 2013
        self.sim.vels = self.sim.vels + 0.5 * self.timestep * (self.sim.accs_last + self.sim.accs) - self.otherparams['alpha'] * (self.sim.coords - self.sim.coords_last) / self.sim.masses.unsqueeze(
            2) + self.otherparams['beta'] / self.sim.masses.unsqueeze(2)
        self.sim.accs_last = self.sim.accs

    def second_step_langevin_simple(self):
        gamma, twokbT = self.otherparams['thermostat_const'], ['temperature']
        self.sim.accs = self.sim.accs + (-gamma * self.sim.vels + np.sqrt(gamma * twokbT) * torch.randn(self.sim.vels.shape,
            device=self.device)) / self.sim.masses.unsqueeze(2)
        self.sim.vels = self.sim.vels + 0.5 * (self.sim.accs_last + self.sim.accs) * self.timestep
        self.sim.accs_last = self.sim.accs

# example kinetic_energy 10
class Reporter(torch.nn.Module):  # prints out observables etc.
    def __init__(self, Simobj, reportdict):
        super(Reporter, self).__init__()
        self.sim = Simobj

        self.keys = []
        self.freq = []
        supportedreports = ['kinetic_energy', 'step']
        for key, item in reportdict:
            if key in supportedreports:
                self.keys.append(key)
                self.freq.append(item)



        self.functiondict = {'kinetic_energy':self.kinetic_energy, }

    def report(self):
        for freq in self.freq:
            if self.sim.step


    def kinetic_energy(self):




# Differentiable molecular simulation of proteins with a coarse-grained potential
class Simulator(torch.nn.Module):

    """
    Parameters is a Dictionary of Tensors that will be learned

    ex. {bond_constants : torch.tensor}

    Application is a Dictionary defining how the tensors will be applied to the simulation data
    """
    def __init__(self, particledict, parameterdict, applicationdict,
                 forcefield_spec, thermostatdict, reportdict, box_size, device='cpu'):
        super(Simulator, self).__init__()
        self.params = {}
        self.application = {}
        for key, item in parameterdict:
            self.params[key] = torch.nn.Parameter(item)
        for key, item in applicationdict:
            self.application[key] = item

        self.masses = particledict.masses
        self.coords = particledict.coords

        # Intialize Tensors which are edited in Integrator and Thermostat Object
        if thermostatdict['type'] != "no_vel":
            self.vels = torch.randn(self.coords.shape, device=device) * thermostatdict['start_temperature']
            self.accs_last = torch.zeros(self.coords.shape, device=device)
            self.accs = torch.zeros(self.coords.shape, device=device)
        else:
            self.accs = torch.zeros(self.coords.shape, device=device)
            self.coords_last = self.coords.clone() + torch.randn(self.coords.shape, device=device) * \
                               thermostatdict['start_temperature'] * thermostatdict['timestep']

        self.Thermostat = Thermostat(self, thermostatdict['type'], thermostatdict['timestep'], thermostatdict['temperature'],
                                     thermostatdict['thermostatparams'], device=device)

        self.Integrator = Integrator(self, thermostatdict['type'], thermostatdict['time'], thermostatdict['timestep'],
                                     thermostatdict['temperature'], otherparams=thermostatdict['thermostatparams'],
                                     device=device)

        self.Reporter = Reporter(self, reportdict)

        self.System_Observables = System_Obervables

        self.Force_Field = ForceField(forcefield_spec, particledict)

        self.Box = torch.tensor([box_size, box_size, box_size], device=device)

        # self.ff_distances = torch.nn.Parameter(ff_distances)
        # self.ff_angles    = torch.nn.Parameter(ff_angles)
        # self.ff_dihedrals = torch.nn.Parameter(ff_dihedrals)

    # def sim_step_novel(self, coords, masses,):
    def center_system(self):
        center = self.Box/2
        current_com = torch.mean(self.coords*F.normalize(self.massses))
        self.coords.add_(center - current_com)

    # returns difference vectors in matrix form for all coordinates and enforces minimum image convention
    # vector from p0 to p1 = min_image[0][1]
    def min_image(self, coords):
        box_size = self.Box[0]  # Assumes Cubic Box at least for now
        n_atoms = coords.shape[0]
        tmp = coords.unsqueeze(1).expand(-1, n_atoms, -1)
        diffs = tmp - tmp.transpose(0, 1)
        min_image = diffs - torch.round(diffs / box_size) * box_size
        return min_image

    # Returns Distances b/t all particles as a symmetric matrixd
    def distances(selfs, min_image):
        return min_image.norm(dim=2)

    # Returns Matrix of normalized vectors ex. vectors[0][1] returns the normalized vector from particle 0 pointing at particle 1
    def vectors(self, min_image):
        return F.normalize(min_image, dim=2)



    def sim_step_vel(self, n_steps, integrator="vel", device="cpu", start_temperature=0.1, timestep=0.02,
                 verbosity = 0,
                 thermostat_const=0.0, # Set to 0.0 to run without a thermostat (NVE ensemble)
                 temperature=0.0, # The effective temperature of the thermostat
         ):



        for i in range(n_steps):

            self.Integrator.first_step()

            min_image = self.min_image(self.coords)
            distances = self.distances(min_image)
            vectors = self.vectors(min_image)

            min_image_mc = min_image[self.mc_mask]
            distances_mc = self.distances(min_image_mc)
            vectors_mc = self.vectors(min_image_mc)

            #force_calculation, return the accs f/mass
            # return energy here as well
            # F, U = self.Force_Field.compute_forces(distances, vectors, distances_mc, vectors_mc)
            self.accs = F/self.masses

            self.Integrator.second_step()

            self.Thermostat.apply()

            self.Reporter.report()


    def forward(self,
                coords,
                orientations,
                inters_flat,
                inters_ang,
                inters_dih,
                ,
                seq,
                native_coords,
                n_steps,
                integrator="vel", # vel/no_vel/min/langevin/langevin_simple
                timestep=0.02,
                start_temperature=0.1,
                thermostat_const=0.0, # Set to 0.0 to run without a thermostat (NVE ensemble)
                temperature=0.0, # The effective temperature of the thermostat
                sim_filepath=None, # Output PDB file to write to or None to not write out
                energy=False, # Return the energy at the end of the simulation
                report_n=10_000, # Print and write PDB every report_n steps
                verbosity=2, # 0 for epoch info, 1 for protein info, 2 for simulation step info
        ):

        assert integrator in ("vel", "no_vel", "min", "langevin", "langevin_simple"), f"Invalid integrator {integrator}"
        device = coords.device


        batch_size, n_atoms = masses.size(0), masses.size(1)

        n_res = n_atoms // len(atoms)
        dist_bin_centres_tensor = torch.tensor(dist_bin_centres, device=device)
        pair_centres_flat = dist_bin_centres_tensor.index_select(0, inters_flat[0]).unsqueeze(0).expand(batch_size, -1, -1)
        pair_pots_flat = self.ff_distances.index_select(0, inters_flat[0]).unsqueeze(0).expand(batch_size, -1, -1)
        angle_bin_centres_tensor = torch.tensor(angle_bin_centres, device=device)
        angle_centres_flat = angle_bin_centres_tensor.unsqueeze(0).unsqueeze(0).expand(batch_size, n_res, -1)
        angle_pots_flat = self.ff_angles.index_select(1, inters_ang[0]).unsqueeze(0).expand(batch_size, -1, -1, -1)
        dih_bin_centres_tensor = torch.tensor(dih_bin_centres, device=device)
        dih_centres_flat = dih_bin_centres_tensor.unsqueeze(0).unsqueeze(0).expand(batch_size, n_res - 1, -1)
        dih_pots_flat = self.ff_dihedrals.index_select(1, inters_dih[0]).unsqueeze(0).expand(batch_size, -1, -1, -1)
        native_coords_ca = native_coords.view(batch_size, n_res, 3 * len(atoms))[0, :, 3:6]
        model_n = 0

        # just preparing needed vectors
        if integrator == "vel" or integrator == "langevin" or integrator == "langevin_simple":
            vels = torch.randn(coords.shape, device=device) * start_temperature
            accs_last = torch.zeros(coords.shape, device=device)
        elif integrator == "no_vel":
            coords_last = coords.clone() + torch.randn(coords.shape, device=device) * start_temperature * timestep


        # The step the energy is return on is not used for simulation so we add an extra step
        if energy:
            n_steps += 1

        for i in range(n_steps):

            # MD Backend First step
            if integrator == "vel":
                coords = coords + vels * timestep + 0.5 * accs_last * timestep * timestep
            elif integrator == "langevin":
                # From Gronbech-Jensen 2013
                alpha, twokbT = thermostat_const, temperature
                beta = np.sqrt(twokbT * alpha * timestep) * torch.randn(vels.shape, device=device)
                b = 1.0 / (1.0 + (alpha * timestep) / (2 * self.masses.unsqueeze(2)))
                coords_last = coords
                coords = coords + b * timestep * vels + 0.5 * b * (timestep ** 2) * accs_last + 0.5 * b * timestep * beta / masses.unsqueeze(2)
            elif integrator == "langevin_simple":
                coords = coords + vels * timestep + 0.5 * accs_last * timestep * timestep

            # See https://arxiv.org/pdf/1401.1181.pdf for derivation of forces.py
            printing = verbosity >= 2 and i % report_n == 0
            returning_energy = energy and i == n_steps - 1
            if printing or returning_energy:
                dist_energy = torch.zeros(1, device=device)
                angle_energy = torch.zeros(1, device=device)
                dih_energy = torch.zeros(1, device=device)

            # Add pairwise distance forces.py
            crep = coords.unsqueeze(1).expand(-1, n_atoms, -1, -1) # makes list of coords like [[ [coord1] n times ], [coord2] n times], [coord3] n times]]
            diffs = crep - crep.transpose(1, 2)
            dists = diffs.norm(dim=3)
            dists_flat = dists.view(batch_size, n_atoms * n_atoms)
            dists_from_centres = pair_centres_flat - dists_flat.unsqueeze(2).expand(-1, -1, n_bins_force)
            dist_bin_inds = dists_from_centres.abs().argmin(dim=2).unsqueeze(2)
            # Force is gradient of potential
            # So it is proportional to difference of previous and next value of potential
            pair_forces_flat = 0.5 * (pair_pots_flat.gather(2, dist_bin_inds) - pair_pots_flat.gather(2, dist_bin_inds + 2))
            # Specify minimum to prevent division by zero errors
            norm_diffs = diffs / dists.clamp(min=0.01).unsqueeze(3)
            pair_accs = (pair_forces_flat.view(batch_size, n_atoms, n_atoms)).unsqueeze(3) * norm_diffs
            accs = pair_accs.sum(dim=1) / masses.unsqueeze(2)
            if printing or returning_energy:
                dist_energy += 0.5 * pair_pots_flat.gather(2, dist_bin_inds + 1).sum()

            atom_coords = coords.view(batch_size, n_res, 3 * len(atoms))
            atom_accs = torch.zeros(batch_size, n_res, 3 * len(atoms), device=device)
            # Angle forces.py
            # across_res is the number of atoms in the next residue, starting from atom_3
            for ai, (atom_1, atom_2, atom_3, across_res) in enumerate(angles):
                ai_1, ai_2, ai_3 = atoms.index(atom_1), atoms.index(atom_2), atoms.index(atom_3)
                if across_res == 0:
                    ba = atom_coords[:, :  , (ai_1 * 3):(ai_1 * 3 + 3)] - atom_coords[:, :  , (ai_2 * 3):(ai_2 * 3 + 3)]
                    bc = atom_coords[:, :  , (ai_3 * 3):(ai_3 * 3 + 3)] - atom_coords[:, :  , (ai_2 * 3):(ai_2 * 3 + 3)]
                    # Use residue potential according to central atom
                    angle_pots_to_use = angle_pots_flat[:, ai, :]
                elif across_res == 1:
                    ba = atom_coords[:, :-1, (ai_1 * 3):(ai_1 * 3 + 3)] - atom_coords[:, :-1, (ai_2 * 3):(ai_2 * 3 + 3)]
                    bc = atom_coords[:, 1: , (ai_3 * 3):(ai_3 * 3 + 3)] - atom_coords[:, :-1, (ai_2 * 3):(ai_2 * 3 + 3)]
                    angle_pots_to_use = angle_pots_flat[:, ai, :-1]
                elif across_res == 2:
                    ba = atom_coords[:, :-1, (ai_1 * 3):(ai_1 * 3 + 3)] - atom_coords[:, 1: , (ai_2 * 3):(ai_2 * 3 + 3)]
                    bc = atom_coords[:, 1: , (ai_3 * 3):(ai_3 * 3 + 3)] - atom_coords[:, 1: , (ai_2 * 3):(ai_2 * 3 + 3)]
                    angle_pots_to_use = angle_pots_flat[:, ai, 1:]
                ba_norms = ba.norm(dim=2)
                bc_norms = bc.norm(dim=2)
                angs = torch.acos((ba * bc).sum(dim=2) / (ba_norms * bc_norms))
                n_angles = n_res if across_res == 0 else n_res - 1
                angles_from_centres = angle_centres_flat[:, :n_angles] - angs.unsqueeze(2)
                angle_bin_inds = angles_from_centres.abs().argmin(dim=2).unsqueeze(2)
                angle_forces = 0.5 * (angle_pots_to_use.gather(2, angle_bin_inds) - angle_pots_to_use.gather(2, angle_bin_inds + 2))
                cross_ba_bc = torch.cross(ba, bc, dim=2)
                fa = angle_forces * normalize(torch.cross( ba, cross_ba_bc, dim=2), dim=2) / ba_norms.unsqueeze(2)
                fc = angle_forces * normalize(torch.cross(-bc, cross_ba_bc, dim=2), dim=2) / bc_norms.unsqueeze(2)
                fb = -fa -fc
                if across_res == 0:
                    atom_accs[:, :  , (ai_1 * 3):(ai_1 * 3 + 3)] += fa
                    atom_accs[:, :  , (ai_2 * 3):(ai_2 * 3 + 3)] += fb
                    atom_accs[:, :  , (ai_3 * 3):(ai_3 * 3 + 3)] += fc
                elif across_res == 1:
                    atom_accs[:, :-1, (ai_1 * 3):(ai_1 * 3 + 3)] += fa
                    atom_accs[:, :-1, (ai_2 * 3):(ai_2 * 3 + 3)] += fb
                    atom_accs[:, 1: , (ai_3 * 3):(ai_3 * 3 + 3)] += fc
                elif across_res == 2:
                    atom_accs[:, :-1, (ai_1 * 3):(ai_1 * 3 + 3)] += fa
                    atom_accs[:, 1: , (ai_2 * 3):(ai_2 * 3 + 3)] += fb
                    atom_accs[:, 1: , (ai_3 * 3):(ai_3 * 3 + 3)] += fc
                if printing or returning_energy:
                    angle_energy += angle_pots_to_use.gather(2, angle_bin_inds + 1).sum()

            # Dihedral forces.py
            # across_res is the number of atoms in the next residue, starting from atom_4
            for di, (atom_1, atom_2, atom_3, atom_4, across_res) in enumerate(dihedrals):
                ai_1, ai_2, ai_3, ai_4 = atoms.index(atom_1), atoms.index(atom_2), atoms.index(atom_3), atoms.index(atom_4)
                if across_res == 1:
                    ab = atom_coords[:, :-1, (ai_2 * 3):(ai_2 * 3 + 3)] - atom_coords[:, :-1, (ai_1 * 3):(ai_1 * 3 + 3)]
                    bc = atom_coords[:, :-1, (ai_3 * 3):(ai_3 * 3 + 3)] - atom_coords[:, :-1, (ai_2 * 3):(ai_2 * 3 + 3)]
                    cd = atom_coords[:, 1: , (ai_4 * 3):(ai_4 * 3 + 3)] - atom_coords[:, :-1, (ai_3 * 3):(ai_3 * 3 + 3)]
                    # Use residue potential according to central atom
                    dih_pots_to_use = dih_pots_flat[:, di, :-1]
                elif across_res == 2:
                    ab = atom_coords[:, :-1, (ai_2 * 3):(ai_2 * 3 + 3)] - atom_coords[:, :-1, (ai_1 * 3):(ai_1 * 3 + 3)]
                    bc = atom_coords[:, 1: , (ai_3 * 3):(ai_3 * 3 + 3)] - atom_coords[:, :-1, (ai_2 * 3):(ai_2 * 3 + 3)]
                    cd = atom_coords[:, 1: , (ai_4 * 3):(ai_4 * 3 + 3)] - atom_coords[:, 1: , (ai_3 * 3):(ai_3 * 3 + 3)]
                    dih_pots_to_use = dih_pots_flat[:, di, 1:]
                elif across_res == 3:
                    ab = atom_coords[:, 1: , (ai_2 * 3):(ai_2 * 3 + 3)] - atom_coords[:, :-1, (ai_1 * 3):(ai_1 * 3 + 3)]
                    bc = atom_coords[:, 1: , (ai_3 * 3):(ai_3 * 3 + 3)] - atom_coords[:, 1: , (ai_2 * 3):(ai_2 * 3 + 3)]
                    cd = atom_coords[:, 1: , (ai_4 * 3):(ai_4 * 3 + 3)] - atom_coords[:, 1: , (ai_3 * 3):(ai_3 * 3 + 3)]
                    dih_pots_to_use = dih_pots_flat[:, di, 1:]
                cross_ab_bc = torch.cross(ab, bc, dim=2)
                cross_bc_cd = torch.cross(bc, cd, dim=2)
                bc_norms = bc.norm(dim=2).unsqueeze(2)
                dihs = torch.atan2(
                    torch.sum(torch.cross(cross_ab_bc, cross_bc_cd, dim=2) * bc / bc_norms, dim=2),
                    torch.sum(cross_ab_bc * cross_bc_cd, dim=2)
                )
                dihs_from_centres = dih_centres_flat - dihs.unsqueeze(2)
                dih_bin_inds = dihs_from_centres.abs().argmin(dim=2).unsqueeze(2)
                dih_forces = 0.5 * (dih_pots_to_use.gather(2, dih_bin_inds) - dih_pots_to_use.gather(2, dih_bin_inds + 2))
                fa = dih_forces * normalize(-cross_ab_bc, dim=2) / ab.norm(dim=2).unsqueeze(2)
                fd = dih_forces * normalize( cross_bc_cd, dim=2) / cd.norm(dim=2).unsqueeze(2)
                # Forces on the middle atoms have to keep the sum of torques null
                # Forces taken from http://www.softberry.com/freedownloadhelp/moldyn/description.html
                fb = ((ab * -bc) / (bc_norms ** 2) - 1) * fa - ((cd * -bc) / (bc_norms ** 2)) * fd
                fc = -fa - fb - fd
                if across_res == 1:
                    atom_accs[:, :-1, (ai_1 * 3):(ai_1 * 3 + 3)] += fa
                    atom_accs[:, :-1, (ai_2 * 3):(ai_2 * 3 + 3)] += fb
                    atom_accs[:, :-1, (ai_3 * 3):(ai_3 * 3 + 3)] += fc
                    atom_accs[:, 1: , (ai_4 * 3):(ai_4 * 3 + 3)] += fd
                elif across_res == 2:
                    atom_accs[:, :-1, (ai_1 * 3):(ai_1 * 3 + 3)] += fa
                    atom_accs[:, :-1, (ai_2 * 3):(ai_2 * 3 + 3)] += fb
                    atom_accs[:, 1: , (ai_3 * 3):(ai_3 * 3 + 3)] += fc
                    atom_accs[:, 1: , (ai_4 * 3):(ai_4 * 3 + 3)] += fd
                elif across_res == 3:
                    atom_accs[:, :-1, (ai_1 * 3):(ai_1 * 3 + 3)] += fa
                    atom_accs[:, 1: , (ai_2 * 3):(ai_2 * 3 + 3)] += fb
                    atom_accs[:, 1: , (ai_3 * 3):(ai_3 * 3 + 3)] += fc
                    atom_accs[:, 1: , (ai_4 * 3):(ai_4 * 3 + 3)] += fd
                if printing or returning_energy:
                    dih_energy += dih_pots_to_use.gather(2, dih_bin_inds + 1).sum()

            accs += atom_accs.view(batch_size, n_atoms, 3) / masses.unsqueeze(2)

            # Shortcut to return energy at a given step
            if returning_energy:
                return dist_energy + angle_energy + dih_energy

            # Second step

            if integrator == "vel":
                vels = vels + 0.5 * (accs_last + accs) * timestep
                accs_last = accs
            elif integrator == "no_vel":
                coords_next = 2 * coords - coords_last + accs * timestep * timestep
                coords_last = coords
                coords = coords_next
            elif integrator == "langevin":
                # From Gronbech-Jensen 2013
                vels = vels + 0.5 * timestep * (accs_last + accs) - alpha * (coords - coords_last) / masses.unsqueeze(2) + beta / masses.unsqueeze(2)
                accs_last = accs
            elif integrator == "langevin_simple":
                gamma, twokbT = thermostat_const, temperature
                accs = accs + (-gamma * vels + np.sqrt(gamma * twokbT) * torch.randn(vels.shape, device=device)) / masses.unsqueeze(2)
                vels = vels + 0.5 * (accs_last + accs) * timestep
                accs_last = accs
            elif integrator == "min":
                coords = coords + accs * 0.1

            # Apply thermostat
            if integrator in ("vel", "no_vel") and thermostat_const > 0.0:
                thermostat_prob = timestep / thermostat_const
                for ai in range(n_atoms):
                    if random() < thermostat_prob:
                        if integrator == "vel":
                            # Actually this should be divided by the mass
                            new_vel = torch.randn(3, device=device) * temperature
                            vels[0, ai] = new_vel
                        elif integrator == "no_vel":
                            new_diff = torch.randn(3, device=device) * temperature * timestep
                            coords_last[0, ai] = coords[0, ai] - new_diff

            if printing:
                total_energy = dist_energy + angle_energy + dih_energy
                out_line = "    Step {:8} / {} - acc {:6.3f} {}- energy {:6.2f} ( {:6.2f} {:6.2f} {:6.2f} ) - Cα RMSD {:6.2f}".format(
                    i + 1, n_steps, torch.mean(accs.norm(dim=2)).item(),
                    "- vel {:6.3f} ".format(torch.mean(vels.norm(dim=2)).item()) if integrator in ("vel", "langevin", "langevin_simple") else "",
                    total_energy.item(), dist_energy.item(), angle_energy.item(), dih_energy.item(),
                    rmsd(coords.view(batch_size, n_res, 3 * len(atoms))[0, :, 3:6], native_coords_ca)[0].item())
                report(out_line, 2, verbosity)

            if sim_filepath and i % report_n == 0:
                model_n += 1
                with open(sim_filepath, "a") as of:
                    of.write("MODEL {:>8}\n".format(model_n))
                    for ri, r in enumerate(seq):
                        for ai, atom in enumerate(atoms):
                            of.write("ATOM   {:>4}  {:<2}  {:3} A{:>4}    {:>8.3f}{:>8.3f}{:>8.3f}  1.00  0.00          {:>2}  \n".format(
                                len(atoms) * ri + ai + 1, atom[:2].upper(),
                                one_to_three_aas[r], ri + 1,
                                coords[0, len(atoms) * ri + ai, 0].item(),
                                coords[0, len(atoms) * ri + ai, 1].item(),
                                coords[0, len(atoms) * ri + ai, 2].item(),
                                atom[0].upper()))
                    of.write("ENDMDL\n")

        return coords





def training_step(model_filepath, atom_ff_definitions, device="cpu", verbosity=0):
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

