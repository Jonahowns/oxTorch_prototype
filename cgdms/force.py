import torch



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

    def init_nonbonded_forces(self, distances, cutoff=1.0):
        for x in self.nonbonded_forces:
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
        angle_forces = torch.zeros((vectors_mc.shape[0], 3))  # Per site forces.py from angles
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
        pair_forces = torch.zeros((self.M, 3))  # per particle forces.py

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

