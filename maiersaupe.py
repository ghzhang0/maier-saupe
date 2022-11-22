from typing import Tuple
import numpy as np
import scipy.optimize as opt

end_seam_slope = {6: [2 * np.pi * 5/6,
                      2 * np.pi * 1/6], 4:[]}

class Patic():
    '''Class for a p-atic (p-fold rotationally symmetric liquid crystal).'''
    def __init__(self, J: float, p: int):
        self.J = J
        self.p = p

class Sector():
    '''Class for a lattice contained in a sector of the polar plane'''
    def __init__(self, r: float, alpha: float, coord_num: int,
                 boundary_condition: str):
        self.r = r
        self.alpha = alpha # sector angle = 2 * pi * alpha / coord_num
        self.coord_num = coord_num
        self.boundary_condition = boundary_condition
        self.lattice = None
        self.ap_ind = None
        self.cut_indices = None
        self.edge_indices = None
        self.bulk_indices = None
        self.start_seam_inds = None
        self.end_seam_inds = None

    def get_boundary_inds(self):
        '''Return indices of boundary sites (sites w/ < coord_num neighbors).'''
        cut_indices = []
        for row_ind, row in enumerate(self.A):
            if np.sum(row) < self.coord_num and row_ind != self.ap_ind:
                cut_indices.append(row_ind)
        self.cut_indices = np.array(cut_indices)

    def order_inds(self, spots_cone: np.ndarray):
        '''Sort lattice sites by angle and associated adjacency matrix.'''
        self.lattice = get_sorted_lattice(spots_cone)
        self.A = get_adjacency_matrix(self.lattice)

class Cone(Sector):
    '''Class for a lattice on an unrolled cone.'''
    def __init__(self, r: float, alpha: float, coord_num: int,
                 boundary_condition: str, sites_rec: np.ndarray):
        self.r = r
        self.alpha = alpha # sector angle = 2 * pi * alpha / coord_num
        self.coord_num = coord_num
        self.boundary_condition = boundary_condition
        self.lattice = None
        self.ap_ind = None
        self.cut_indices = None
        self.edge_indices = None
        self.bulk_indices = None
        self.start_seam_inds = None
        self.end_seam_inds = None
        sites = cut_sector(sites_rec, self.alpha, self.r)
        self.update_latt_inds(sites)

    def separate_inds(self):
        '''Separate indices of bulk, edge, seam, and apex sites.'''
        self.ap_ind = np.where(np.abs(self.lattice) == 0)[0][0]
        self.get_boundary_inds() # Get indices of boundary points
        self.separate_rim_and_seam()
        self.del_seam_int()
        self.del_ap_int()

    def separate_rim_and_seam(self):
        '''Separate edge indices from seam indices.'''
        rlen = int(self.r)
        start_seam_inds = self.cut_indices[:rlen] # Seam at phi = 0
        end_seam_inds = self.cut_indices[-rlen:] # Seam at phi = sector_angle
        self.start_seam_inds = start_seam_inds[
            np.argsort(self.lattice[start_seam_inds].real)
            ] # Order by increasing x

        if self.alpha in end_seam_slope[self.coord_num]:
            self.end_seam_inds = end_seam_inds[
                np.argsort(self.lattice[end_seam_inds].real)
                ] # Order by increasing x
        else:
            self.end_seam_inds = end_seam_inds[
                np.argsort(self.lattice[end_seam_inds].real)
                ][::-1] # Order by decreasing x
        # Last site on seam is also on edge
        self.edge_indices = np.append(self.cut_indices[rlen:-rlen],
                                      self.start_seam_inds[-1])
        if self.boundary_condition == 'tang':
            self.edge_seam_indices = np.concatenate(
                (self.edge_indices, self.end_seam_inds)
                ) # Indices of edge and end seam sites
        elif self.boundary_condition == 'free':
            self.edge_seam_indices = self.end_seam_inds
        else:
            raise ValueError("Undefined boundary condition.")
        self.bulk_indices = np.array(
            [i for i in np.arange(len(self.lattice))
             if i not in self.edge_seam_indices])

    def del_seam_int(self):
        '''Delete interactions between the two seams.'''
        start_seam = self.start_seam_inds
        for i_ind in range(len(start_seam)-1):
            self.A[start_seam[i_ind], start_seam[i_ind+1]] = 0
            self.A[start_seam[i_ind+1], start_seam[i_ind]] = 0

    def del_ap_int(self):
        '''Delete interactions with apex.'''
        self.A[self.ap_ind] = 0
        self.A[:, self.ap_ind] = 0

    def initialize_m0(self, seed_num: int) -> np.ndarray:
        '''Initialize random orientations with assigned boundary conditions.'''
        N = len(self.lattice)
        N_bulk = len(self.bulk_indices)
        np.random.seed(seed_num)
        m0 = np.random.rand(N)*2*np.pi
        if self.boundary_condition == 'tang':
            m0[self.edge_indices] = np.angle(
                self.lattice[self.edge_indices]
                )-np.pi/2
        m0[self.bulk_indices] = np.random.rand(N_bulk)*2*np.pi
        # Reinforces periodicity along phi at the seams
        m0[self.end_seam_inds] = m0[self.start_seam_inds]-(2*np.pi-self.alpha)
        return m0

    def get_energy(self, LC: Patic, m0: np.ndarray, mbulk: np.ndarray) -> float:
        '''Compute Maier-Saupe energy of LC on cone_latt.'''
        m0[self.bulk_indices] = mbulk
        m0[self.end_seam_inds] = m0[self.start_seam_inds]- (2*np.pi-self.alpha)
        ddtheta = np.array([m0[i] - m0 for i in np.arange(len(m0))])
        energy = -np.sum(LC.J*self.A*(np.cos(LC.p*(ddtheta))-1))
        return energy

    def update_latt_inds(self, spots_cone: np.ndarray):
        '''Order and separate lattice indices of spots_cone.'''
        self.order_inds(spots_cone)
        self.separate_inds()

    def get_ground_state_energy(self, LC: Patic,
                                seed_num: int) -> Tuple[float, np.ndarray]:
        '''Minimize Maier-Saupe energy of LC.'''
        m0 = self.initialize_m0(seed_num)
        energy_func = lambda mbulk: self.get_energy(LC, m0, mbulk)
        mbulk_res = opt.minimize(energy_func, m0[self.bulk_indices],
                                 method='BFGS').x
        m0[self.bulk_indices] = mbulk_res
        m0[self.end_seam_inds] = m0[self.start_seam_inds]- (2*np.pi-self.alpha)
        return m0, energy_func(mbulk_res)

class Disk(Cone):
    '''Class for a lattice on a disk.'''
    def __init__(self, r: float, coord_num: int, boundary_condition: str,
                 sites_rec: np.ndarray, cut: bool = False):
        self.r = r
        self.coord_num = coord_num
        self.boundary_condition = boundary_condition
        self.lattice = None
        self.ap_ind = None
        self.cut_indices = None
        self.edge_indices = None
        self.bulk_indices = None
        self.start_seam_inds = None
        self.end_seam_inds = None
        sites = cut_sector(sites_rec, 2 * np.pi, self.r)
        self.update_latt_inds(sites)
        self.ap_ind = np.where(np.abs(self.lattice) == 0)[0][0]
        if cut:
            self.add_seam()

    def separate_inds(self):
        '''Separate indices of bulk and edge sites.'''
        self.get_boundary_inds()
        if self.boundary_condition == 'tang':
            self.edge_indices = self.cut_indices
        elif self.boundary_condition == 'free':
            self.edge_indices = np.array([])
        self.bulk_indices = np.array(
            [i for i in np.arange(len(self.lattice))
             if i not in self.edge_indices])

    def add_seam(self):
        '''Add seam to disk at x = 0 and update adjancency matrix.'''
        start_seam_inds = np.where(((self.lattice.imag == 0)
                                    & (self.lattice.real > 0)))[0]
        self.start_seam_inds = start_seam_inds[
            np.argsort(self.lattice[start_seam_inds].real)]
        n = len(self.lattice)
        self.lattice = np.concatenate(
            (self.lattice, np.array([i + 0j for i in range(1, self.r + 1)])))
        self.end_seam_inds = np.arange(n, n + self.r)
        # Update adjancency matrix
        self.A = np.block([[self.A, np.zeros((n, self.r))],
                           [np.zeros((self.r, n)), np.zeros((self.r, self.r))]])
        for x, i in enumerate(self.start_seam_inds):
            for j in range(n):
                if self.A[i, j] > 0 and self.lattice[j].imag < 0:
                    self.A[i, j] = 0 # Delete original interaction
                    self.A[j, i] = 0
                    self.A[n + x, j] = 1
                    self.A[j, n + x] = 1
        self.del_ap_int()

    def del_ap_int(self):
        '''Delete interactions with apex.'''
        self.A[self.ap_ind] = 0
        self.A[:, self.ap_ind] = 0

    def initialize_m0(self, seed_num: int = 1) -> np.ndarray:
        '''Initialize random orientations with assigned boundary conditions.'''
        N = len(self.lattice)
        N_bulk = len(self.bulk_indices)
        np.random.seed(seed_num)
        m0 = np.random.rand(N)*2*np.pi
        if self.boundary_condition == 'tang':
            m0[self.edge_indices] = np.angle(
                self.lattice[self.edge_indices]
                )-np.pi/2
        m0[self.bulk_indices] = np.random.rand(N_bulk)*2*np.pi
        return m0

    def initialize_m0_disk_defect(self, dx: np.ndarray, dy: np.ndarray,
                                  p: int) -> np.ndarray:
        '''Initialize 1/p defects at [dx, dy] w/ assigned boundary condition.'''
        N = len(self.lattice)
        m0 = np.random.rand(N)*2*np.pi
        m0[self.edge_indices] = np.angle(
            self.lattice[self.edge_indices]
            )-np.pi/2
        dr = np.array([np.arctan2((self.lattice.real-dx[i]),
                                  (self.lattice.imag-dy[i]))
                       for i in range(len(dx))])
        m0_d = np.einsum('ij->j', dr)%(2*np.pi)*(-1/p)
        m0[self.bulk_indices] = m0_d[self.bulk_indices]
        return m0

class Hyperbolic(Cone):
    '''Class for a lattice on an unrolled hyperbolic cone.'''
    def __init__(self, bottom_layer: Disk, top_layer: Cone):
        self.r = top_layer.r
        self.alpha = top_layer.alpha + 2 * np.pi
        self.coord_num = top_layer.coord_num
        self.boundary_condition = top_layer.boundary_condition
        self.top = top_layer
        self.bottom = bottom_layer
        self.bulk_indices = np.concatenate(
            (np.array(self.top.bulk_indices),
             len(self.top.lattice) + np.array(self.bottom.bulk_indices))
            )
        self.lattice = np.concatenate((self.top.lattice, self.bottom.lattice))
        self.A = None
        self.get_A()

    def get_A(self):
        '''Get adjancency matrix for hyperbolic cone.'''
        n = len(self.top.A)
        m = len(self.bottom.A)
        self.A = np.block([[self.top.A, np.zeros((n, m))], \
             [np.zeros((m, n)),   self.bottom.A]])

    def initialize_m0(self, seed_num: int = 1) -> np.ndarray:
        '''Initialize random orientations with assigned boundary condition.'''
        m0_top = self.top.initialize_m0(seed_num)
        m0_bottom = self.bottom.initialize_m0(seed_num)
        m0 = np.concatenate((m0_top, m0_bottom))
        return m0

    def get_energy(self, LC: Patic, m0: np.ndarray, mbulk: np.ndarray) -> float:
        '''Compute Maier-Saupe energy of LC on cone_latt.'''
        n = len(self.top.A)
        m0[self.top.bulk_indices] = mbulk[:len(self.top.bulk_indices)]
        m0[n + self.bottom.bulk_indices] = mbulk[len(self.top.bulk_indices):]
        m0[self.top.end_seam_inds] = (m0[n + self.bottom.start_seam_inds] 
                                      - (2*np.pi-self.top.alpha))
        m0[n + self.bottom.end_seam_inds] = m0[self.top.start_seam_inds]
        ddtheta = np.array([m0[i] - m0 for i in np.arange(len(m0))])
        energy = -np.sum(LC.J*self.A*(np.cos(LC.p*(ddtheta))-1))
        return energy

    def get_ground_state_energy(self, LC: Patic, 
                                seed_num: int = 1) -> Tuple[float, np.ndarray]:
        '''Minimize Maier-Saupe energy of LC.'''
        n = len(self.top.A)
        m0 = self.initialize_m0(seed_num)
        energy_func = lambda mbulk: self.get_energy(LC, m0, mbulk)
        mbulk_res = opt.minimize(energy_func, m0[self.bulk_indices],
                                 method='BFGS').x
        nbulk = len(self.top.bulk_indices)
        m0[self.top.bulk_indices] = mbulk_res[:nbulk]
        m0[n + self.bottom.bulk_indices] = mbulk_res[nbulk:]
        m0[self.top.end_seam_inds] = (m0[n + self.bottom.start_seam_inds]
                                      - (2*np.pi-self.top.alpha))
        m0[n + self.bottom.end_seam_inds] = m0[self.top.start_seam_inds]
        E0 = energy_func(m0[self.bulk_indices])
        return m0, E0

def prepare_lattice(coord_num: int, size: float = 80.) -> np.ndarray:
    '''Get sites of a 80x80 sized lattice with assigned coordination number.'''
    if coord_num == 6:
        lattice_vectors = np.array([[-1, 0.], [1/2., -np.sqrt(3)/2]])
    elif coord_num == 4:
        lattice_vectors = np.array([[1, 0.], [0, 1]])
    else:
        raise ValueError("Undefined coordination number.")
    image_shape = (size, size)
    sites = generate_lattice(image_shape, lattice_vectors)
    return sites

def generate_lattice(image_shape: tuple, lattice_vectors: np.ndarray,
                     edge_buffer: float = 1.) -> np.ndarray:
    '''Generate lattice points in given rectangular domain.'''
    num_vectors = int( ##Estimate how many lattice points we need
        max(image_shape) / np.sqrt(lattice_vectors[0]**2).sum())
    lattice_pts = []
    lower_bounds = np.array((edge_buffer, edge_buffer))
    upper_bounds = np.array(image_shape) - edge_buffer

    for i in range(-num_vectors, num_vectors):
        for j in range(-num_vectors, num_vectors):
            lp = (i * lattice_vectors[0] + j * lattice_vectors[1] 
                  + np.array(image_shape) // 2)
            if all(lower_bounds < lp) and all(lp < upper_bounds):
                lattice_pts.append(lp)
    lattice_pts = np.array(lattice_pts)
    # Center lattice at (0,0)
    lattice_pts[:, 0] = lattice_pts[:, 0] - np.mean(lattice_pts[:, 0])
    lattice_pts[:, 1] = lattice_pts[:, 1] - np.mean(lattice_pts[:, 1])
    return lattice_pts

def in_circ(pos: list, r: float) -> bool:
    '''Return Boolean mask for points inside a circle of radius r.'''
    x, y = map(abs, pos)
    return y**2 <= (r**2 - x**2)

def in_cone(pos: list, sec_angle: float) -> bool:
    '''Return boolean mask for points inside a sector with angle sec_angle.'''
    x = np.round(pos[0], 5)
    y = np.round(pos[1], 5)
    if np.arctan2(y, x) % (2 * np.pi) <= (sec_angle * 1.001):
        return True
    else:
        return False

def cut_sector(spots_hex: np.ndarray, sec_angle: float, r: float) -> np.ndarray:
    '''Cut sector from a rectangular sheet of triangular lattice. '''
    spots_cone = []
    for i in range(len(spots_hex)):
        pt_x = spots_hex[i][0]
        pt_y = spots_hex[i][1]
        if in_cone([pt_x, pt_y], sec_angle) and in_circ([pt_x, pt_y], r):
            spots_cone.append(spots_hex[i])
    return np.array(spots_cone)

def get_sorted_lattice(spots_cone: np.ndarray) -> np.ndarray:
    '''Sort lattice points by angular coordinate. '''
    sites = np.array(spots_cone).T[0]+1j*np.array(spots_cone).T[1]
    inds = np.argsort(np.arctan2(sites.imag, sites.real)%(2*np.pi))
    sorted_sites = np.array(sites)[inds].T
    return sorted_sites

def get_adjacency_matrix(lattice: np.ndarray) -> np.ndarray:
    '''Get adjacency matrix of given lattice.'''
    N = len(lattice)
    A = np.zeros((N, N))
    for i, ri in enumerate(lattice):
        for j, rj  in enumerate(lattice):
            if abs(ri - rj) < 1.2 and i != j:
                A[i, j] = 1 
    return A

def get_neighbors_list(A: np.ndarray) -> list:
    '''Given an adjacency matrix A, return a list of neighbors.'''
    N = len(A)
    neighbors = [[] for i in range(N)]
    for i in range(N):
        for j in range(N):
            if A[i, j] == 1:
                neighbors[i].append(j)
                neighbors[j].append(i)
    return neighbors
    