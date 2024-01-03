import numpy as np
from multiprocessing import Process, Manager, Lock, cpu_count

class Simulation: # 
    def __init__(self, p_imax=50, p_jmax=50, p_xlength=1.0, p_ylength=1.0, p_t_end=50.0, p_tau=0.5, p_eps=1e-3,
                 p_omg=1.7, p_itermax=100, p_alpha=0.9, p_Re=100.0, p_t=0, p_dt=0.05, p_res=99999):
        self.imax = p_imax
        self.jmax = p_jmax
        self.xlength = p_xlength
        self.ylength = p_ylength
        self.t_end = p_t_end
        self.tau = p_tau
        self.eps = p_eps
        self.omg = p_omg
        self.itermax = p_itermax
        self.alpha = p_alpha
        self.Re = p_Re
        self.t = p_t
        self.dt = p_dt
        self.res = p_res

class StaggeredGrid:
    def __init__(self, p_imax=50, p_jmax=50, p_xlength=1.0, p_ylength=1.0):
        self.imax = p_imax
        self.jmax = p_jmax
        self.xlength = p_xlength
        self.ylength = p_ylength
        self.p = np.zeros((p_imax + 2, p_jmax + 2))
        self.po = np.zeros((p_imax + 2, p_jmax + 2))
        self.RHS = np.zeros((p_imax + 2, p_jmax + 2))
        self.u = np.zeros((p_imax + 2, p_jmax + 3))
        self.F = np.zeros((p_imax + 2, p_jmax + 3))
        self.v = np.zeros((p_imax + 3, p_jmax + 2))
        self.G = np.zeros((p_imax + 3, p_jmax + 2))

    def dx(self):
        return self.xlength / self.imax

    def dy(self):
        return self.ylength / self.jmax

    def find_max_absolute_u(self):
        return np.max(np.abs(self.u[1:self.imax + 1, 1:self.jmax + 2]))

    def find_max_absolute_v(self):
        return np.max(np.abs(self.v[1:self.imax + 2, 1:self.jmax + 1]))

def select_dt_according_to_stability_condition(grid, sim):
    left = (sim.Re / 2) * ((1 / grid.dx() ** 2) + (1 / grid.dy() ** 2)) ** -1
    middle = grid.dx() / grid.find_max_absolute_u()
    right = grid.dy() / grid.find_max_absolute_v()
    sim.dt = sim.tau * min(left, middle, right)

def set_boundary_conditions_u(grid):
    for j in range(grid.jmax + 3):
        grid.u[0, j] = 0.0
        grid.u[grid.imax, j] = 0.0

    for i in range(grid.imax + 2):
        grid.u[i, 0] = -grid.u[i, 1]
        grid.u[i, grid.jmax + 1] = -grid.u[i, grid.jmax]

    for i in range(grid.jmax + 2):
        grid.u[i, grid.jmax + 1] = 2.0 - grid.u[i, grid.jmax]

def set_boundary_conditions_v(grid):
    for j in range(grid.jmax + 2):
        grid.v[0, j] = -grid.v[1, j]
        grid.v[grid.imax + 1, j] = -grid.v[grid.imax, j]

    for i in range(grid.imax + 3):
        grid.v[i, 0] = 0.0
        grid.v[i, grid.jmax] = 0.0

def set_boundary_conditions_p(grid):
    for i in range(grid.imax + 2):
        grid.p[i, 0] = grid.p[i, 1]
        grid.p[i, grid.jmax + 1] = grid.p[i, grid.jmax]

    for j in range(grid.jmax + 2):
        grid.p[0, j] = grid.p[1, j]
        grid.p[grid.imax + 1, j] = grid.p[grid.imax, j]

# ME_X namespace
# Folien 5, Slide 15
def uu_x(grid, sim, i, j):
    return (
        (1 / grid.dx()) * ((0.5 * (grid.u[i, j] + grid.u[i + 1, j])) ** 2 - (0.5 * (grid.u[i - 1, j] + grid.u[i, j])) ** 2)
        + (sim.alpha / grid.dx())
        * (
            abs(0.5 * (grid.u[i, j] + grid.u[i + 1, j])) * (0.5 * (grid.u[i, j] - grid.u[i + 1, j])) / 4
            - abs(0.5 * (grid.u[i - 1, j] + grid.u[i, j])) * (0.5 * (grid.u[i - 1, j] - grid.u[i, j])) / 4
        )
    )

def uv_y(grid, sim, i, j):
    return (
        (1 / grid.dy()) * (
            (0.25 * (grid.v[i, j] + grid.v[i + 1, j]) * (grid.u[i, j] + grid.u[i, j + 1]))
            - (0.25 * (grid.v[i, j - 1] + grid.v[i + 1, j - 1]) * (grid.u[i, j - 1] + grid.u[i, j])) 
        )
        + (sim.alpha / grid.dy())
        * (
            abs(0.5 * (grid.v[i, j] + grid.v[i + 1, j])) * (0.5 * (grid.u[i, j] - grid.u[i, j + 1])) / 4
            - abs(0.5 * (grid.v[i, j - 1] + grid.v[i + 1, j - 1])) * (0.5 * (grid.u[i, j - 1] - grid.u[i, j])) / 4
        )
    )

# Folien 5, Slide 16
def uu_xx(grid, sim, i, j):
    return (grid.u[i + 1, j] - 2 * grid.u[i, j] + grid.u[i - 1, j]) / grid.dx() ** 2

def uu_yy(grid, sim, i, j):
    return (grid.u[i, j + 1] - 2 * grid.u[i, j] + grid.u[i, j - 1]) / grid.dy() ** 2

def p_x(grid, sim, i, j):
    return (grid.p[i + 1, j] - grid.p[i, j]) / grid.dx()

# ME_Y namespace
# Folien 5, Slide 17
def uv_x(grid, sim, i, j):
    return (
        (1 / grid.dx()) * (
            (0.25 * (grid.u[i, j] + grid.u[i, j + 1]) * (grid.v[i, j] + grid.v[i + 1, j]))
            - (0.25 * (grid.u[i - 1, j] + grid.u[i - 1, j + 1]) * (grid.v[i - 1, j] + grid.v[i, j]))
        )
        + (sim.alpha / grid.dx())
        * (
            abs(0.5 * (grid.u[i, j] + grid.u[i, j + 1])) * (0.5 * (grid.v[i, j] - grid.v[i + 1, j])) / 4
            - abs(0.5 * (grid.u[i - 1, j] + grid.u[i - 1, j + 1])) * (0.5 * (grid.v[i - 1, j] - grid.v[i, j])) / 4
        )
    )

def vv_y(grid, sim, i, j):
    return (
        (1 / grid.dy()) * ((0.5 * (grid.v[i, j] + grid.v[i, j + 1])) ** 2 - (0.5 * (grid.v[i, j - 1] + grid.v[i, j])) ** 2)
        + (sim.alpha / grid.dy())
        * (
            abs(0.5 * (grid.v[i, j] + grid.v[i, j + 1])) * (0.5 * (grid.v[i, j] - grid.v[i, j + 1])) / 4
            - abs(0.5 * (grid.v[i, j - 1] + grid.v[i, j])) * (0.5 * (grid.v[i, j - 1] - grid.v[i, j])) / 4
        )
    )

# Folien 5, Slide 18
def vv_xx(grid, sim, i, j):
    return (grid.v[i + 1, j] - 2 * grid.v[i, j] + grid.v[i - 1, j]) / grid.dx() ** 2

def vv_yy(grid, sim, i, j):
    return (grid.v[i, j + 1] - 2 * grid.v[i, j] + grid.v[i, j - 1]) / grid.dy() ** 2

def p_y(grid, sim, i, j):
    return (grid.p[i, j + 1] - grid.p[i, j]) / grid.dy()

# CE namespace
# Folien 5, Slide 19
def u_x(grid, sim, i, j):
    return (grid.u[i, j] - grid.u[i - 1, j]) / grid.dx()

def v_y(grid, sim, i, j):
    return (grid.v[i, j] - grid.v[i, j - 1]) / grid.dy()


def compute_f(grid, sim):
    for j in range(1, grid.jmax + 2):
        for i in range(1, grid.imax + 1):
            grid.F[i, j] = grid.u[i, j] + sim.dt * (
                (1 / sim.Re) * (uu_xx(grid, sim, i, j) + uu_yy(grid, sim, i, j))
                - uu_x(grid, sim, i, j)
                - uv_y(grid, sim, i, j)
            )

def compute_g(grid, sim):
    for i in range(1, grid.imax + 2):
        for j in range(1, grid.jmax + 1):
            grid.G[i, j] = grid.v[i, j] + sim.dt * (
                (1 / sim.Re) * (vv_xx(grid, sim, i, j) + vv_yy(grid, sim, i, j))
                - uv_x(grid, sim, i, j)
                - vv_y(grid, sim, i, j)
            )

def compute_rhs(grid, sim):
    for i in range(1, grid.imax + 1):
        for j in range(1, grid.jmax + 1):
            grid.RHS[i, j] = (1 / sim.dt) * (
                (grid.F[i, j] - grid.F[i - 1, j]) / grid.dx() + (grid.G[i, j] - grid.G[i, j - 1]) / grid.dy()
            )

def update_step_lgls(grid, sim):
    for i in range(1, grid.imax + 1):
        for j in range(1, grid.jmax + 1):
            grid.p[i, j] = (
                (1 / (-2 * grid.dx() ** 2 - 2 * grid.dy() ** 2))
                * (
                    grid.RHS[i, j] * grid.dx() ** 2 * grid.dy() ** 2
                    - grid.dy() ** 2 * (grid.p[i + 1, j] + grid.p[i - 1, j])
                    - grid.dx() ** 2 * (grid.p[i, j + 1] + grid.p[i, j - 1])
                )
            )

def compute_residual(grid, sim):
    sim.res = np.linalg.norm(grid.p - grid.po)

def compute_u(grid, sim):
    for i in range(1, grid.imax + 1):
        for j in range(1, grid.jmax + 2):
            grid.u[i, j] = grid.F[i, j] - (sim.dt / grid.dx()) * (grid.p[i + 1, j] - grid.p[i, j])

def compute_v(grid, sim):
    for i in range(1, grid.imax + 2):
        for j in range(1, grid.jmax + 1):
            grid.v[i, j] = grid.G[i, j] - (sim.dt / grid.dy()) * (grid.p[i, j + 1] - grid.p[i, j])

def save_matrix(filename, matrix):
    with open(filename, 'w') as file:
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                file.write(f"{matrix[i, j]:.5f} ")
            file.write("\n")
    print(f"Matrix saved to {filename}")

# Constants
imax = 50
jmax = 50
xlength = 1.0
ylength = 1.0
t_end = 5.0
tau = 0.5
eps = 1e-3
omg = 1.7
itermax = 100
alpha = 0.5
Re = 100.0

# Variables
t = 0
dt = 0.05
res = 99999

# Initialization
sim = Simulation(imax, jmax, xlength, ylength, t_end, tau, eps, omg, itermax, alpha, Re, t, dt, res)
grid = StaggeredGrid(imax, jmax, xlength, ylength)

n = 0

# ... (Ihr vorhandener Code)

def worker(grid, sim, n, lock):
    while sim.t < sim.t_end:
        lock.acquire()
        select_dt_according_to_stability_condition(grid, sim)
        print(f"t: {sim.t:.3f} dt: {sim.dt} res: {sim.res}")
        set_boundary_conditions_u(grid)
        set_boundary_conditions_v(grid)
        set_boundary_conditions_p(grid)
        compute_f(grid, sim)
        compute_g(grid, sim)
        compute_rhs(grid, sim)
        update_step_lgls(grid, sim)
        compute_residual(grid, sim)
        compute_u(grid, sim)
        compute_v(grid, sim)
        sim.t += sim.dt

        
        grid.u, grid.v, grid.p = u.copy(), v.copy(), p.copy()
        n.value += 1  # Use Value to share 'n' between processes
        lock.release()

if __name__ == '__main__':
    manager = Manager()
    lock = Lock()
    num_processes = cpu_count()
    print(f"Starting {num_processes} processes")

    # Use 'Value' to share 'n' between processes
    n = manager.Value('i', 0)

    p = Process(target=worker, args=(grid, sim, n, lock))
    p.start()
    p.join()

    # Matrizen nach dem Beenden des Arbeitsprozesses speichern
    save_matrix("u.dat_multiprocess", grid.u)
    save_matrix("v.dat_multiprocess", grid.v)
    save_matrix("p.dat_multiprocess", grid.p)

    print(f"Simulation finished after {n.value} iterations")