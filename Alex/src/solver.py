import numpy as np
from mpi4py import MPI
from numba import jit

class Alex_Louis_Solver:
    def __init__(self, MPIcomm, rank, rankShift, coords, gridSize, dt, t_end, tau, eps, omg, itermax, alpha, dx, dy, subPos, Re):
        self.MPIcomm = MPIcomm
        self.rank = rank

        self.coords = np.array(coords)
        self.rankShift = np.array(rankShift)

        self.Nx, self.Ny = gridSize
        self.dx, self.dy = dx, dy

        self.subPos = np.array(subPos)
        self.tEnd, self.Re, self.dt = t_end, Re, dt

        self.U = 1.0  # Assuming a constant value for lid velocity

        self.coeff = np.zeros(3)        
        
        self.tau = tau
        self.eps = eps
        self.omg = omg
        self.itermax = itermax
        self.alpha = alpha
        
        # coords vs. subPos
        print(f'Rank {self.rank}\t | Coords: {self.coords}\t | Subgrid Position: ({self.subPos[0]:.4f},{self.subPos[1]:.4f}) \t | Rank Shift: {self.rankShift}')

        self.poissonSolver = None
        
    def initialize(self):
        self.coeff[0] = -1.0 / (self.dy ** 2)                   # Coefficient for the y-direction (laplacian)
        self.coeff[2] = -1.0 / (self.dx ** 2)                   # Coefficient for the x-direction (laplacian)
        self.coeff[1] = -2.0 * (self.coeff[0] + self.coeff[2])  # Coefficient for the center (laplacian)

        if self.rankShift[1] == -2:
            for i in range(1, self.Nx):
                self.velU[i * self.Ny - 1] = self.U
        
        self.p = np.zeros((self.Nx + 2, self.Ny + 2))
        self.po = np.zeros((self.Nx + 2, self.Ny + 2))
        self.RHS = np.zeros((self.Nx + 2, self.Ny + 2))
        self.u = np.zeros((self.Nx + 2, self.Ny + 3))
        self.F = np.zeros((self.Nx + 2, self.Ny + 3))
        self.v = np.zeros((self.Nx + 3, self.Ny + 2))
        self.G = np.zeros((self.Nx + 3, self.Ny + 2))

    def solve(self):
        t = 0.0
        res = 9999
        while t < self.tEnd:
            n = 0
            set_boundary_conditions(self.u, self.v, self.Ny, self.Nx)
            compute_f(self.Re, self.F, self.u, self.v, self.dx, self.dy, self.dt, self.Nx, self.Ny, self.alpha)
            compute_g(self.Re, self.G, self.u, self.v, self.dx, self.dy, self.dt, self.Nx, self.Ny, self.alpha)
            compute_rhs(self.RHS, self.F, self.G, self.dx, self.dy, self.dt, self.Nx, self.Ny)
            while (res > self.eps or res == 0) and n < self.itermax:
                set_boundary_conditions_p(self.p, self.Ny, self.Nx)
                update_step_lgls(self.RHS, self.p, self.dx, self.dy, self.Nx, self.Ny)
                res = compute_residual(self.p, self.po)
                n += 1
            compute_u(self.u, self.F, self.p, self.dx, self.dt, self.Nx, self.Ny)
            compute_v(self.v, self.G, self.p, self.dy, self.dt, self.Nx, self.Ny)
            self.po = self.p
            t += self.dt
            self.dt = select_dt_according_to_stability_condition(self.Re, self.dx, self.dy, self.tau, self.u, self.v, self.Nx, self.Ny)
            

def find_max_absolute_u(u, imax, jmax):
    return np.max(np.abs(u[1:imax + 1, 1:jmax + 2]))

def find_max_absolute_v(v, imax, jmax):
    return np.max(np.abs(v[1:imax + 2, 1:jmax + 1]))

def select_dt_according_to_stability_condition(Re, dx, dy, tau, u, v, imax, jmax):
    left = (Re / 2) * ((1 / dx ** 2) + (1 / dy ** 2)) ** -1
    middle = dx / find_max_absolute_u(u, imax, jmax)
    right = dy / find_max_absolute_v(v, imax, jmax)
    return tau * min(left, middle, right)

@jit(fastmath=True, nopython=True)
def set_boundary_conditions_u(u, jmax, imax):
    for j in range(jmax + 3):
        u[0, j] = 0.0
        u[imax, j] = 0.0

    for i in range(imax + 2):
        u[i, 0] = -u[i, 1]
        u[i, jmax + 1] = -u[i, jmax]

    for i in range(jmax + 2):
        u[i, jmax + 1] = 2.0 - u[i, jmax]

@jit(fastmath=True, nopython=True)
def set_boundary_conditions_v(v, jmax, imax):
    for j in range(jmax + 2):
        v[0, j] = -v[1, j]
        v[imax + 1, j] = -v[imax, j]

    for i in range(imax + 3):
        v[i, 0] = 0.0
        v[i, jmax] = 0.0
@jit(fastmath=True, nopython=True)
def set_boundary_conditions(u, v, jmax, imax):
    # Set boundary conditions for u
    for j in range(jmax + 3):
        u[0, j] = 0.0
        u[imax, j] = 0.0

    for i in range(imax + 2):
        u[i, 0] = -u[i, 1]
        u[i, jmax + 1] = -u[i, jmax]

    for i in range(jmax + 2):
        u[i, jmax + 1] = 2.0 - u[i, jmax]

    # Set boundary conditions for v
    for j in range(jmax + 2):
        v[0, j] = -v[1, j]
        v[imax + 1, j] = -v[imax, j]

    for i in range(imax + 3):
        v[i, 0] = 0.0
        v[i, jmax] = 0.0


@jit(fastmath=True, nopython=True)
def set_boundary_conditions_p(p, jmax, imax):
    for i in range(imax + 2):
        p[i, 0] = p[i, 1]
        p[i, jmax + 1] = p[i, jmax]

    for j in range(jmax + 2):
        p[0, j] = p[1, j]
        p[imax + 1, j] = p[imax, j]

# ME_X namespace
# Folien 5, Slide 15
@jit(fastmath=True, nopython=True)
def uu_x(u, dx, i, j, alpha):
    return (
        (1 / dx) * ((0.5 * (u[i, j] + u[i + 1, j])) ** 2 - (0.5 * (u[i - 1, j] + u[i, j])) ** 2)
        + (alpha / dx)
        * (
            abs(0.5 * (u[i, j] + u[i + 1, j])) * (0.5 * (u[i, j] - u[i + 1, j])) / 4
            - abs(0.5 * (u[i - 1, j] + u[i, j])) * (0.5 * (u[i - 1, j] - u[i, j])) / 4
        )
    )

@jit(fastmath=True, nopython=True)
def uv_y(u, v, dy, i, j, alpha):
    return (
        (1 / dy) * (
            (0.25 * (v[i, j] + v[i + 1, j]) * (u[i, j] + u[i, j + 1]))
            - (0.25 * (v[i, j - 1] + v[i + 1, j - 1]) * (u[i, j - 1] + u[i, j])) 
        )
        + (alpha / dy)
        * (
            abs(0.5 * (v[i, j] + v[i + 1, j])) * (0.5 * (u[i, j] - u[i, j + 1])) / 4
            - abs(0.5 * (v[i, j - 1] + v[i + 1, j - 1])) * (0.5 * (u[i, j - 1] - u[i, j])) / 4
        )
    )

@jit(fastmath=True, nopython=True)
# Folien 5, Slide 16
def uu_xx(u, dx, i, j):
    return (u[i + 1, j] - 2 * u[i, j] + u[i - 1, j]) / dx ** 2

@jit(fastmath=True, nopython=True)
def uu_yy(u, dy, i, j):
    return (u[i, j + 1] - 2 * u[i, j] + u[i, j - 1]) / dy ** 2

@jit(fastmath=True, nopython=True)
def p_x(p, dx, i, j):
    return (p[i + 1, j] - p[i, j]) / dx

# ME_Y namespace
# Folien 5, Slide 17
@jit(fastmath=True, nopython=True)
def uv_x(u, v, dx, i, j, alpha):
    return (
        (1 / dx) * (
            (0.25 * (u[i, j] + u[i, j + 1]) * (v[i, j] + v[i + 1, j]))
            - (0.25 * (u[i - 1, j] + u[i - 1, j + 1]) * (v[i - 1, j] + v[i, j]))
        )
        + (alpha / dx)
        * (
            abs(0.5 * (u[i, j] + u[i, j + 1])) * (0.5 * (v[i, j] - v[i + 1, j])) / 4
            - abs(0.5 * (u[i - 1, j] + u[i - 1, j + 1])) * (0.5 * (v[i - 1, j] - v[i, j])) / 4
        )
    )

@jit(fastmath=True, nopython=True)
def vv_y(v, dy, i, j, alpha):
    return (
        (1 / dy) * ((0.5 * (v[i, j] + v[i, j + 1])) ** 2 - (0.5 * (v[i, j - 1] + v[i, j])) ** 2)
        + (alpha / dy)
        * (
            abs(0.5 * (v[i, j] + v[i, j + 1])) * (0.5 * (v[i, j] - v[i, j + 1])) / 4
            - abs(0.5 * (v[i, j - 1] + v[i, j])) * (0.5 * (v[i, j - 1] - v[i, j])) / 4
        )
    )

# Folien 5, Slide 18
@jit(fastmath=True, nopython=True)
def vv_xx(v, dx, i, j):
    return (v[i + 1, j] - 2 * v[i, j] + v[i - 1, j]) / dx ** 2

@jit(fastmath=True, nopython=True)
def vv_yy(v, dy, i, j):
    return (v[i, j + 1] - 2 * v[i, j] + v[i, j - 1]) / dy ** 2

@jit(fastmath=True, nopython=True)
def p_y(p, dy, i, j):
    return (p[i, j + 1] - p[i, j]) / dy

# CE namespace
# Folien 5, Slide 19
@jit(fastmath=True, nopython=True)
def u_x(u, dx, i, j):
    return (u[i, j] - u[i - 1, j]) / dx

@jit(fastmath=True, nopython=True)
def v_y(v, dy, i, j):
    return (v[i, j] - v[i, j - 1]) / dy


@jit(fastmath=True, nopython=True)
def compute_f(Re, F, u, v, dx, dy, dt, imax, jmax, alpha):
    for j in range(1, jmax + 2):
        for i in range(1, imax + 1):
            F[i, j] = u[i, j] + dt * (
                (1 / Re) * (uu_xx(u, dx, i, j) + uu_yy(u, dy, i, j))
                - uu_x(u, dx, i, j, alpha)
                - uv_y(u, v, dy, i, j, alpha)
            )

@jit(fastmath=True, nopython=True)
def compute_g(Re, G, u, v, dx, dy, dt, imax, jmax, alpha):
    for i in range(1, imax + 2):
        for j in range(1, jmax + 1):
            G[i, j] = v[i, j] + dt * (
                (1 / Re) * (vv_xx(v, dx, i, j) + vv_yy(v, dy, i, j))
                - uv_x(u, v, dx, i, j, alpha)
                - vv_y(v, dy, i, j, alpha)
            )

@jit(fastmath=True, nopython=True)
def compute_rhs(RHS, F, G, dx, dy, dt, imax, jmax):
    for i in range(1, imax + 1):
        for j in range(1, jmax + 1):
            RHS[i, j] = (1 / dt) * (
                (F[i, j] - F[i - 1, j]) / dx + 
                (G[i, j] - G[i, j - 1]) / dy
            )

@jit(fastmath=True, nopython=True)
def update_step_lgls(RHS, p, dx, dy, imax, jmax):
    for i in range(1, imax + 1):
        for j in range(1, jmax + 1):
            p[i, j] = (
                (1 / (-2 * dx ** 2 - 2 * dy ** 2))
                * (
                    RHS[i, j] * dx ** 2 * dy ** 2
                    - dy ** 2 * (p[i + 1, j] + p[i - 1, j])
                    - dx ** 2 * (p[i, j + 1] + p[i, j - 1])
                )
            )

@jit(fastmath=True, nopython=True)
def compute_residual(p, po):
    return np.linalg.norm(p - po)

@jit(fastmath=True, nopython=True)
def compute_u(u, F, p, dx, dt, imax, jmax):
    for i in range(1, imax + 1):
        for j in range(1, jmax + 2):
            u[i, j] = F[i, j] - (dt / dx) * (p[i + 1, j] - p[i, j])

@jit(fastmath=True, nopython=True)
def compute_v(v, G, p, dy, dt, imax, jmax):
    for i in range(1, imax + 2):
        for j in range(1, jmax + 1):
            v[i, j] = G[i, j] - (dt / dy) * (p[i, j + 1] - p[i, j])