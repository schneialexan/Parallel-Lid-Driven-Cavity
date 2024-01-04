import numpy as np
from numba import jit
from mpi4py import MPI
from typing import Tuple
import os

def find_max_absolute_u(u, imax, jmax):
    return np.max(np.abs(u[1:imax + 1, 1:jmax + 2]))

def find_max_absolute_v(v, imax, jmax):
    return np.max(np.abs(v[1:imax + 2, 1:jmax + 1]))

def select_dt_according_to_stability_condition(Re, dx, dy, tau, u, v, imax, jmax):
    left = (Re / 2) * ((1 / dx ** 2) + (1 / dy ** 2)) ** -1
    middle = dx / find_max_absolute_u(u, imax, jmax)
    right = dy / find_max_absolute_v(v, imax, jmax)
    return tau * min(left, middle, right)

# @jit(fastmath=True, nopython=True)
def set_boundary_conditions_u(u, jmax, imax):
    for j in range(jmax + 3):
        u[0, j] = 0.0
        u[imax, j] = 0.0

    for i in range(imax + 2):
        u[i, 0] = -u[i, 1]
        u[i, jmax + 1] = -u[i, jmax]

    for i in range(jmax + 2):
        u[i, jmax + 1] = 2.0 - u[i, jmax]

# @jit(fastmath=True, nopython=True)
def set_boundary_conditions_v(v, jmax, imax):
    for j in range(jmax + 2):
        v[0, j] = -v[1, j]
        v[imax + 1, j] = -v[imax, j]

    for i in range(imax + 3):
        v[i, 0] = 0.0
        v[i, jmax] = 0.0
# @jit(fastmath=True, nopython=True)      
def set_boundary_conditions(u, v, jmax, imax):
    jmax -= 1
    imax -= 1
    # Set boundary conditions for u
    for j in range(jmax + 2):
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

    for i in range(imax + 2):
        v[i, 0] = 0.0
        v[i, jmax] = 0.0


# @jit(fastmath=True, nopython=True)
def set_boundary_conditions_p(p, jmax, imax):
    for i in range(imax + 2):
        p[i, 0] = p[i, 1]
        p[i, jmax + 1] = p[i, jmax]

    for j in range(jmax + 2):
        p[0, j] = p[1, j]
        p[imax + 1, j] = p[imax, j]

# ME_X namespace
# Folien 5, Slide 15
# @jit(fastmath=True, nopython=True)
def uu_x(u, dx, i, j, alpha):
    return (
        (1 / dx) * ((0.5 * (u[i, j] + u[i + 1, j])) ** 2 - (0.5 * (u[i - 1, j] + u[i, j])) ** 2)
        + (alpha / dx)
        * (
            abs(0.5 * (u[i, j] + u[i + 1, j])) * (0.5 * (u[i, j] - u[i + 1, j])) / 4
            - abs(0.5 * (u[i - 1, j] + u[i, j])) * (0.5 * (u[i - 1, j] - u[i, j])) / 4
        )
    )

# @jit(fastmath=True, nopython=True)
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

# @jit(fastmath=True, nopython=True)
# Folien 5, Slide 16
def uu_xx(u, dx, i, j):
    return (u[i + 1, j] - 2 * u[i, j] + u[i - 1, j]) / dx ** 2

# @jit(fastmath=True, nopython=True)
def uu_yy(u, dy, i, j):
    return (u[i, j + 1] - 2 * u[i, j] + u[i, j - 1]) / dy ** 2

# @jit(fastmath=True, nopython=True)
def p_x(p, dx, i, j):
    return (p[i + 1, j] - p[i, j]) / dx

# ME_Y namespace
# Folien 5, Slide 17
# @jit(fastmath=True, nopython=True)
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

# @jit(fastmath=True, nopython=True)
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
# @jit(fastmath=True, nopython=True)
def vv_xx(v, dx, i, j):
    return (v[i + 1, j] - 2 * v[i, j] + v[i - 1, j]) / dx ** 2

# @jit(fastmath=True, nopython=True)
def vv_yy(v, dy, i, j):
    return (v[i, j + 1] - 2 * v[i, j] + v[i, j - 1]) / dy ** 2

# @jit(fastmath=True, nopython=True)
def p_y(p, dy, i, j):
    return (p[i, j + 1] - p[i, j]) / dy

# CE namespace
# Folien 5, Slide 19
# @jit(fastmath=True, nopython=True)
def u_x(u, dx, i, j):
    return (u[i, j] - u[i - 1, j]) / dx

# @jit(fastmath=True, nopython=True)
def v_y(v, dy, i, j):
    return (v[i, j] - v[i, j - 1]) / dy


# @jit(fastmath=True, nopython=True)
def compute_f(Re, F, u, v, dx, dy, dt, imax, jmax, alpha):
    for j in range(1, jmax):
        for i in range(1, imax):
            F[i, j] = u[i, j] + dt * (
                (1 / Re) * (uu_xx(u, dx, i, j) + uu_yy(u, dy, i, j))
                - uu_x(u, dx, i, j, alpha)
                - uv_y(u, v, dy, i, j, alpha)
            )

# @jit(fastmath=True, nopython=True)
def compute_g(Re, G, u, v, dx, dy, dt, imax, jmax, alpha):
    for i in range(1, imax):
        for j in range(1, jmax):
            G[i, j] = v[i, j] + dt * (
                (1 / Re) * (vv_xx(v, dx, i, j) + vv_yy(v, dy, i, j))
                - uv_x(u, v, dx, i, j, alpha)
                - vv_y(v, dy, i, j, alpha)
            )

# @jit(fastmath=True, nopython=True)
def compute_rhs(RHS, F, G, dx, dy, dt, imax, jmax):
    for i in range(1, imax + 1):
        for j in range(1, jmax + 1):
            RHS[i, j] = (1 / dt) * (
                (F[i, j] - F[i - 1, j]) / dx + 
                (G[i, j] - G[i, j - 1]) / dy
            )

# @jit(fastmath=True, nopython=True)
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

# @jit(fastmath=True, nopython=True)
def compute_residual(p, po):
    return np.linalg.norm(p - po)

# @jit(fastmath=True, nopython=True)
def compute_u(u, F, p, dx, dt, imax, jmax):
    for i in range(1, imax + 1):
        for j in range(1, jmax + 2):
            u[i, j] = F[i, j] - (dt / dx) * (p[i + 1, j] - p[i, j])

# @jit(fastmath=True, nopython=True)
def compute_v(v, G, p, dy, dt, imax, jmax):
    for i in range(1, imax + 2):
        for j in range(1, jmax + 1):
            v[i, j] = G[i, j] - (dt / dy) * (p[i, j + 1] - p[i, j])

def save_matrix(filename, matrix):
    with open(filename, 'w') as file:
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                file.write(f"{matrix[i, j]:.5f} ")
            file.write("\n")
    print(f"Matrix saved to {filename}")
    
def split_grid(grid_size: Tuple[int, int], partition_size: Tuple[int, int], coords: Tuple[int, int],
               dx: float, dy: float) -> Tuple[Tuple[int, int], Tuple[float, float]]:
    # Splits the grid into the specified number of subdomains, distributing nodes as equally as possible
    sub_nx_quotient, sub_nx_remainder = divmod(grid_size[1], partition_size[1])
    sub_ny_quotient, sub_ny_remainder = divmod(grid_size[0], partition_size[0])

    if coords[1] < sub_nx_remainder:
        sub_grid_size_x = sub_nx_quotient + 1
        sub_pos_x = dx * sub_grid_size_x * coords[1]
    else:
        sub_grid_size_x = sub_nx_quotient
        sub_pos_x = dx * ((sub_grid_size_x + 1) * sub_nx_remainder +
                          sub_grid_size_x * (coords[1] - sub_nx_remainder))

    if coords[0] < sub_ny_remainder:
        sub_grid_size_y = sub_ny_quotient + 1
        sub_pos_y = dy * sub_grid_size_y * coords[0]
    else:
        sub_grid_size_y = sub_ny_quotient
        sub_pos_y = dy * ((sub_grid_size_y + 1) * sub_ny_remainder +
                          sub_grid_size_y * (coords[0] - sub_ny_remainder))

    return (sub_grid_size_x, sub_grid_size_y), (sub_pos_x, sub_pos_y)


# MPI Initialisierung
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Constants
imax = 50
jmax = 50
xlength = 1.0
ylength = 1.0
dx = xlength / imax
dy = ylength / jmax
t_end = 5.0
tau = 0.5
eps = 1e-3
omg = 1.7
itermax = 100
alpha = 0.5
Re = 100.0

partitionSize = (1, 1) if comm.size == 1 else (2, 2) # überprüft ob 1 oder 4 Prozesse verwendet werden

# Variables
t = 0
dt = 0.005
res = 99999

# Full Simulation Arrays
p = np.zeros((imax + 2, jmax + 2))
po = np.zeros((imax + 2, jmax + 2))
RHS = np.zeros((imax + 2, jmax + 2))
u = np.zeros((imax + 2, jmax + 2))
F = np.zeros((imax + 2, jmax + 2))
v = np.zeros((imax + 2, jmax + 2))
G = np.zeros((imax + 2, jmax + 2))

cartGrid =  comm.Create_cart(dims=partitionSize, periods=[False, False])
rankShift = cartGrid.Shift(0, 1)
rankShift += cartGrid.Shift(1, 1)
subgrid_size, subgrid_pos = split_grid((imax, jmax), partitionSize, cartGrid.Get_coords(rank), dx, dy)


append_x = 0
append_y = 0
if rankShift[0] != -1:
    append_x += 1
if rankShift[1] != -1:
    append_x += 1
if rankShift[2] != -1:
    append_y += 1
if rankShift[3] != -1:
    append_y += 1

print(f'Rank {rank}\t | Subgrid Size: {subgrid_size}\t | Shift x: {append_x}\t | Shift y: {append_y}')
comm.Barrier()

import time
tStart = time.time()

# Create local variables for each process
local_u = np.zeros((subgrid_size[0] + append_x, subgrid_size[1] + append_y))
local_v = np.zeros((subgrid_size[0] + append_x, subgrid_size[1] + append_y))
local_F = np.zeros((subgrid_size[0] + append_x, subgrid_size[1] + append_y))
local_G = np.zeros((subgrid_size[0] + append_x, subgrid_size[1] + append_y))
local_RHS = np.zeros((subgrid_size[0] + append_x, subgrid_size[1] + append_y))

while t < t_end:
    n = 0
    print(f"rank: {rank}, t: {t:.3f} dt: {dt} res: {res} - size: {subgrid_size}")
    
    # Distribute u and v data to each process
    comm.Scatter(u, local_u, root=0)
    comm.Scatter(v, local_v, root=0)

    # Set boundary conditions for local_u and local_v
    set_boundary_conditions(local_u, local_v, subgrid_size[1], subgrid_size[0])
    
    # Compute f and g for local_u and local_v
    compute_f(Re, local_F, local_u, local_v, dx, dy, dt, subgrid_size[0], subgrid_size[1], alpha)
    compute_g(Re, local_G, local_u, local_v, dx, dy, dt, subgrid_size[0], subgrid_size[1], alpha)
    

    # Gather f and g from each process to root (process 0)
    comm.Gather(local_F, F, root=0)
    comm.Gather(local_G, G, root=0)

    # Compute RHS using the gathered f and g
    if rank == 0:
        compute_rhs(RHS, F, G, dx, dy, dt, imax, jmax)

    # Scatter the RHS data to each process
    comm.Scatter(RHS, local_RHS, root=0)

    # Iteratively solve for pressure using MPI
    while (res > eps or res == 0) and n < itermax:
        set_boundary_conditions_p(p, jmax, imax)

        # MPI-Berechnung der Teilsummen
        local_sum = np.sum(local_RHS[1:-1, 1:-1] * dx ** 2 * dy ** 2)

        # MPI-Reduktion: Summieren aller Teilsummen über alle Prozesse
        global_sum = comm.allreduce(local_sum, op=MPI.SUM)

        # MPI-Broadcast: Verteilen des globalen Ergebnisses an alle Prozesse
        global_sum /= (subgrid_size[1] * subgrid_size[0])
        global_sum = comm.bcast(global_sum, root=0)

        # MPI-Aktualisierung der Druckwerte
        p[1:-1, 1:-1] = (
            (1 / (-2 * dx ** 2 - 2 * dy ** 2))
            * (
                global_sum
                - dy ** 2 * (p[2:, 1:-1] + p[:-2, 1:-1])
                - dx ** 2 * (p[1:-1, 2:] + p[1:-1, :-2])
            )
        )

        # MPI-Aktualisierung der Residualnorm
        local_residual = np.linalg.norm(p - po)
        global_residual = comm.allreduce(local_residual, op=MPI.SUM)
        global_residual = comm.bcast(global_residual, root=0)

        res = global_residual
        n += 1
    comm.Barrier()
    # Compute u and v using pressure field p
    if rank == 0:
        compute_u(u, F, p, dx, dt, imax, jmax)
        compute_v(v, G, p, dy, dt, imax, jmax)

    # Gather the updated u and v to root
    comm.Gather(u, local_u, root=0)
    comm.Gather(v, local_v, root=0)

    # Assign the gathered u and v to the full arrays
    if rank == 0:
        u = local_u.copy()
        v = local_v.copy()

    po = p
    t += dt
    dt = select_dt_according_to_stability_condition(Re, dx, dy, tau, u, v, imax, jmax)

print(f"Time elapsed: {time.time() - tStart:.3f} seconds")


path = os.path.join(os.getcwd(), "data/")
save_matrix(path+"u.dat", u)
save_matrix(path+"v.dat", v)