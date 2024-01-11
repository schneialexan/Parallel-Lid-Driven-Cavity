import matplotlib.pyplot as plt
import numpy as np
from mpi4py import MPI
import time
from typing import Tuple

N_GRIDPOINTS = 256                      # Anzahl Gitterpunkte
DOMAIN_SIZE = 1.                        # Länge des Gebietes
TIME_STEP_LENGTH = 0.00035               # Länge des Zeitschrittes, aufgrund der CFL-Bedingung und der Stabilität
DENSITY = 1.                            # Dichte
KINEMATIC_VISCOSITY = 0.01              # kinematische Viskosität
HORIZONTAL_VELOCITY_TOP = 1.            # Geschwindigkeit oben (Deckel bzw. Wand)

N_PRESSURE_ITERATIONS = 100            # Anzahl Iterationen für den Druck
STABILITY_SAFETY_FACTOR = 0.5           # Sicherheitsfaktor für die Stabilität (=tau für CFL-Bedingung)
ELEMENT_LENGTH = DOMAIN_SIZE / (N_GRIDPOINTS - 1)   # Länge eines Elements

def find_max_absolute_u(u, imax, jmax):
    return np.max(np.abs(u[1:imax + 1, 1:jmax + 2]))

def find_max_absolute_v(v, imax, jmax):
    return np.max(np.abs(v[1:imax + 2, 1:jmax + 1]))

def select_dt_according_to_stability_condition(Re, dx, dy, tau, u, v, imax, jmax):
    left = (Re / 2) * ((1 / dx ** 2) + (1 / dy ** 2)) ** -1
    middle = dx / find_max_absolute_u(u, imax, jmax)
    right = dy / find_max_absolute_v(v, imax, jmax)
    return tau * min(left, middle, right)

def central_difference_x(f):
    diff = np.zeros_like(f)
    diff[1:-1, 1:-1] = (f[1:-1, 2:]-f[1:-1, 0:-2]) / (2 * ELEMENT_LENGTH)
    return diff

def central_difference_y(f):
    diff = np.zeros_like(f)
    diff[1:-1, 1:-1] = (f[2:, 1:-1]-f[0:-2, 1:-1]) / (2 * ELEMENT_LENGTH)
    return diff

def laplace(f):
    diff = np.zeros_like(f)
    diff[1:-1, 1:-1] = (f[1:-1, 0:-2]+f[0:-2, 1:-1]-4*f[1:-1, 1:-1]+f[1:-1, 2:]+f[2:, 1:-1]) / (ELEMENT_LENGTH**2)
    return diff

def enforce_boundary_conditions(u, v):
    # Enforce boundary conditions for u
    u[0, :] = 0.0   # bottom boundary
    u[-1, :] = HORIZONTAL_VELOCITY_TOP  # top boundary
    u[:, 0] = 0.0   # left boundary
    u[:, -1] = 0.0  # right boundary
    
    # Enforce boundary conditions for v
    v[0, :] = 0.0   # bottom boundary
    v[-1, :] = 0.0  # top boundary
    v[:, 0] = 0.0   # left boundary
    v[:, -1] = 0.0  # right boundary
    return u, v

def plot_veloctiy_and_pressure(X, Y, p_next, u_next, v_next):
    plt.figure(figsize=(8, 6))
    plt.contourf(X[::2, ::2], Y[::2, ::2], p_next[::2, ::2])
    plt.quiver(X[::2, ::2], Y[::2, ::2], u_next[::2, ::2], v_next[::2, ::2])
    plt.colorbar()

    # Set title and axis labels
    plt.title('Velocity and Pressure Contours')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.xlim((0, 1))
    plt.ylim((0, 1))

    # Show the plot
    path = f'Parallel-Lid-Driven-Cavity\Alex\proof_of_concepts\data'
    plt.savefig(f'{path}/threshold_{threshold}.png')
    plt.show()

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

# 0. Initialisierung
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
partitionSize = (int(size**0.5), int(size**0.5)) # (Px, Py) --> Px * Py = Number of cores
cartGrid =  comm.Create_cart(dims=partitionSize, periods=[False, False])
rankShift = cartGrid.Shift(0, 1)
rankShift += cartGrid.Shift(1, 1)
rankShiftX = cartGrid.Shift(0, 1)
rankShiftY = cartGrid.Shift(1, 1)
subgrid_size, subgrid_pos = split_grid((N_GRIDPOINTS, N_GRIDPOINTS), partitionSize, cartGrid.Get_coords(rank), ELEMENT_LENGTH, ELEMENT_LENGTH)
subgrid_coords = cartGrid.Get_coords(rank)
print(f'Rank {rank}\t | Subgrid Size: {subgrid_size}\t | Subgrid Position: ({subgrid_pos[0]:.4f},{subgrid_pos[1]:.4f})')

u = np.zeros((N_GRIDPOINTS, N_GRIDPOINTS))
v = np.zeros((N_GRIDPOINTS, N_GRIDPOINTS))
p = np.zeros((N_GRIDPOINTS, N_GRIDPOINTS))

error = 1.
threshold = 1e-5
n_iter = 0
start = time.time()
while error > threshold and n_iter < 10000:
    #TIME_STEP_LENGTH = select_dt_according_to_stability_condition(KINEMATIC_VISCOSITY, ELEMENT_LENGTH, ELEMENT_LENGTH, STABILITY_SAFETY_FACTOR, u, v, N_GRIDPOINTS, N_GRIDPOINTS)
    if n_iter % 1000 == 0:
        print(f'Rank {rank}\t | Iteration: {n_iter} with error: {error:.4f} | dt: {TIME_STEP_LENGTH}')
    # 0. Initialisierung
    du_dx = central_difference_x(u)
    du_dy = central_difference_y(u)
    dv_dx = central_difference_x(v)
    dv_dy = central_difference_y(v)
    laplace_u = laplace(u)
    laplace_v = laplace(v)
    
    # 1. Lösung Impulsgleichung ohne Druck
    u_tentative = (u + TIME_STEP_LENGTH*(KINEMATIC_VISCOSITY*laplace_u - (u*du_dx + v*du_dy)))
    v_tentative = (v + TIME_STEP_LENGTH*(KINEMATIC_VISCOSITY*laplace_v - (u*dv_dx + v*dv_dy)))
    # 1. Randbedingungen erzwingen
    u_tentative, v_tentative = enforce_boundary_conditions(u_tentative, v_tentative)
    
    du_tentative_dx = central_difference_x(u_tentative)
    dv_tentative_dy = central_difference_y(v_tentative)
    
    # 2. Pressure-Poisson Equation
    # 2. rechte Seite berechnen
    rhs = (DENSITY / TIME_STEP_LENGTH * (du_tentative_dx+dv_tentative_dy))
    
    # 2. Druck Laplace lösen
    for _ in range(N_PRESSURE_ITERATIONS):
        p_next = np.zeros_like(p)
        p_next[1:-1, 1:-1] = 1/4 * (+p[1:-1, 0:-2]+p[0:-2, 1:-1]+p[1:-1, 2:]+p[2:, 1:-1]-ELEMENT_LENGTH**2*rhs[1:-1, 1:-1])
        # 2. Randbedingungen erzwingen
        p_next[:, -1] = p_next[:, -2]   # right boundary (Neumann)
        p_next[0,  :] = p_next[1,  :]   # bottom boundary (Neumann)
        p_next[:,  0] = p_next[:,  1]   # left boundary (Neumann)
        p_next[-1, :] = 0.0             # top boundary (Dirichlet)
    
    dp_next_dx = central_difference_x(p_next)
    dp_next_dy = central_difference_y(p_next)
    
    # 3. Geschwindigkeit mit Druck korrigieren
    u_next = (u_tentative - TIME_STEP_LENGTH/DENSITY * dp_next_dx)
    v_next = (v_tentative - TIME_STEP_LENGTH/DENSITY * dp_next_dy)
    
    # 3. Randbedingungen erzwingen
    u_next, v_next = enforce_boundary_conditions(u_next, v_next)
    u_old = u
    v_old = v
    p_old = p
    u = u_next
    v = v_next
    p = p_next
    n_iter += 1
    u_error = np.max(np.abs((u - u_old) / TIME_STEP_LENGTH))
    v_error = np.max(np.abs((v - v_old) / TIME_STEP_LENGTH))
    p_error = np.max(np.abs((p - p_old) / TIME_STEP_LENGTH))
    error = max(u_error, v_error, p_error)
print(f'Overall: {n_iter:3d} iterations with error: {error} | Time used: {time.time()-start:.2f}s')   

x = np.linspace(0.0, DOMAIN_SIZE, N_GRIDPOINTS)
y = np.linspace(0.0, DOMAIN_SIZE, N_GRIDPOINTS)
X, Y = np.meshgrid(x, y)
plot_veloctiy_and_pressure(X, Y, p_next, u_next, v_next)