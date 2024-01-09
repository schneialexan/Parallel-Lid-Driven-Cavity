import matplotlib.pyplot as plt
import numpy as np
import time
from typing import Tuple

N_GRIDPOINTS = 64                      # Anzahl Gitterpunkte
DOMAIN_SIZE = 1.                        # Länge des Gebietes
TIME_STEP_LENGTH = 0.001                # Länge des Zeitschrittes, aufgrund der CFL-Bedingung und der Stabilität
DENSITY = 1.                            # Dichte
KINEMATIC_VISCOSITY = 0.01              # kinematische Viskosität
HORIZONTAL_VELOCITY_TOP = 1.            # Geschwindigkeit oben (Deckel bzw. Wand)

N_PRESSURE_ITERATIONS = 100            # Anzahl Iterationen für den Druck
STABILITY_SAFETY_FACTOR = 0.5           # Sicherheitsfaktor für die Stabilität
ELEMENT_LENGTH = DOMAIN_SIZE / (N_GRIDPOINTS - 1)   # Länge eines Elements

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
    scale = 2
    plt.contourf(X[::scale, ::scale], Y[::scale, ::scale], p_next[::scale, ::scale])
    plt.quiver(X[::scale, ::scale], Y[::scale, ::scale], u_next[::scale, ::scale], v_next[::scale, ::scale])
    plt.colorbar()

    # Set title and axis labels
    plt.title('Velocity and Pressure Contours')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.xlim((0, 1))
    plt.ylim((0, 1))

    # Show the plot
    plt.show()

# 0. Initialisierung
u = np.zeros((N_GRIDPOINTS, N_GRIDPOINTS))
v = np.zeros((N_GRIDPOINTS, N_GRIDPOINTS))
p = np.zeros((N_GRIDPOINTS, N_GRIDPOINTS))

n_iter = 0
start = time.time()

error = 1.
threshold = 1e-5
    
while error > threshold and n_iter < 50000:
    if n_iter % 1000 == 0:
        print(f'Iteration: {n_iter:3d} with error: {error}')
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

    p_local = np.zeros_like(p)
    rhs_local = np.zeros_like(rhs)
    p_local[1:-1, 1:-1] = p[1:-1, 1:-1]
    rhs_local[1:-1, 1:-1] = rhs[1:-1, 1:-1]

    # 2. Druck Laplace lösen
    for _ in range(N_PRESSURE_ITERATIONS):
        p_next = np.zeros_like(p_local)
        p_next[1:-1, 1:-1] = 1/4 * (+p_local[1:-1, 0:-2]+p_local[0:-2, 1:-1]+p_local[1:-1, 2:]+p_local[2:, 1:-1]-ELEMENT_LENGTH**2*rhs_local[1:-1, 1:-1])
        # 2. Randbedingungen erzwingen
        p_next[:,  0] = p_next[:,  1]   # left boundary (Neumann)
        p_next[:, -1] = p_next[:, -2]   # right boundary (Neumann)
        p_next[-1, :] = 0.0               # top boundary (Neumann)
        p_next[0,  :] = p_next[1,  :]   # bottom boundary (Neumann)
        p_local = p_next
    
    dp_next_dx = central_difference_x(p_local)
    dp_next_dy = central_difference_y(p_local)
    
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
    
    u_error = np.max(np.abs((u - u_old) / TIME_STEP_LENGTH))
    v_error = np.max(np.abs((v - v_old) / TIME_STEP_LENGTH))
    p_error = np.max(np.abs((p - p_old) / TIME_STEP_LENGTH))
    
    error = max(u_error, v_error, p_error)
    
    n_iter += 1

print(f'Overall: {n_iter:3d} iterations with error: {error} | Time used: {time.time()-start:.2f}s')   

# Plot
x = np.linspace(0.0, DOMAIN_SIZE, N_GRIDPOINTS)
y = np.linspace(0.0, DOMAIN_SIZE, N_GRIDPOINTS)
X, Y = np.meshgrid(x, y)
plot_veloctiy_and_pressure(X, Y, p, u, v)