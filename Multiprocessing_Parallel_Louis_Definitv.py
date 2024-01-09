import matplotlib.pyplot as plt
import numpy as np
import time
from typing import Tuple
import multiprocessing
num_cpus = multiprocessing.cpu_count()

print(f"Anzahl der verf�gbaren CPU-Kerne: {num_cpus}")


# Deine Konstanten
N_GRIDPOINTS = 64
DOMAIN_SIZE = 1.0
TIME_STEP_LENGTH = 0.001
DENSITY = 1.0
KINEMATIC_VISCOSITY = 0.01
HORIZONTAL_VELOCITY_TOP = 1.0
N_PRESSURE_ITERATIONS = 100
STABILITY_SAFETY_FACTOR = 0.5
ELEMENT_LENGTH = DOMAIN_SIZE / (N_GRIDPOINTS - 1)



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

# Funktion f�r eine Iteration der Schleife
def process_iteration(iteration_number, u, v, p, threshold, TIME_STEP_LENGTH, KINEMATIC_VISCOSITY,
                      N_PRESSURE_ITERATIONS, ELEMENT_LENGTH, DENSITY):
    # Initialisierung
    du_dx = central_difference_x(u)
    du_dy = central_difference_y(u)
    dv_dx = central_difference_x(v)
    dv_dy = central_difference_y(v)
    laplace_u = laplace(u)
    laplace_v = laplace(v)
    
    # L�sung Impulsgleichung ohne Druck
    u_tentative = (u + TIME_STEP_LENGTH * (KINEMATIC_VISCOSITY * laplace_u - (u * du_dx + v * du_dy)))
    v_tentative = (v + TIME_STEP_LENGTH * (KINEMATIC_VISCOSITY * laplace_v - (u * dv_dx + v * dv_dy)))
    
    # Randbedingungen erzwingen
    u_tentative, v_tentative = enforce_boundary_conditions(u_tentative, v_tentative)
    
    du_tentative_dx = central_difference_x(u_tentative)
    dv_tentative_dy = central_difference_y(v_tentative)
    
    # Pressure-Poisson Equation
    # Rechte Seite berechnen
    rhs = (DENSITY / TIME_STEP_LENGTH * (du_tentative_dx + dv_tentative_dy))

    p_local = np.zeros_like(p)
    rhs_local = np.zeros_like(rhs)
    p_local[1:-1, 1:-1] = p[1:-1, 1:-1]
    rhs_local[1:-1, 1:-1] = rhs[1:-1, 1:-1]

    # Druck Laplace l�sen
    for _ in range(N_PRESSURE_ITERATIONS):
        p_next = np.zeros_like(p_local)
        p_next[1:-1, 1:-1] = 1/4 * (+p_local[1:-1, 0:-2] + p_local[0:-2, 1:-1] + p_local[1:-1, 2:] +
                                    p_local[2:, 1:-1] - ELEMENT_LENGTH**2 * rhs_local[1:-1, 1:-1])
        # Randbedingungen erzwingen
        p_next[:,  0] = p_next[:,  1]   # left boundary (Neumann)
        p_next[:, -1] = p_next[:, -2]   # right boundary (Neumann)
        p_next[-1, :] = 0.0             # top boundary (Neumann)
        p_next[0,  :] = p_next[1,  :]   # bottom boundary (Neumann)
        p_local = p_next
    
    dp_next_dx = central_difference_x(p_local)
    dp_next_dy = central_difference_y(p_local)
    
    # Geschwindigkeit mit Druck korrigieren
    u_next = (u_tentative - TIME_STEP_LENGTH / DENSITY * dp_next_dx)
    v_next = (v_tentative - TIME_STEP_LENGTH / DENSITY * dp_next_dy)
    
    # Randbedingungen erzwingen
    u_next, v_next = enforce_boundary_conditions(u_next, v_next)
    
    u_old = u
    v_old = v
    p_old = p
    
    u = u_next
    v = v_next
    p = p_next
    
    # Fehler berechnen
    u_error = np.max(np.abs((u - u_old) / TIME_STEP_LENGTH))
    v_error = np.max(np.abs((v - v_old) / TIME_STEP_LENGTH))
    p_error = np.max(np.abs((p - p_old) / TIME_STEP_LENGTH))
    
    error = max(u_error, v_error, p_error)
    
    return u, v, p, error




# Hauptprogramm
if __name__ == '__main__':
    # Initialisierung
    u = np.zeros((N_GRIDPOINTS, N_GRIDPOINTS))
    v = np.zeros((N_GRIDPOINTS, N_GRIDPOINTS))
    p = np.zeros((N_GRIDPOINTS, N_GRIDPOINTS))

    n_iter = 0
    start = time.time()

    error = 1.0
    threshold = 1e-5 # noch anpassen 
    num_processes = 1 # noch anpassen

    # Pool erstellen

    
    pool = multiprocessing.Pool(processes=num_processes)

    while error > threshold and n_iter < 50000:
        if n_iter % 1000 == 0:
            print(f'Iteration: {n_iter:3d} with error: {error}')

        # Prozesse vorbereiten
        process_args = [(n_iter, u, v, p, threshold, TIME_STEP_LENGTH, KINEMATIC_VISCOSITY,
                         N_PRESSURE_ITERATIONS, ELEMENT_LENGTH, DENSITY) for _ in range(num_processes)]

        # Prozesse ausf�hren
        results = pool.starmap(process_iteration, process_args)

        # Ergebnisse sammeln
        for result in results:
            u, v, p, error = result

        n_iter += num_processes

    pool.close()
    pool.join()
    num_cpus = multiprocessing.cpu_count()
    print(f"Anzahl der verf�gbaren CPU-Kerne: {num_cpus}")
    print(f'benutzte CPU-Kerne: {num_processes}')
    print(f'Overall: {n_iter:3d} iterations with error: {error} | Time used: {time.time() - start:.2f}s')

    # Plot
    x = np.linspace(0.0, DOMAIN_SIZE, N_GRIDPOINTS)
    y = np.linspace(0.0, DOMAIN_SIZE, N_GRIDPOINTS)
    X, Y = np.meshgrid(x, y)
    plot_veloctiy_and_pressure(X, Y, p, u, v)