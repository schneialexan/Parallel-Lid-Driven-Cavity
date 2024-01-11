import numpy as np
import time
from typing import Tuple
from multiprocessing import Process, Array, Barrier, Lock
import matplotlib.pyplot as plt


N_GRIDPOINTS = 64
DOMAIN_SIZE = 1.0
TIME_STEP_LENGTH = 0.001
DENSITY = 1.0
KINEMATIC_VISCOSITY = 0.01
HORIZONTAL_VELOCITY_TOP = 1.0
N_PRESSURE_ITERATIONS = 100
STABILITY_SAFETY_FACTOR = 0.5
ELEMENT_LENGTH = DOMAIN_SIZE / (N_GRIDPOINTS - 1)  # Länge eines Elements

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
    path = f'data'
    plt.savefig(f'{path}/threshold_{threshold}.png')
    plt.show()

def simulate_parallel(u, v, p, u_next, v_next, p_next, u_barrier, v_barrier, p_barrier, u_lock, v_lock, p_lock, threshold):
    # Simulate
    for n in range(1, 10000):
        # Compute intermediate velocity step
        u_star = u + TIME_STEP_LENGTH * (
            -u * central_difference_x(u) - v * central_difference_y(u)
            - central_difference_x(p) / DENSITY
            + KINEMATIC_VISCOSITY * laplace(u)
        )
        v_star = v + TIME_STEP_LENGTH * (
            -u * central_difference_x(v) - v * central_difference_y(v)
            - central_difference_y(p) / DENSITY
            + KINEMATIC_VISCOSITY * laplace(v)
        )

        # Apply boundary conditions
        u_star, v_star = enforce_boundary_conditions(u_star, v_star)

        # Synchronize
        u_barrier.wait()
        v_barrier.wait()

        # Compute next velocity step
        u_next[1:-1, 1:-1] = u_star[1:-1, 1:-1] + TIME_STEP_LENGTH * (
            -u_star[1:-1, 1:-1] * central_difference_x(u_star)[1:-1, 1:-1]
            - v_star[1:-1, 1:-1] * central_difference_y(u_star)[1:-1, 1:-1]
            - central_difference_x(p)[1:-1, 1:-1] / DENSITY
            + KINEMATIC_VISCOSITY * laplace(u_star)[1:-1, 1:-1]
        )
        v_next[1:-1, 1:-1] = v_star[1:-1, 1:-1] + TIME_STEP_LENGTH * (
            -u_star[1:-1, 1:-1] * central_difference_x(v_star)[1:-1, 1:-1]
            - v_star[1:-1, 1:-1] * central_difference_y(v_star)[1:-1, 1:-1]
            - central_difference_y(p)[1:-1, 1:-1] / DENSITY
            + KINEMATIC_VISCOSITY * laplace(v_star)[1:-1, 1:-1]
        )

        # Apply boundary conditions
        u_next, v_next = enforce_boundary_conditions(u_next, v_next)

        # Synchronize
        u_barrier.wait()
        v_barrier.wait()

        # Compute next pressure step
        p_next[1:-1, 1:-1] = p[1:-1, 1:-1] + STABILITY_SAFETY_FACTOR * (
            -DENSITY * (1 / TIME_STEP_LENGTH * (
                central_difference_x(u_next)[1:-1, 1:-1]
                + central_difference_y(v_next)[1:-1, 1:-1]
            ))
            - (
                central_difference_x(u_next)[1:-1, 1:-1]**2
                + 2 * central_difference_y(u_next)[1:-1, 1:-1] * central_difference_x(v_next)[1:-1, 1:-1]
                + central_difference_y(v_next)[1:-1, 1:-1]**2
            )
        )

        # Apply boundary conditions
        p_next[1:-1, 0] = p_next[1:-1, 1]
        p_next[1:-1, -1] = p_next[1:-1, -2]
        p_next[0, :] = p_next[1, :]
        p_next[-1, :] = 0.0

        # Synchronize
        p_barrier.wait()
        # Check convergence
        if np.all(np.abs(p_next - p) < threshold):
            print(f'Converged at iteration {n}')
            break

        # Output information at each iteration
        if n % 100 == 0:
            print(f'Iteration {n} - Max pressure difference: {np.max(np.abs(p_next - p))}')


        # Check convergence
        if np.all(np.abs(p_next - p) < threshold):
            break


        # Update variables
        u_lock.acquire()
        v_lock.acquire()
        p_lock.acquire()

        u[1:-1, 1:-1] = u_next[1:-1, 1:-1]
        v[1:-1, 1:-1] = v_next[1:-1, 1:-1]
        p[1:-1, 1:-1] = p_next[1:-1, 1:-1]

        u_lock.release()
        v_lock.release()
        p_lock.release()
        print(f'Iteration {n} finished')
    return u, v, p, n


def simulate(u, v, p, threshold):   
    # Simulate
    for n in range(1, 10000):
        # Compute intermediate velocity step
        u_star = u + TIME_STEP_LENGTH * (
            -u * central_difference_x(u) - v * central_difference_y(u)
            - central_difference_x(p) / DENSITY
            + KINEMATIC_VISCOSITY * laplace(u)
        )
        v_star = v + TIME_STEP_LENGTH * (
            -u * central_difference_x(v) - v * central_difference_y(v)
            - central_difference_y(p) / DENSITY
            + KINEMATIC_VISCOSITY * laplace(v)
        )

        # Apply boundary conditions
        u_star, v_star = enforce_boundary_conditions(u_star, v_star)

        # Compute next velocity step
        u_next[1:-1, 1:-1] = u_star[1:-1, 1:-1] + TIME_STEP_LENGTH * (
            -u_star[1:-1, 1:-1] * central_difference_x(u_star)[1:-1, 1:-1]
            - v_star[1:-1, 1:-1] * central_difference_y(u_star)[1:-1, 1:-1]
            - central_difference_x(p)[1:-1, 1:-1] / DENSITY
            + KINEMATIC_VISCOSITY * laplace(u_star)[1:-1, 1:-1]
        )
        v_next[1:-1, 1:-1] = v_star[1:-1, 1:-1] + TIME_STEP_LENGTH * (
            -u_star[1:-1, 1:-1] * central_difference_x(v_star)[1:-1, 1:-1]
            - v_star[1:-1, 1:-1] * central_difference_y(v_star)[1:-1, 1:-1]
            - central_difference_y(p)[1:-1, 1:-1] / DENSITY
            + KINEMATIC_VISCOSITY * laplace(v_star)[1:-1, 1:-1]
        )

        # Apply boundary conditions
        u_next, v_next = enforce_boundary_conditions(u_next, v_next)

        # Compute next pressure step
        p_next[1:-1, 1:-1] = p[1:-1, 1:-1] + STABILITY_SAFETY_FACTOR * (
            -DENSITY * (1 / TIME_STEP_LENGTH * (
                central_difference_x(u_next)[1:-1, 1:-1]
                + central_difference_y(v_next)[1:-1, 1:-1]
            ))
            - (
                central_difference_x(u_next)[1:-1, 1:-1]**2
                + 2 * central_difference_y(u_next)[1:-1, 1:-1] * central_difference_x(v_next)[1:-1, 1:-1]
                + central_difference_y(v_next)[1:-1, 1:-1]**2
            )
        )



if __name__ == "__main__":
    N_GRIDPOINTS = 64                      # Anzahl Gitterpunkte
    DOMAIN_SIZE = 1.                        # Länge des Gebietes
    TIME_STEP_LENGTH = 0.001                # Länge des Zeitschrittes
    DENSITY = 1.                            # Dichte
    KINEMATIC_VISCOSITY = 0.01              # kinematische Viskosität
    HORIZONTAL_VELOCITY_TOP = 1.            # Geschwindigkeit oben (Deckel bzw. Wand)

    N_PRESSURE_ITERATIONS = 100            # Anzahl Iterationen für den Druck
    ELEMENT_LENGTH = DOMAIN_SIZE / (N_GRIDPOINTS - 1)   # Länge eines Elements

    # Initialisierung
    u = np.zeros((N_GRIDPOINTS, N_GRIDPOINTS))
    v = np.zeros((N_GRIDPOINTS, N_GRIDPOINTS))
    p = np.zeros((N_GRIDPOINTS, N_GRIDPOINTS))

    # Gitter erstellen
    x = np.linspace(0, DOMAIN_SIZE, N_GRIDPOINTS)
    y = np.linspace(0, DOMAIN_SIZE, N_GRIDPOINTS)
    X, Y = np.meshgrid(x, y)

    # Initialisierung für die Parallelisierung
    u_next = np.zeros_like(u)
    v_next = np.zeros_like(v)
    p_next = np.zeros_like(p)

    # Initialisierung für die Synchronisation
    u_barrier = Barrier(N_GRIDPOINTS - 1)
    v_barrier = Barrier(N_GRIDPOINTS - 1)
    p_barrier = Barrier(N_GRIDPOINTS - 1)

    u_lock = Lock()
    v_lock = Lock()
    p_lock = Lock()

    # Initialisierung für die Konvergenz
    threshold = 1e-1

    # Simuliere parallel
    start = time.time()
    u, v, p, n = simulate_parallel(u, v, p, u_next, v_next, p_next, u_barrier, v_barrier, p_barrier, u_lock, v_lock, p_lock, threshold)
    end = time.time()
    print(f'Overall: {n} iterations with error: {threshold} | Time used: {end - start:.2f}s')
