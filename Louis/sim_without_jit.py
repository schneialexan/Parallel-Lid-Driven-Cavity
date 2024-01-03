import numpy as np
#from numba import jit

def find_max_absolute_u(u, imax, jmax):
    return np.max(np.abs(u[1:imax + 1, 1:jmax + 2]))

def find_max_absolute_v(v, imax, jmax):
    return np.max(np.abs(v[1:imax + 2, 1:jmax + 1]))

def select_dt_according_to_stability_condition(Re, dx, dy, tau):
    left = (Re / 2) * ((1 / dx ** 2) + (1 / dy ** 2)) ** -1
    middle = dx / find_max_absolute_u(u, imax, jmax)
    right = dy / find_max_absolute_v(v, imax, jmax)
    return tau * min(left, middle, right)

#@jit(fastmath=True)
def set_boundary_conditions_u(u, jmax, imax):
    for j in range(jmax + 3):
        u[0, j] = 0.0
        u[imax, j] = 0.0

    for i in range(imax + 2):
        u[i, 0] = -u[i, 1]
        u[i, jmax + 1] = -u[i, jmax]

    for i in range(jmax + 2):
        u[i, jmax + 1] = 2.0 - u[i, jmax]

#@jit(fastmath=True)
def set_boundary_conditions_v(v, jmax, imax):
    for j in range(jmax + 2):
        v[0, j] = -v[1, j]
        v[imax + 1, j] = -v[imax, j]

    for i in range(imax + 3):
        v[i, 0] = 0.0
        v[i, jmax] = 0.0

#@jit(fastmath=True)
def set_boundary_conditions_p(p, jmax, imax):
    for i in range(imax + 2):
        p[i, 0] = p[i, 1]
        p[i, jmax + 1] = p[i, jmax]

    for j in range(jmax + 2):
        p[0, j] = p[1, j]
        p[imax + 1, j] = p[imax, j]

# ME_X namespace
# Folien 5, Slide 15
#@jit(fastmath=True)
def uu_x(u, dx, i, j, alpha):
    return (
        (1 / dx) * ((0.5 * (u[i, j] + u[i + 1, j])) ** 2 - (0.5 * (u[i - 1, j] + u[i, j])) ** 2)
        + (alpha / dx)
        * (
            abs(0.5 * (u[i, j] + u[i + 1, j])) * (0.5 * (u[i, j] - u[i + 1, j])) / 4
            - abs(0.5 * (u[i - 1, j] + u[i, j])) * (0.5 * (u[i - 1, j] - u[i, j])) / 4
        )
    )

#@jit(fastmath=True)
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

#@jit(fastmath=True)
# Folien 5, Slide 16
def uu_xx(u, dx, i, j):
    return (u[i + 1, j] - 2 * u[i, j] + u[i - 1, j]) / dx ** 2

#@jit(fastmath=True)
def uu_yy(u, dy, i, j):
    return (u[i, j + 1] - 2 * u[i, j] + u[i, j - 1]) / dy ** 2

#@jit(fastmath=True)
def p_x(p, dx, i, j):
    return (p[i + 1, j] - p[i, j]) / dx

# ME_Y namespace
# Folien 5, Slide 17
#@jit(fastmath=True)
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

#@jit(fastmath=True)
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
#@jit(fastmath=True)
def vv_xx(v, dx, i, j):
    return (v[i + 1, j] - 2 * v[i, j] + v[i - 1, j]) / dx ** 2

#@jit(fastmath=True)
def vv_yy(v, dy, i, j):
    return (v[i, j + 1] - 2 * v[i, j] + v[i, j - 1]) / dy ** 2

#@jit(fastmath=True)
def p_y(p, dy, i, j):
    return (p[i, j + 1] - p[i, j]) / dy

# CE namespace
# Folien 5, Slide 19
#@jit(fastmath=True)
def u_x(u, dx, i, j):
    return (u[i, j] - u[i - 1, j]) / dx

#@jit(fastmath=True)
def v_y(v, dy, i, j):
    return (v[i, j] - v[i, j - 1]) / dy


#@jit(fastmath=True)
def compute_f(Re, F, u, v, dx, dy, dt, imax, jmax, alpha):
    for j in range(1, jmax + 2):
        for i in range(1, imax + 1):
            F[i, j] = u[i, j] + dt * (
                (1 / Re) * (uu_xx(u, dx, i, j) + uu_yy(u, dy, i, j))
                - uu_x(u, dx, i, j, alpha)
                - uv_y(u, v, dy, i, j, alpha)
            )

#@jit(fastmath=True)
def compute_g(Re, G, u, v, dx, dy, dt, imax, jmax, alpha):
    for i in range(1, imax + 2):
        for j in range(1, jmax + 1):
            G[i, j] = v[i, j] + dt * (
                (1 / Re) * (vv_xx(v, dx, i, j) + vv_yy(v, dy, i, j))
                - uv_x(u, v, dx, i, j, alpha)
                - vv_y(v, dy, i, j, alpha)
            )

#@jit(fastmath=True)
def compute_rhs(RHS, F, G, dx, dy, dt, imax, jmax):
    for i in range(1, imax + 1):
        for j in range(1, jmax + 1):
            RHS[i, j] = (1 / dt) * (
                (F[i, j] - F[i - 1, j]) / dx + 
                (G[i, j] - G[i, j - 1]) / dy
            )

#@jit(fastmath=True)
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

#@jit(fastmath=True)
def compute_residual(p, po):
    return np.linalg.norm(p - po)

#@jit(fastmath=True)
def compute_u(u, F, p, dx, dt, imax, jmax):
    for i in range(1, imax + 1):
        for j in range(1, jmax + 2):
            u[i, j] = F[i, j] - (dt / dx) * (p[i + 1, j] - p[i, j])

#@jit(fastmath=True)
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

# Variables
t = 0
dt = 0.05
res = 99999


p = np.zeros((imax + 2, jmax + 2))
po = np.zeros((imax + 2, jmax + 2))
RHS = np.zeros((imax + 2, jmax + 2))
u = np.zeros((imax + 2, jmax + 3))
F = np.zeros((imax + 2, jmax + 3))
v = np.zeros((imax + 3, jmax + 2))
G = np.zeros((imax + 3, jmax + 2))

n = 0

while t < t_end:
    n = 0
    dt = select_dt_according_to_stability_condition(Re, dx, dy, tau)
    print(f"t: {t:.3f} dt: {dt} res: {res}")
    set_boundary_conditions_u(u, jmax, imax)
    set_boundary_conditions_v(v, jmax, imax)
    compute_f(Re, F, u, v, dx, dy, dt, imax, jmax, alpha)
    compute_g(Re, G, u, v, dx, dy, dt, imax, jmax, alpha)
    compute_rhs(RHS, F, G, dx, dy, dt, imax, jmax)
    while (res > eps or res == 0) and n < itermax:
        set_boundary_conditions_p(p, jmax, imax)
        update_step_lgls(RHS, p, dx, dy, imax, jmax)
        res = compute_residual(p, po)
        n += 1
    compute_u(u, F, p, dx, dt, imax, jmax)
    compute_v(v, G, p, dy, dt, imax, jmax)
    po = p
    t += dt


save_matrix("u.dat", u)
save_matrix("v.dat", v)
save_matrix("p.dat", p)