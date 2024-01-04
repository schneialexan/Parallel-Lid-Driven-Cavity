import numpy as np
from multiprocessing import Process, Manager, Lock, cpu_count, Value
import multiprocessing

class Simulation: # 
    def __init__(self, p_imax=50, p_jmax=50, p_xlength=1.0, p_ylength=1.0, p_t_end=50.0, p_tau=0.5, p_eps=1e-3,
                 p_omg=1.7, p_itermax=100, p_alpha=0.9, p_Re=100.0, p_t=0, p_dt=0.005, p_res=99999):
        self.imax = p_imax # grid Dimension i (The number of grid cells in the x-direction)
        self.jmax = p_jmax # grid Dimension j (he number of grid cells in the y-direction)
        self.xlength = p_xlength # physische Dimension
        self.ylength = p_ylength # physische Dimension
        self.t_end = p_t_end #  simulation time
        self.tau = p_tau 
        self.eps = p_eps # positive constant
	# relaxation parameter used in the iterative solution of the pressure equation
        self.omg = p_omg

        self.itermax = p_itermax
        self.alpha = p_alpha
        self.Re = p_Re # Reynold
        self.t = p_t #  Initial time
        self.dt = p_dt 
        self.res = p_res

class StaggeredGrid:
    def __init__(self, p_imax=50, p_jmax=50, p_xlength=1.0, p_ylength=1.0):
        self.imax = p_imax
        self.jmax = p_jmax
        self.xlength = p_xlength
        self.ylength = p_ylength
        self.p = np.zeros((p_imax + 2, p_jmax + 2)) # pressure field
        self.po = np.zeros((p_imax + 2, p_jmax + 2)) # previous pressure field 
        self.RHS = np.zeros((p_imax + 2, p_jmax + 2)) # right-hand side values during the solution of the pressure equation
        self.u = np.zeros((p_imax + 2, p_jmax + 3)) # x-component of velocity
        self.F = np.zeros((p_imax + 2, p_jmax + 3)) # intermediate values related to the x-component of velocity during the simulation
        self.v = np.zeros((p_imax + 3, p_jmax + 2)) # y-component of velocity
        self.G = np.zeros((p_imax + 3, p_jmax + 2)) # intermediate values related to the y-component of velocity during the simulation
 
    def dx(self): # r�umliche Schrittgr�sse in x-Richtung
        return self.xlength / self.imax

    def dy(self): # r�umliche Schrittgr�sse in y-Richtung
        return self.ylength / self.jmax

    def find_max_absolute_u(self): # maximum absolute value of the x-component of velocity 
        return np.max(np.abs(self.u[1:self.imax + 1, 1:self.jmax + 2]))

    def find_max_absolute_v(self): #  maximum absolute value of the y-component of velocity
        return np.max(np.abs(self.v[1:self.imax + 2, 1:self.jmax + 1]))

def select_dt_according_to_stability_condition(grid, sim):
    left = (sim.Re / 2) * ((1 / grid.dx() ** 2) + (1 / grid.dy() ** 2)) ** -1 # left term in the stability condition equation (dx and dy are the grid Abst�nde)
    middle = grid.dx() / grid.find_max_absolute_u() # Maximum absolute value of the x-component of velocity within the grid
    right = grid.dy() / grid.find_max_absolute_v() #  Maximum absolute value of the y-component of velocity within the grid
    sim.dt = sim.tau * min(left, middle, right) # Time Step Determination

def set_boundary_conditions_u(grid): #Randbedingungen Geschwindigkeit u
    for j in range(grid.jmax + 3):
        grid.u[0, j] = 0.0
        grid.u[grid.imax, j] = 0.0

    for i in range(grid.imax + 2): # 
        grid.u[i, 0] = -grid.u[i, 1]
        grid.u[i, grid.jmax + 1] = -grid.u[i, grid.jmax]

    for i in range(grid.jmax + 2):
        grid.u[i, grid.jmax + 1] = 2.0 - grid.u[i, grid.jmax]

def set_boundary_conditions_v(grid): #Randbedingungen Geschwindigkeit v
    for j in range(grid.jmax + 2):
        grid.v[0, j] = -grid.v[1, j]
        grid.v[grid.imax + 1, j] = -grid.v[grid.imax, j]

    for i in range(grid.imax + 3):
        grid.v[i, 0] = 0.0
        grid.v[i, grid.jmax] = 0.0

def set_boundary_conditions_p(grid): #  boundary conditions for the pressure
    for i in range(grid.imax + 2):
        grid.p[i, 0] = grid.p[i, 1]
        grid.p[i, grid.jmax + 1] = grid.p[i, grid.jmax] # Left and Right Boundaries 

    for j in range(grid.jmax + 2):
        grid.p[0, j] = grid.p[1, j]
        grid.p[grid.imax + 1, j] = grid.p[grid.imax, j] # Top and Bottom Boundaries

# ME_X namespace
# Folien 5, Slide 15
def uu_x(grid, sim, i, j): #  x-component of the nonlinear convection term
    return (
        (1 / grid.dx()) * ((0.5 * (grid.u[i, j] + grid.u[i + 1, j])) ** 2 - (0.5 * (grid.u[i - 1, j] + grid.u[i, j])) ** 2)
        + (sim.alpha / grid.dx())
        * (
            abs(0.5 * (grid.u[i, j] + grid.u[i + 1, j])) * (0.5 * (grid.u[i, j] - grid.u[i + 1, j])) / 4
            - abs(0.5 * (grid.u[i - 1, j] + grid.u[i, j])) * (0.5 * (grid.u[i - 1, j] - grid.u[i, j])) / 4
        )
    )

def uv_y(grid, sim, i, j): # y-component of the nonlinear convection term
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
def uu_xx(grid, sim, i, j): #partielle Ableitung 
    return (grid.u[i + 1, j] - 2 * grid.u[i, j] + grid.u[i - 1, j]) / grid.dx() ** 2

def uu_yy(grid, sim, i, j): #partielle Ableitung 
    return (grid.u[i, j + 1] - 2 * grid.u[i, j] + grid.u[i, j - 1]) / grid.dy() ** 2

def p_x(grid, sim, i, j): #partielle Ableitung 
    return (grid.p[i + 1, j] - grid.p[i, j]) / grid.dx()

# ME_Y namespace
# Folien 5, Slide 17
def uv_x(grid, sim, i, j): # responsible for computing the x-component of the nonlinear convection term
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

def vv_y(grid, sim, i, j): # responsible for computing the y-component of the nonlinear convection term
    return (
        (1 / grid.dy()) * ((0.5 * (grid.v[i, j] + grid.v[i, j + 1])) ** 2 - (0.5 * (grid.v[i, j - 1] + grid.v[i, j])) ** 2)
        + (sim.alpha / grid.dy())
        * (
            abs(0.5 * (grid.v[i, j] + grid.v[i, j + 1])) * (0.5 * (grid.v[i, j] - grid.v[i, j + 1])) / 4
            - abs(0.5 * (grid.v[i, j - 1] + grid.v[i, j])) * (0.5 * (grid.v[i, j - 1] - grid.v[i, j])) / 4
        )
    )

# Folien 5, Slide 18
def vv_xx(grid, sim, i, j): #partielle Ableitung 
    return (grid.v[i + 1, j] - 2 * grid.v[i, j] + grid.v[i - 1, j]) / grid.dx() ** 2

def vv_yy(grid, sim, i, j): #partielle Ableitung 
    return (grid.v[i, j + 1] - 2 * grid.v[i, j] + grid.v[i, j - 1]) / grid.dy() ** 2

def p_y(grid, sim, i, j): #partielle Ableitung 
    return (grid.p[i, j + 1] - grid.p[i, j]) / grid.dy()

# CE namespace
# Folien 5, Slide 19
def u_x(grid, sim, i, j): # r�umliche Ableitungen der u-Komponente der Geschwindigkeit
    return (grid.u[i, j] - grid.u[i - 1, j]) / grid.dx()

def v_y(grid, sim, i, j): # r�umliche Ableitungen der v-Komponente der Geschwindigkeit
    return (grid.v[i, j] - grid.v[i, j - 1]) / grid.dy()


# -----------------------
#  numerical solution of the Navier-Stokes equations for incompressible fluid flow. update the intermediate variables F and G based on the current velocity field, are used in the pressure Poisson equation to calculate the pressure field in each time step 

def parallel_compute_f(grid, sim, start, end):
    for j in range(1, grid.jmax + 2):
        for i in range(start, end):
            grid.F[i, j] = grid.u[i, j] + sim.dt * (
                (1 / sim.Re) * (uu_xx(grid, sim, i, j) + uu_yy(grid, sim, i, j))
                - uu_x(grid, sim, i, j)
                - uv_y(grid, sim, i, j)
            )

def parallel_compute_g(grid, sim, start, end):
    for i in range(1, grid.imax + 2):
        for j in range(start, end):
            grid.G[i, j] = grid.v[i, j] + sim.dt * (
                (1 / sim.Re) * (vv_xx(grid, sim, i, j) + vv_yy(grid, sim, i, j))
                - uv_x(grid, sim, i, j)
                - vv_y(grid, sim, i, j)
            )

def parallel_compute_rhs(grid, sim, start, end):
    for i in range(1, grid.imax + 1):
        for j in range(start, end):
            grid.RHS[i, j] = (1 / sim.dt) * (
                (grid.F[i, j] - grid.F[i - 1, j]) / grid.dx() + (grid.G[i, j] - grid.G[i, j - 1]) / grid.dy()
            )

def parallel_update_step_lgls(grid, sim, start, end):
    for i in range(1, grid.imax + 1):
        for j in range(start, end):
            grid.p[i, j] = (
                (1 / (-2 * grid.dx() ** 2 - 2 * grid.dy() ** 2))
                * (
                    grid.RHS[i, j] * grid.dx() ** 2 * grid.dy() ** 2
                    - grid.dy() ** 2 * (grid.p[i + 1, j] + grid.p[i - 1, j])
                    - grid.dx() ** 2 * (grid.p[i, j + 1] + grid.p[i, j - 1])
                )
            )

def parallel_compute_u(grid, sim, start, end):
    for i in range(1, grid.imax + 1):
        for j in range(start, end):
            grid.u[i, j] = grid.F[i, j] - (sim.dt / grid.dx()) * (grid.p[i + 1, j] - grid.p[i, j])

def parallel_compute_v(grid, sim, start, end):
    for i in range(1, grid.imax + 2):
        for j in range(start, end):
            grid.v[i, j] = grid.G[i, j] - (sim.dt / grid.dy()) * (grid.p[i, j + 1] - grid.p[i, j])


def compute_residual(grid, sim): # residual represents difference between the current pressure field (grid.p) and previous pressure field (grid.po)
    sim.res = np.linalg.norm(grid.p - grid.po)
    
    
    


def save_matrix(filename, matrix):
    with open(filename, 'w') as file:
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                file.write(f"{matrix[i, j]:.5f} ")
            file.write("\n")
    print(f"Matrix saved to {filename}")

# ------------------------------

# Constants
# Maximum index in the x-direction (grid points in i direction)
imax = 50 
# Maximum index in the y-direction (grid points in j direction)
jmax = 50
# Length of the domain 
xlength = 1.0
ylength = 1.0
#Time step factor
t_end = 5.0
tau = 0.5
eps = 1e-3
#  Relaxation parameter
omg = 1.7
# Maximum number of iterations
itermax = 100
alpha = 0.5
Re = 100.0

# Variables
t = 0
dt = 0.05
res = 99999 # Residual

# Initialization
sim = Simulation(imax, jmax, xlength, ylength, t_end, tau, eps, omg, itermax, alpha, Re, t, dt, res)
grid = StaggeredGrid(imax, jmax, xlength, ylength)

n = 0

lock = Lock()


def simulate_part(grid, sim, start, end, counter, lock):
    while sim.t < sim.t_end:
        with lock:  # Verwenden Sie den Lock hier
            n = counter.value
            counter.value += 1

        
        set_boundary_conditions_u(grid)
        set_boundary_conditions_v(grid)

        with multiprocessing.Pool() as pool:
            pool.starmap(parallel_compute_f, [(grid, sim, start, end)])
            pool.starmap(parallel_compute_g, [(grid, sim, start, end)])
            pool.starmap(parallel_compute_rhs, [(grid, sim, start, end)])

        while (sim.res > sim.eps or sim.res == 0) and n < sim.itermax:
            set_boundary_conditions_p(grid)

            with multiprocessing.Pool() as pool:
                pool.starmap(parallel_update_step_lgls, [(grid, sim, start, end)])

            compute_residual(grid, sim)

            with lock:  # Verwenden Sie den Lock hier
                n = counter.value
                counter.value += 1

        with multiprocessing.Pool() as pool:
            pool.starmap(parallel_compute_u, [(grid, sim, start, end)])
            pool.starmap(parallel_compute_v, [(grid, sim, start, end)])

        grid.po = grid.p
        sim.t += sim.dt
        select_dt_according_to_stability_condition(grid, sim)





if __name__ == "__main__":
    manager = Manager()
    counter = manager.Value("i", 0)
    lock = Lock()  # Erstellen Sie den Lock hier

    processes = []

    chunk_size = 2
    
    for start in range(1, grid.imax + 1, chunk_size):
        end = min(start + chunk_size, grid.imax + 1)
        process = Process(target=simulate_part, args=(grid, sim, start, end, counter, lock))  # Übergeben Sie den Lock an das simulate_part
        processes.append(process)
        process.start()

    for process in processes:
        process.join()





    # Kombinieren Sie die Ergebnisse aus verschiedenen Prozessen
    combined_u = np.zeros_like(grid.u)
    combined_v = np.zeros_like(grid.v)
    combined_p = np.zeros_like(grid.p)

    for start in range(1, grid.imax + 1, chunk_size):
        end = min(start + chunk_size, grid.imax + 1)
        combined_u[start:end, :] = grid.u[start:end, :]
        combined_v[start:end, :] = grid.v[start:end, :]
        combined_p[start:end, :] = grid.p[start:end, :]

    # Speichern Sie das kombinierte Ergebnis
    save_matrix("u.dat_combined", combined_u)
    save_matrix("v.dat_combined", combined_v)
    save_matrix("p.dat_combined", combined_p)
    