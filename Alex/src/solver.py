import numpy as np
from mpi4py import MPI

class Alex_Louis_Solver:
    def __init__(self, MPIcomm, rank, rankShift, coords, gridSize, dt, dx, dy, subPos, T, Re):
        self.MPIcomm = MPIcomm
        self.rank = rank

        self.coords = np.array(coords)
        self.rankShift = np.array(rankShift)

        self.Nx, self.Ny = gridSize
        self.dx, self.dy = dx, dy

        self.subPos = np.array(subPos)
        self.T, self.Re, self.dt = T, Re, dt

        self.U = 1.0  # Assuming a constant value for lid velocity

        self.coeff = np.zeros(3)

        self.s = np.zeros((self.Nx, self.Ny))
        self.v = np.zeros((self.Nx, self.Ny))
        self.v_new = np.zeros((self.Nx, self.Ny))
        self.velU = np.zeros((self.Nx, self.Ny))
        self.velV = np.zeros((self.Nx, self.Ny))
        self.bufNx = np.zeros(self.Nx)
        self.bufNy = np.zeros(self.Ny)

        self.poissonSolver = None