'''
This file is used to run the parallel MPI version of the lid driven cavity problem

@Author: Alexandru Schneider
@Date: 03.01.2024

Vorschlag 2: Parallelisierung
Druckgleichung parallelisieren mit MPI
Geschwindigkeit geht so schnell, das muss eigentlich nicht parallelisiert werden
Naiver Ansatz: 1 Prozess rechnet Geschwindigkeit, dann rechnen viele Prozesse gemeinsam den Druck. Problem: Kommunikation. Nicht machen.
Vorschlag RPM: Jeder Prozess löst die Geschwindigkeit und nutzt sie für seine Druckberechnung
Gebiet zerlegen, jeder Prozess rechnet u, v, p in seinem Gebiet, Daten austauschen (oder so?)
er schlägt vor zu zweit machen
'''
import sys
import os
from mpi4py import MPI
from LDC_program_options import LDCprogram_options
from MPI_Manager import validate, split_grid
from solver import Alex_Louis_Solver

def save_matrix(path, filename, matrix):
    if not os.path.exists(path):
        os.makedirs(path)
    with open(path+filename, 'w') as file:
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                file.write(f"{matrix[i, j]:.5f} ")
            file.write("\n")
    print(f"Matrix saved to {filename}")

def main():
    ##########################
    # Initialize MPI
    ##########################
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    help, args = LDCprogram_options()
    
    if help:
        if rank == 0:
            print("Program complete with exit code 0")
        MPI.Finalize()
        sys.exit(0)
        
    Nx, Ny, Px, Py, dt, t_end, tau, eps, omg, itermax, alpha, Re = args.Nx, args.Ny, args.Px, args.Py, args.dt, args.T, args.tau, args.eps, args.omg, args.itermax, args.alpha, args.Re
    partitionSize = (Px, Py)
    dx = 1.0 / (Nx - 1)
    dy = 1.0 / (Ny - 1)

    
    ##########################
    # Validate Inputs
    ##########################
    validate(size, rank, partitionSize, dt, dx, dy, Re)
    
    ##########################
    # Start Simulation
    ##########################
    tStart = MPI.Wtime()
    
    cartGrid =  comm.Create_cart(dims=partitionSize, periods=[False, False])
    
    rankShift = cartGrid.Shift(0, 1)
    rankShift += cartGrid.Shift(1, 1)
    subgrid_size, subgrid_pos = split_grid((Nx, Ny), partitionSize, cartGrid.Get_coords(rank), dx, dy)
    
    print(f'Rank {rank}\t | Subgrid Size: {subgrid_size}\t | Subgrid Position: ({subgrid_pos[0]:.4f},{subgrid_pos[1]:.4f}) \t | Rank Shift: {rankShift}')
    solver = Alex_Louis_Solver(comm, 
                               rank, 
                               rankShift,
                               cartGrid.Get_coords(rank), 
                               subgrid_size, 
                               dt, 
                               t_end, 
                               tau, 
                               eps, 
                               omg, 
                               itermax, 
                               alpha,
                               dx, 
                               dy, 
                               subgrid_pos, 
                               Re)

    solver.initialize()
    solver.solve()
    
    tElapsed = MPI.Wtime() - tStart
    ##########################
    # End Simulation
    ##########################
    
    tElapsed_global = comm.reduce(tElapsed, op=MPI.MAX, root=0)
    u_global = comm.gather(solver.u, root=0)
    v_global = comm.gather(solver.v, root=0)
    if rank == 0:
        print(f"Total Wall Time: {tElapsed_global:.5f} seconds")
        print(f'U: {len(u_global)}')
        print(f'V: {len(v_global)}')
        save_matrix(f'./data/{rank}-{size}/', f'u.dat', u_global[0])
        save_matrix(f'./data/{rank}-{size}/', f'v.dat', v_global[0])
        
    

if __name__ == '__main__':
    main()