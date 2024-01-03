'''
This file is used to run the parallel MPI version of the lid driven cavity problem

@Author: Alexandru Schneider
@Date: 03.01.2024
'''
import sys
from mpi4py import MPI
from LDC_program_options import LDCprogram_options
from MPI_Manager import validate, split_grid

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
    subgrid_size, subgrid_pos = split_grid((Nx, Ny), partitionSize, cartGrid.Get_coords(rank), dx, dy)
    
    
    tElapsed = MPI.Wtime() - tStart
    print(f'Rank {rank}\tSubgrid Size: {subgrid_size}\tSubgrid Position: ({subgrid_pos[0]:.4f},{subgrid_pos[1]:.4f})\tWall Time: {tElapsed:.6f} seconds')
    ##########################
    # End Simulation
    ##########################
    
    tElapsed_global = comm.reduce(tElapsed, op=MPI.MAX, root=0)
    if rank == 0:
        print(f"Total Wall Time: {tElapsed_global:.5f} seconds")
    

if __name__ == '__main__':
    main()