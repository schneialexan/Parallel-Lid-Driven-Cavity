from mpi4py import MPI
from typing import Tuple
import numpy as np

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

# Example usage in a main program
if __name__ == "__main__":
    # MPI Setup
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Simulation Params
    Nx = Ny = 10
    x_length = y_length = 1.0
    dx = x_length / (Nx - 1)
    dy = y_length / (Ny - 1)
    partitionSize = (2, 2) # (Px, Py) --> Px * Py = Number of cores    
    
    # Create Cartesian Grid
    cartGrid =  comm.Create_cart(dims=partitionSize, periods=[False, False])
    
    # Get Rank Shifts (for communication between neighboring processes)
    rankShift = cartGrid.Shift(0, 1)
    rankShift += cartGrid.Shift(1, 1)
    rankShiftX = cartGrid.Shift(0, 1)
    rankShiftY = cartGrid.Shift(1, 1)
    
    # Get Subgrid Size and Position (for each process)
    subgrid_size, subgrid_pos = split_grid((Nx, Ny), partitionSize, cartGrid.Get_coords(rank), dx, dy)
    print(f'Rank {rank}\t | Subgrid Size: {subgrid_size}\t | Subgrid Position: ({subgrid_pos[0]:.4f},{subgrid_pos[1]:.4f})')
    
    # Lid Driven Cavity - Global Variables
    u = np.zeros((Nx, Ny))
    v = np.zeros((Nx, Ny))
    p = np.zeros((Nx, Ny))
    
    # Lid Driven Cavity - Local Variables
    u_local = np.zeros((subgrid_size[0], subgrid_size[1]))
    v_local = np.zeros((subgrid_size[0], subgrid_size[1]))
    p_local = np.zeros((subgrid_size[0], subgrid_size[1]))
    
    # Scatter Data
    if rank == 0:
        for i in range(size):
            if i != rank:
                comm.send(u, dest=i, tag=0)
                comm.send(v, dest=i, tag=1)
                comm.send(p, dest=i, tag=2)
    else:
        u_local = comm.recv(source=0, tag=0)
        v_local = comm.recv(source=0, tag=1)
        p_local = comm.recv(source=0, tag=2)
    
    # change values of u_local, v_local, p_local to test communication
    u_local = np.ones((subgrid_size[0], subgrid_size[1])) * rank + 1
    v_local = np.ones((subgrid_size[0], subgrid_size[1])) * rank + 1
    p_local = np.ones((subgrid_size[0], subgrid_size[1])) * rank + 1
    
    # Gather Data
    if rank == 0:
        # Gather data from all processes
        for i in range(size):
            if i != rank:
                coords = cartGrid.Get_coords(i)
                u[coords[0]*subgrid_size[0]:(coords[0]+1)*subgrid_size[0], coords[1]*subgrid_size[1]:(coords[1]+1)*subgrid_size[1]] = comm.recv(source=i, tag=0)
                v[coords[0]*subgrid_size[0]:(coords[0]+1)*subgrid_size[0], coords[1]*subgrid_size[1]:(coords[1]+1)*subgrid_size[1]] = comm.recv(source=i, tag=1)
                p[coords[0]*subgrid_size[0]:(coords[0]+1)*subgrid_size[0], coords[1]*subgrid_size[1]:(coords[1]+1)*subgrid_size[1]] = comm.recv(source=i, tag=2)
        # Gather data from local process
        u[0:subgrid_size[0], 0:subgrid_size[1]] = u_local
        v[0:subgrid_size[0], 0:subgrid_size[1]] = v_local
        p[0:subgrid_size[0], 0:subgrid_size[1]] = p_local
    else:
        comm.send(u_local, dest=0, tag=0)
        comm.send(v_local, dest=0, tag=1)
        comm.send(p_local, dest=0, tag=2)
    
    if rank == 0:
        # print the unique values in u, v, p, which now shouldn't be all zeros
        print(f'Unique values in u: {np.unique(u)}')
        print(f'Unique values in v: {np.unique(v)}')
        print(f'Unique values in p: {np.unique(p)}')
        print(f'{u=}')
        print(f'{v=}')
        print(f'{p=}')
