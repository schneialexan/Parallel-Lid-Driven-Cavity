from mpi4py import MPI
from typing import Tuple

def send_data(data, dest, tag, comm):
    """
    Sends data to a neighboring MPI process.

    Parameters:
    - data: The data to be sent.
    - dest: The rank of the destination MPI process.
    - tag: An integer tag to identify the message.
    - comm: MPI communicator.
    """
    comm.send(data, dest=dest, tag=tag)

def receive_data(source, tag, comm):
    """
    Receives data from a neighboring MPI process.

    Parameters:
    - source: The rank of the source MPI process.
    - tag: An integer tag to identify the message.
    - comm: MPI communicator.

    Returns:
    - Received data.
    """
    return comm.recv(source=source, tag=tag)

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

def interface_broadcast(comm, dest, data):
    pass

def interface_gather(comm, source):
    pass

# Example usage in a main program
if __name__ == "__main__":
    # MPI Setup
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Simulation Params
    Nx = Ny = 50
    x_length = y_length = 1.0
    dx = x_length / (Nx - 1)
    dy = y_length / (Ny - 1)
    partitionSize = (2, 2) # (Px, Py) --> Px * Py = Number of cores
    
    # Create Cartesian Grid
    cartGrid =  comm.Create_cart(dims=partitionSize, periods=[False, False])
    
    # Get Rank Shifts (for communication between neighboring processes)
    rankShift = cartGrid.Shift(0, 1)
    rankShift += cartGrid.Shift(1, 1)
    
    # Get Subgrid Size and Position (for each process)
    subgrid_size, subgrid_pos = split_grid((Nx, Ny), partitionSize, cartGrid.Get_coords(rank), dx, dy)
    
    print(f'Rank {rank}\t | Subgrid Size: {subgrid_size}\t | Subgrid Position: ({subgrid_pos[0]:.4f},{subgrid_pos[1]:.4f})')
    