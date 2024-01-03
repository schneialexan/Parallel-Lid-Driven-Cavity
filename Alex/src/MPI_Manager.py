from typing import Tuple
from mpi4py import MPI
import sys

def validate_np(np: int, partition_size: Tuple[int, int]) -> bool:
    # Check if the number of processes is compatible with the number of domain partitions
    return np == partition_size[0] * partition_size[1]

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


def validate(size, rank, partitionSize, dt, dx, dy, Re):
    """
    Validates the input parameters for the lid-driven cavity simulation.

    Parameters:
    - size (int): Total number of processes.
    - rank (int): Rank of the current process.
    - partitionSize (tuple): Tuple containing the number of partitions in x and y directions.
    - dt (float): Time step.
    - dx (float): Grid spacing in the x direction.
    - dy (float): Grid spacing in the y direction.
    - Re (float): Reynolds number.

    Returns:
    None
    """
    if not validate_np(size, partitionSize):
        print("np =", size, "is not compatible with Px =", partitionSize[1], "and Py =", partitionSize[0])
        if rank == 0:
            print("\nPlease enter a valid combination of processes and domain partitions such that:")
            print("np = Px * Py\n")
            print("Program complete with exit code 0")
        MPI.Finalize()
        sys.exit(0)
    
    if dt >= 0.25 * dx * dy * Re:
        if rank == 0:
            print("\nInvalid time step dt =", dt)
            print("Time step must satisfy the following condition:")
            print("dt >= 0.25 * dx * dy * Re")
            print("For the chosen values of dx, dy, Re, please enter a time step such that:")
            print("dt <= ", 0.25 * dx * dy * Re, "\n")
            print("Program complete with exit code 0")
        MPI.Finalize()
        sys.exit(0)