import argparse

def LDCprogram_options():
    parser = argparse.ArgumentParser(description="Lid Driven Cavity Simulation program options", add_help=False)

    # Specify allowed options and default values
    parser.add_argument('--help', action='store_true', help="Produce help message.")
    parser.add_argument('--Nx', type=int, default=50, help="Number of grid points in x-direction.")
    parser.add_argument('--Ny', type=int, default=50, help="Number of grid points in y-direction.")
    parser.add_argument('--Px', type=int, default=1, help="Number of partitions in x-direction.")
    parser.add_argument('--Py', type=int, default=1, help="Number of partitions in y-direction.")
    parser.add_argument('--dt', type=float, default=0.005, help="Time step size.")
    parser.add_argument('--T', type=float, default=5.0, help="Final time.")
    parser.add_argument('--tau', type=float, default=0.5, help="Safety factor for time step size.")
    parser.add_argument('--eps', type=float, default=1e-3, help="Convergence criterion for the iterative solver.")
    parser.add_argument('--omg', type=float, default=1.7, help="Relaxation parameter for the iterative solver.")
    parser.add_argument('--itermax', type=int, default=100, help="Maximum number of iterations for the iterative solver.")
    parser.add_argument('--alpha', type=float, default=0.5, help="Upwind differencing factor.")
    parser.add_argument('--Re', type=float, default=100.0, help="Reynolds number.")

    args = parser.parse_args()

    # If the user gives the --help argument, print the help and quit.
    if args.help:
        print(parser.format_help())
        return True, None

    # Return false, and proceed with the rest of the program
    return False, args