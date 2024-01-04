from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if rank == 0:
    msg = "Hi 1, I am 0"
    comm.send(msg, dest=1)
    print("Rank 0 sent message")
elif rank == 1:
    msg = comm.recv(source=0)
    print("Rank 1 received message: ", msg)