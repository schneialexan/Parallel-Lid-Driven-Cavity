import numpy as np
import matplotlib.pyplot as plt
import os

path = os.path.join(os.getcwd(), r'Parallel-Lid-Driven-Cavity\Alex\proof_of_concepts\data\\')
data_u = np.loadtxt(path+'u.dat')
data_v = np.loadtxt(path+'v.dat')
data_interpolated_u = np.zeros((data_u.shape[0], data_v.shape[1]))
data_interpolated_v = np.zeros((data_u.shape[0], data_v.shape[1]))

for i in range(data_interpolated_u.shape[0]-1):
    for j in range(data_interpolated_u.shape[1]-1):
        # Siehe Bild im Slack von stagered grid
        data_interpolated_u[i, j]  = (data_u[i, j-1] + data_u[i, j+1])/2
        data_interpolated_v[i, j]  = (data_v[i-1, j] + data_v[i+1, j])/2

data = np.sqrt(data_interpolated_u**2 + data_interpolated_v**2)

# rotate data 45 degrees to the left
data_interpolated_u = np.rot90(data_interpolated_u, k=3, axes=(0, 1))
data_interpolated_v = np.rot90(data_interpolated_v, k=3, axes=(0, 1))
data = np.rot90(data, k=3, axes=(0, 1))

plt.imshow(data)
plt.colorbar()

# draw quick and dirty vector field
st = 2 # stride factor
scale = 5.*np.max([np.max(data_interpolated_u), np.max(data_interpolated_v)])
x = np.linspace(0, data_interpolated_u.shape[0], data_interpolated_u.shape[0])
y = np.linspace(0, data_interpolated_u.shape[0], data_interpolated_u.shape[1])
X, Y = np.meshgrid(x, y)
plt.quiver(X[::st,::st] , Y[::st,::st], data_interpolated_u[::st,::st], data_interpolated_v[::st,::st], scale=scale, color='white')

plt.gca().invert_yaxis()
#plt.savefig('plot_velocity.png')
plt.show()