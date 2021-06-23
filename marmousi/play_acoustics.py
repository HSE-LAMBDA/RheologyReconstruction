import numpy as np
import matplotlib.pyplot as plt
from pipeline.acoustics_solver import acoustics_solver

# This sample will run in less than a minute
x_size = 20
y_size = 20
cp = 10 * np.ones((200, 200), dtype=np.float64)
cp[:,100:] *= 2
rho = 1 * np.ones((200, 200), dtype=np.float64)

slvr = acoustics_solver(x_size, y_size, cp, rho, 1.25, 0.01, 2, dump_vtk=False, verbose=False)
data = slvr.forward()

# Just show the recorded data
fig, ax = plt.subplots()
ax.set_aspect('equal')
ax.pcolormesh(data, vmin=-0.5, vmax=0.5)
plt.show()