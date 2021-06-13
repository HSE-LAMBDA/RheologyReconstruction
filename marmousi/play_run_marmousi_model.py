import numpy as np
import matplotlib.pyplot as plt

from fenics import File
from pipeline.dolfin_adjoint.elasticity_solver import elasticity_solver
from marmousi.marmousi2_tools import read_data

config_path = 'play_run_marmousi_model_solver_config.yaml'

# mask = np.ones(shape=(128,128))
# la = mask * 1.0e+10
# mu = mask * 7.0e+9
# rho = mask * 1.0e+3
# la[100:110,48:80] -= 8.0e+9
# mu[100:110,48:80] -= 5.0e+9
# rho[100:110,48:80] += 0.0e+3

start_z = 462.5
stop_z = 3500.0
start_x = 2000
stop_x = 8000
coarse_factor = 20

rho, cp_coeffs, cs_coeffs, la, mu = read_data(start_z=start_z, stop_z=stop_z,
                                              start_x=start_x, stop_x=stop_x,
                                              coarse_factor=coarse_factor)
print("Mesh shape:", rho.shape)

res_v_file = File("data/v.pvd")
def save(timestep, curr_time, u_field, v_field, a_field):
    #if timestep % 1 == 0:
    print("Dumping v snapg, time step %d" % timestep)
    res_v_file << v_field

slvr = elasticity_solver(la.T, mu.T, rho.T, config_path)
slvr.forward(save_callback=save)

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)

for (ax, data, title) in [(ax1, la, "lambda"), (ax2, mu, "mu"), (ax3, np.sqrt((la + 2 * mu) / rho), "Cp"), (ax4, np.sqrt(mu / rho), "Cs")]:
    ax.set_aspect('equal')
    im = ax.pcolormesh(data)
    fig.colorbar(im, ax=ax)
    ax.set_title(title)

fig.tight_layout()
plt.show()