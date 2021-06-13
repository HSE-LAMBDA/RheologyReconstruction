import matplotlib.pyplot as plt

from marmousi.layered_structures_tools import get_mask

# The mask will be here
mask = get_mask(170, 35, 2, 8, 5, 10)

# Draw the mask
fig, ax = plt.subplots()
ax.set_aspect('equal')
im = ax.pcolormesh(mask.T)
fig.colorbar(im, ax=ax, orientation='horizontal')
ax.set_title("mask")
plt.show()