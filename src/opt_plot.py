import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Generate some sample data
np.random.seed(0)
M_data = np.random.rand(50) * 10 + 1
T_base_data = np.random.rand(50) * 20 + 5
n_data = np.random.rand(50) * 5

# Create a 3D scatter plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(M_data, T_base_data, n_data, c='r', marker='o')

ax.set_xlabel(r'$M$')
ax.set_ylabel(r'$T_{\text{base}}$')
ax.set_zlabel(r'$\vec{n}$')

# Define the constants
T_g = 100
T_QNS = 50  # T_QNS < T_g

# Create the surface M * T_base = T_g
M_surf = np.linspace(M_data.min(), M_data.max(), 20)
n_surf = np.linspace(n_data.min(), n_data.max(), 20)
M_surf, n_surf = np.meshgrid(M_surf, n_surf)
T_base_surf = T_g / M_surf

# Plot the T_g surface
ax.plot_surface(M_surf, T_base_surf, n_surf, alpha=0.2, color='blue')

# Create and plot the T_QNS plane
T_QNS_surf = np.full_like(M_surf, T_QNS)
ax.plot_surface(M_surf, T_QNS_surf, n_surf, alpha=0.2, color='green')

# Shade the region between the surfaces where T_base_surf > T_QNS_surf
for i in range(M_surf.shape[0]):
    for j in range(M_surf.shape[1]):
        if T_base_surf[i, j] > T_QNS:
            ax.plot([M_surf[i, j], M_surf[i, j]], [T_QNS, T_base_surf[i, j]], [n_surf[i, j], n_surf[i, j]], color='gray', alpha=0.5, linestyle=':')

plt.show()
