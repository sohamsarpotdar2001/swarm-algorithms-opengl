import matplotlib.pyplot as plt
import numpy as np


def ackley(x, y, a=20, b=0.2, c=2 * np.pi):
  """Computes the 2D Ackley function."""
  term1 = -a * np.exp(-b * np.sqrt(0.5 * (x**2 + y**2)))
  term2 = -np.exp(0.5 * (np.cos(c * x) + np.cos(c * y)))
  return term1 + term2 + a + np.e


# 1. Create a grid of (x, y) values
# Standard domain for Ackley function is typically [-5, 5] or [-32.768, 32.768]
limit = 5
x = np.linspace(-limit, limit, 200)
y = np.linspace(-limit, limit, 200)
X, Y = np.meshgrid(x, y)

# 2. Evaluate the function over the grid
Z = ackley(X, Y)

# 3. Set up the 3D plot
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Plot the surface
surf = ax.plot_surface(
    X, Y, Z, cmap='viridis', edgecolor='none', antialiased=True
)

# 4. Customize labels, title, and colorbar
ax.set_title('3D Surface Plot of the Ackley Function', fontsize=14, pad=15)
ax.set_xlabel('X', fontsize=11)
ax.set_ylabel('Y', fontsize=11)
ax.set_zlabel('f(X, Y)', fontsize=11)

# Add a color bar to map values to colors
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label='Function Value')

# Adjust viewing angle (elevation, azimuth)
ax.view_init(elev=30, azim=45)

plt.tight_layout()
plt.show()
