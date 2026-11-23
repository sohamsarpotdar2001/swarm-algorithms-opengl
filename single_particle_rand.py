import sys
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline

# 1. Simulation parameters
num_waypoints = 100
step_std = 0.5  # Standard deviation of random steps
smooth_factor = 10  # Interpolated points per step (higher = smoother)
trail_length = 20  # Keep only the last 4 trail points
sphere_radius = 0.2  # Visual radius of the particle

# 2. Generate 3D random walk waypoints
steps = np.random.normal(loc=0.0, scale=step_std, size=(num_waypoints, 3))
steps[0] = [0, 0, 0]
waypoints = np.cumsum(steps, axis=0)

# 3. Smoothen the trajectory using cubic B-spline interpolation
t_orig = np.linspace(0, 1, num_waypoints)
t_smooth = np.linspace(0, 1, num_waypoints * smooth_factor)
spline = make_interp_spline(t_orig, waypoints, k=3)
positions = spline(t_smooth)
total_frames = len(positions)

# 4. Precompute unit sphere coordinates
u = np.linspace(0, 2 * np.pi, 20)
v = np.linspace(0, np.pi, 15)
sphere_x = sphere_radius * np.outer(np.cos(u), np.sin(v))
sphere_y = sphere_radius * np.outer(np.sin(u), np.sin(v))
sphere_z = sphere_radius * np.outer(np.ones(np.size(u)), np.cos(v))

# 5. Set up the 3D plot
fig = plt.figure(figsize=(9, 7))
ax = fig.add_subplot(111, projection='3d')

max_range = np.max(np.abs(positions)) + sphere_radius * 2
ax.set_xlim(-max_range, max_range)
ax.set_ylim(-max_range, max_range)
ax.set_zlim(-max_range, max_range)
ax.set_box_aspect([1, 1, 1])

ax.set_title('Smoothed 3D Random Motion (Short Trail)', fontsize=14, pad=12)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Initialize trajectory line, start marker, and sphere surface
(trail_line,) = ax.plot(
    [], [], [], color='steelblue', lw=2.0, alpha=0.85, label='Recent Trail'
)
ax.scatter(0, 0, 0, color='green', s=50, label='Start')
sphere_surf = ax.plot_surface(
    sphere_x, sphere_y, sphere_z, color='crimson', alpha=0.9
)
ax.legend(loc='upper left')


# 6. Animation update function
def update(frame):
  global sphere_surf

  # Slice only the last `trail_length` points up to the current frame
  start_idx = max(0, frame - trail_length + 1)
  recent_pos = positions[start_idx : frame + 1]

  trail_line.set_data(recent_pos[:, 0], recent_pos[:, 1])
  trail_line.set_3d_properties(recent_pos[:, 2])

  # Update sphere position
  sphere_surf.remove()
  cx, cy, cz = positions[frame]
  sphere_surf = ax.plot_surface(
      sphere_x + cx,
      sphere_y + cy,
      sphere_z + cz,
      color='crimson',
      alpha=0.9,
      shade=True,
  )
  return trail_line, sphere_surf


# 7. Run animation
ani = animation.FuncAnimation(
    fig, update, frames=total_frames, interval=20, blit=False
)

try:
  plt.tight_layout()
  plt.show()
except KeyboardInterrupt:
  plt.close('all')
  sys.exit(0)
