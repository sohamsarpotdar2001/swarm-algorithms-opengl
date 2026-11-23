import signal
import sys
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

# 1. Simulation & Boids Parameters
num_boids = 25
num_steps = 600
trail_length = 15  # Keep only the last 4 trajectory points per boid
box_size = 10.0  # 3D bounding cube [-box_size, box_size]
sphere_radius = 0.2  # Visual radius of each spherical boid

# Boids steering weights & radii
max_speed = 0.35
max_force = 0.03
perception_radius = 3.0
separation_radius = 1.2

w_separation = 1.6
w_alignment = 1.0
w_cohesion = 1.0
w_noise = 0.15  # Subtle random perturbation for organic motion

# 2. Initialize Positions & Velocities
pos = np.random.uniform(-box_size * 0.7, box_size * 0.7, size=(num_boids, 3))
vel = np.random.normal(0, 0.1, size=(num_boids, 3))


def limit_vectors(vectors, max_val):
  """Clamps the magnitude of each row vector to max_val."""
  norms = np.linalg.norm(vectors, axis=1, keepdims=True)
  scale = np.where(norms > max_val, max_val / (norms + 1e-8), 1.0)
  return vectors * scale


def step_boids(positions, velocities):
  """Computes one vectorized step of 3D Boids flocking + soft boundary steering."""
  # Pairwise displacement vectors: diff[i, j] = pos[j] - pos[i]
  diff = positions[np.newaxis, :, :] - positions[:, np.newaxis, :]
  dist = np.linalg.norm(diff, axis=2)

  # Exclude self-interactions
  np.fill_diagonal(dist, np.inf)

  # --- Rule 1: Separation (steer away from crowding neighbors) ---
  sep_mask = dist < separation_radius
  # Repulsion inversely proportional to distance squared
  inv_dist_sq = np.where(sep_mask, 1.0 / (dist**2 + 1e-8), 0.0)
  sep_force = -np.sum(diff * inv_dist_sq[:, :, np.newaxis], axis=1)

  # --- Rule 2: Alignment (steer towards average heading of neighbors) ---
  align_mask = dist < perception_radius
  neighbor_counts = np.sum(align_mask, axis=1, keepdims=True)
  avg_vel = (align_mask @ velocities) / np.maximum(neighbor_counts, 1)
  align_force = np.where(neighbor_counts > 0, avg_vel - velocities, 0.0)

  # --- Rule 3: Cohesion (steer towards center of mass of neighbors) ---
  center_of_mass = (align_mask @ positions) / np.maximum(neighbor_counts, 1)
  coh_force = np.where(neighbor_counts > 0, center_of_mass - positions, 0.0)

  # --- Rule 4: Soft Boundary Avoidance (smoothly turn away from box walls) ---
  margin = box_size * 0.75
  turn_factor = 0.04
  bound_force = np.zeros_like(positions)
  bound_force += np.where(positions < -margin, turn_factor, 0.0)
  bound_force -= np.where(positions > margin, turn_factor, 0.0)

  # --- Random Walk / Brownian Jitter ---
  noise_force = np.random.normal(0, 1.0, size=positions.shape)

  # Combine forces & apply steering limits
  total_force = (
      w_separation * limit_vectors(sep_force, max_force)
      + w_alignment * limit_vectors(align_force, max_force)
      + w_cohesion * limit_vectors(coh_force, max_force)
      + w_noise * limit_vectors(noise_force, max_force)
      + bound_force
  )

  # Update velocity (inertia naturally smoothens the trajectory) and position
  new_vel = limit_vectors(velocities + total_force, max_speed)
  new_pos = positions + new_vel
  return new_pos, new_vel


# 3. Precompute the full 3D Boids trajectory
history = np.zeros((num_steps, num_boids, 3))
history[0] = pos
for t in range(1, num_steps):
  pos, vel = step_boids(pos, vel)
  history[t] = pos

# 4. Precompute low-poly unit sphere mesh for fast rendering of multiple boids
u = np.linspace(0, 2 * np.pi, 10)
v = np.linspace(0, np.pi, 8)
sphere_x = sphere_radius * np.outer(np.cos(u), np.sin(v))
sphere_y = sphere_radius * np.outer(np.sin(u), np.sin(v))
sphere_z = sphere_radius * np.outer(np.ones(np.size(u)), np.cos(v))

# 5. Set up the 3D Plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")
ax.set_xlim(-box_size, box_size)
ax.set_ylim(-box_size, box_size)
ax.set_zlim(-box_size, box_size)
ax.set_box_aspect([1, 1, 1])

ax.set_title("3D Boids Flocking Simulation (Spherical Particles)", fontsize=14)
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

colors = plt.cm.plasma(np.linspace(0.15, 0.85, num_boids))

# Initialize 4-point trail lines and 3D sphere surfaces for every boid
trail_lines = [
    ax.plot([], [], [], color=colors[i], lw=1.8, alpha=0.85)[0]
    for i in range(num_boids)
]
sphere_surfs = [
    ax.plot_surface(
        sphere_x + history[0, i, 0],
        sphere_y + history[0, i, 1],
        sphere_z + history[0, i, 2],
        color=colors[i],
        shade=True,
    )
    for i in range(num_boids)
]


# 6. Animation Update Function
def update(frame):
  start_idx = max(0, frame - trail_length + 1)

  for i in range(num_boids):
    # Update sliding 4-point trail
    recent = history[start_idx : frame + 1, i, :]
    trail_lines[i].set_data(recent[:, 0], recent[:, 1])
    trail_lines[i].set_3d_properties(recent[:, 2])

    # Update spherical particle position
    sphere_surfs[i].remove()
    cx, cy, cz = history[frame, i]
    sphere_surfs[i] = ax.plot_surface(
        sphere_x + cx,
        sphere_y + cy,
        sphere_z + cz,
        color=colors[i],
        shade=True,
    )

  return trail_lines + sphere_surfs


# 7. Clean Ctrl + C (SIGINT) Exit Handling
def handle_sigint(sig, frame):
  plt.close("all")
  print("\nSimulation stopped via Ctrl+C. Exiting cleanly (code 130)...")
  sys.exit(130)


signal.signal(signal.SIGINT, handle_sigint)

if __name__ == "__main__":
  try:
    ani = animation.FuncAnimation(
        fig, update, frames=num_steps, interval=25, blit=False
    )
    plt.tight_layout()
    plt.show()
  except KeyboardInterrupt:
    handle_sigint(None, None)
