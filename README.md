# Swarm Algorithms in C++ and OpenGL

A collection of swarm intelligence and multi-agent algorithms implemented in C++ and visualized with OpenGL.

## Algorithms

- Boids
- Consensus
- Particle Swarm Optimization (PSO)
- Potential Fields
- Ant Colony Optimization (ACO)
- Artificial Bee Colony (ABC)
- Formation Control

## Build
Install system dependencies (Assuming ubuntu installed)
```
sudo apt update
sudo apt install build-essential cmake libgl1-mesa-dev libglew-dev libglfw3-dev libglu1-mesa-dev
```

Clone the repo and compile the code

```
git clone https://github.com/sohamsarpotdar2001/swarm-algorithms-opengl.git
cd swarm-algorithms-opengl
g++ boids_gl.cpp -o boids -lGL -lGLU -lGLEW -lglfw
./boids
```
| Boids | Potential Fields |
|:---:|:---:|
| <img width="426" height="328" alt="Boids simulation" src="https://github.com/user-attachments/assets/fa2d69ae-e1ac-43b2-b200-03f3b7018d2d" /> | <img width="426" height="328" alt="Swarm simulation" src="https://github.com/user-attachments/assets/9bbdd997-ee0f-49a0-9035-4747c548357b" /> |
