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
