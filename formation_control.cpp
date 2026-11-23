// formation_triangle_gl.cpp
// Dependencies: GLFW, GLEW, OpenGL (compatibility profile / GLU)
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <GL/glu.h>

#include <algorithm>
#include <cmath>
#include <csignal>
#include <deque>
#include <iostream>
#include <random>
#include <vector>

volatile std::sig_atomic_t g_stop_requested = 0;
void handle_sigint(int) { g_stop_requested = 1; }

struct Vec3 {
    double x = 0.0, y = 0.0, z = 0.0;

    Vec3 operator+(const Vec3& o) const { return {x + o.x, y + o.y, z + o.z}; }
    Vec3 operator-(const Vec3& o) const { return {x - o.x, y - o.y, z - o.z}; }
    Vec3 operator*(double s) const { return {x * s, y * s, z * s}; }
    Vec3 operator/(double s) const { return {x / s, y / s, z / s}; }

    Vec3& operator+=(const Vec3& o) { x += o.x; y += o.y; z += o.z; return *this; }
    Vec3& operator-=(const Vec3& o) { x -= o.x; y -= o.y; z -= o.z; return *this; }

    double length_sq() const { return x * x + y * y + z * z; }
    double length() const { return std::sqrt(length_sq()); }

    Vec3 limited(double max_val) const {
        double len = length();
        return (len > max_val && len > 1e-8) ? (*this * (max_val / len)) : *this;
    }
};

struct Color3 {
    float r, g, b;
};

struct Particle {
    Vec3 pos;
    Vec3 vel;
    Vec3 offset;          // Desired position relative to formation centroid
    Color3 color;
    std::deque<Vec3> trail;
};

struct FormationParams {
    size_t num_particles = 12;     // Try 3 for pure vertices, or 6, 9, 12, 15... for perimeter triangle
    size_t trail_length = 4;       // Sliding window of last 4 trajectory points
    double triangle_radius = 4.5;  // Circumradius of the equilateral triangle
    double box_size = 10.0;
    double sphere_radius = 0.28;

    // Control Gains (Second-order consensus + collision avoidance)
    double k_consensus = 0.035;    // Inter-agent relative formation stiffness
    double k_leader    = 0.025;    // Tracking gain toward moving formation center
    double k_damp      = 0.18;     // Velocity damping to eliminate overshoot
    double k_repel     = 0.08;     // Short-range collision avoidance gain
    double safe_dist   = 0.9;      // Minimum desired spacing between particles

    double max_speed   = 0.30;
    double max_accel   = 0.04;
};

// Generates N target offsets evenly spaced along an equilateral triangle centered at (0,0,0)
std::vector<Vec3> generate_triangle_offsets(size_t n, double radius) {
    std::vector<Vec3> offsets;
    offsets.reserve(n);
    if (n == 0) return offsets;

    // 3 vertices of an equilateral triangle in the X-Z plane
    Vec3 v[3];
    for (int k = 0; k < 3; ++k) {
        double angle = (2.0 * M_PI * k) / 3.0 - M_PI / 2.0;
        v[k] = {radius * std::cos(angle), 0.0, radius * std::sin(angle)};
    }

    if (n <= 3) {
        for (size_t i = 0; i < n; ++i) offsets.push_back(v[i]);
        return offsets;
    }

    // Distribute N points uniformly along the 3 edges of the triangle
    for (size_t i = 0; i < n; ++i) {
        double u = (3.0 * static_cast<double>(i)) / static_cast<double>(n);
        int edge = std::min(2, static_cast<int>(u));
        double t = u - static_cast<double>(edge);
        Vec3 p0 = v[edge];
        Vec3 p1 = v[(edge + 1) % 3];
        offsets.push_back(p0 * (1.0 - t) + p1 * t);
    }
    return offsets;
}

class TriangleFormation {
public:
    explicit TriangleFormation(const FormationParams& params)
        : params_(params), rng_(std::random_device{}()) {
        std::uniform_real_distribution<double> pos_dist(-params_.box_size * 0.7, params_.box_size * 0.7);
        auto offsets = generate_triangle_offsets(params_.num_particles, params_.triangle_radius);

        particles_.resize(params_.num_particles);
        for (size_t i = 0; i < params_.num_particles; ++i) {
            particles_[i].pos = {pos_dist(rng_), pos_dist(rng_), pos_dist(rng_)};
            particles_[i].vel = {0.0, 0.0, 0.0};
            particles_[i].offset = offsets[i];
            particles_[i].trail.push_back(particles_[i].pos);

            float t = static_cast<float>(i) / std::max<size_t>(1, params_.num_particles - 1);
            particles_[i].color = {
                0.10f + 0.75f * t,
                0.22f + 0.45f * (1.0f - std::abs(2.0f * t - 1.0f)),
                0.85f - 0.65f * t
            };
        }
    }

    void step(double time_sec) {
        const size_t n = particles_.size();
        std::vector<Vec3> next_vel(n);

        // Smoothly moving virtual leader (centroid of the triangle formation)
        Vec3 leader_pos{
            3.5 * std::sin(0.5 * time_sec),
            1.8 * std::sin(0.8 * time_sec),
            3.5 * std::cos(0.5 * time_sec)
        };
        Vec3 leader_vel{
            3.5 * 0.5 * std::cos(0.5 * time_sec) * 0.016,
            1.8 * 0.8 * std::cos(0.8 * time_sec) * 0.016,
           -3.5 * 0.5 * std::sin(0.5 * time_sec) * 0.016
        };
        current_leader_ = leader_pos;

        // Rotate formation yaw slowly to demonstrate rigid-body consensus tracking
        double yaw = 0.35 * time_sec;
        double cy = std::cos(yaw), sy = std::sin(yaw);
        auto rotate_y = [&](const Vec3& v) -> Vec3 {
            return {v.x * cy + v.z * sy, v.y, -v.x * sy + v.z * cy};
        };

        const double safe_sq = params_.safe_dist * params_.safe_dist;

        for (size_t i = 0; i < n; ++i) {
            Vec3 r_i = rotate_y(particles_[i].offset);
            Vec3 u_consensus{0.0, 0.0, 0.0};
            Vec3 u_repel{0.0, 0.0, 0.0};

            // 1. Distributed Consensus + Collision Avoidance over neighbors
            for (size_t j = 0; j < n; ++j) {
                if (i == j) continue;
                Vec3 r_j = rotate_y(particles_[j].offset);

                // Relative formation error: (p_j - p_i) - (r_j - r_i)
                Vec3 rel_pos_err = (particles_[j].pos - particles_[i].pos) - (r_j - r_i);
                Vec3 rel_vel_err = (particles_[j].vel - particles_[i].vel);
                u_consensus += rel_pos_err * params_.k_consensus + rel_vel_err * (params_.k_damp * 0.25);

                // Short-range repulsion to prevent particle collisions during convergence
                Vec3 diff = particles_[i].pos - particles_[j].pos;
                double d_sq = diff.length_sq();
                if (d_sq < safe_sq && d_sq > 1e-8) {
                    double d = std::sqrt(d_sq);
                    u_repel += (diff / d) * (params_.k_repel * (params_.safe_dist - d) / d);
                }
            }
            if (n > 1) {
                u_consensus = u_consensus / static_cast<double>(n - 1);
            }

            // 2. Leader-Follower Anchor & Velocity Damping
            Vec3 target_i = leader_pos + r_i;
            Vec3 u_leader = (target_i - particles_[i].pos) * params_.k_leader
                          + (leader_vel - particles_[i].vel) * params_.k_damp;

            Vec3 accel = (u_consensus + u_leader + u_repel).limited(params_.max_accel);
            next_vel[i] = (particles_[i].vel + accel).limited(params_.max_speed);
        }

        // Synchronous position and 4-point trail update
        for (size_t i = 0; i < n; ++i) {
            particles_[i].vel = next_vel[i];
            particles_[i].pos += particles_[i].vel;

            particles_[i].trail.push_back(particles_[i].pos);
            if (particles_[i].trail.size() > params_.trail_length) {
                particles_[i].trail.pop_front();
            }
        }
    }

    const std::vector<Particle>& particles() const { return particles_; }
    const Vec3& leader() const { return current_leader_; }

private:
    FormationParams params_;
    std::vector<Particle> particles_;
    Vec3 current_leader_{0.0, 0.0, 0.0};
    std::mt19937 rng_;
};

// Interactive Orbit Camera
struct Camera {
    float yaw = 35.0f;
    float pitch = 32.0f;
    float distance = 28.0f;
    bool dragging = false;
    double last_x = 0.0, last_y = 0.0;
} g_cam;

void mouse_button_cb(GLFWwindow* win, int button, int action, int) {
    if (button == GLFW_MOUSE_BUTTON_LEFT) {
        g_cam.dragging = (action == GLFW_PRESS);
        glfwGetCursorPos(win, &g_cam.last_x, &g_cam.last_y);
    }
}

void cursor_pos_cb(GLFWwindow*, double x, double y) {
    if (g_cam.dragging) {
        g_cam.yaw += static_cast<float>(x - g_cam.last_x) * 0.4f;
        g_cam.pitch += static_cast<float>(y - g_cam.last_y) * 0.4f;
        g_cam.pitch = std::clamp(g_cam.pitch, -89.0f, 89.0f);
        g_cam.last_x = x;
        g_cam.last_y = y;
    }
}

void scroll_cb(GLFWwindow*, double, double yoffset) {
    g_cam.distance = std::clamp(g_cam.distance - static_cast<float>(yoffset) * 1.5f, 8.0f, 80.0f);
}

void draw_wireframe_box(double s) {
    glDisable(GL_LIGHTING);
    glColor4f(0.25f, 0.28f, 0.35f, 0.35f);
    glLineWidth(1.0f);
    glBegin(GL_LINES);
    double c[2] = {-s, s};
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            glVertex3d(c[0], c[i], c[j]); glVertex3d(c[1], c[i], c[j]);
            glVertex3d(c[i], c[0], c[j]); glVertex3d(c[i], c[1], c[j]);
            glVertex3d(c[i], c[j], c[0]); glVertex3d(c[i], c[j], c[1]);
        }
    }
    glEnd();
    glEnable(GL_LIGHTING);
}

int main() {
    std::signal(SIGINT, handle_sigint);

    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW\n";
        return 1;
    }

    GLFWwindow* window = glfwCreateWindow(1024, 768, "3D Triangle Formation Control (N Particles)", nullptr, nullptr);
    if (!window) {
        glfwTerminate();
        return 1;
    }

    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    if (glewInit() != GLEW_OK) {
        std::cerr << "Failed to initialize GLEW\n";
        return 1;
    }

    glfwSetMouseButtonCallback(window, mouse_button_cb);
    glfwSetCursorPosCallback(window, cursor_pos_cb);
    glfwSetScrollCallback(window, scroll_cb);

    // Bright White Studio Lighting
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_LIGHTING);
    glEnable(GL_LIGHT0);
    glEnable(GL_COLOR_MATERIAL);
    glColorMaterial(GL_FRONT, GL_AMBIENT_AND_DIFFUSE);

    GLfloat global_amb[] = {0.45f, 0.45f, 0.45f, 1.0f};
    glLightModelfv(GL_LIGHT_MODEL_AMBIENT, global_amb);

    GLfloat light_pos[]  = {15.0f, 30.0f, 20.0f, 1.0f};
    GLfloat light_amb[]  = {0.35f, 0.35f, 0.35f, 1.0f};
    GLfloat light_dif[]  = {1.00f, 1.00f, 1.00f, 1.0f};
    GLfloat light_spec[] = {0.80f, 0.80f, 0.80f, 1.0f};
    glLightfv(GL_LIGHT0, GL_POSITION, light_pos);
    glLightfv(GL_LIGHT0, GL_AMBIENT, light_amb);
    glLightfv(GL_LIGHT0, GL_DIFFUSE, light_dif);
    glLightfv(GL_LIGHT0, GL_SPECULAR, light_spec);

    GLUquadric* quad = gluNewQuadric();
    gluQuadricNormals(quad, GLU_SMOOTH);

    FormationParams params;
    TriangleFormation formation(params);

    double sim_time = 0.0;
    while (!glfwWindowShouldClose(window)) {
        if (g_stop_requested || glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
            break;
        }

        sim_time += 0.016;
        formation.step(sim_time);

        int width, height;
        glfwGetFramebufferSize(window, &width, &height);
        glViewport(0, 0, width, height);

        glClearColor(0.97f, 0.98f, 0.99f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        double aspect = static_cast<double>(width) / std::max(1, height);
        gluPerspective(45.0, aspect, 0.1, 200.0);

        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();
        double rad_yaw = g_cam.yaw * M_PI / 180.0;
        double rad_pitch = g_cam.pitch * M_PI / 180.0;
        double cam_x = g_cam.distance * std::cos(rad_pitch) * std::sin(rad_yaw);
        double cam_y = g_cam.distance * std::sin(rad_pitch);
        double cam_z = g_cam.distance * std::cos(rad_pitch) * std::cos(rad_yaw);
        gluLookAt(cam_x, cam_y, cam_z, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0);

        draw_wireframe_box(params.box_size);

        // Draw subtle formation perimeter links between adjacent particles
        const auto& pts = formation.particles();
        if (pts.size() >= 3) {
            glDisable(GL_LIGHTING);
            glColor4f(0.30f, 0.35f, 0.45f, 0.35f);
            glLineWidth(1.5f);
            glBegin(GL_LINE_LOOP);
            for (const auto& p : pts) {
                glVertex3d(p.pos.x, p.pos.y, p.pos.z);
            }
            glEnd();
            glEnable(GL_LIGHTING);
        }

        // Draw Particles & 4-Point Trails
        for (const auto& p : pts) {
            glPushMatrix();
            glTranslated(p.pos.x, p.pos.y, p.pos.z);
            glColor3f(p.color.r, p.color.g, p.color.b);
            gluSphere(quad, params.sphere_radius, 16, 12);
            glPopMatrix();

            if (p.trail.size() > 1) {
                glDisable(GL_LIGHTING);
                glLineWidth(2.2f);
                glBegin(GL_LINE_STRIP);
                for (size_t k = 0; k < p.trail.size(); ++k) {
                    float alpha = static_cast<float>(k + 1) / static_cast<float>(p.trail.size());
                    glColor4f(p.color.r, p.color.g, p.color.b, alpha * 0.9f);
                    glVertex3d(p.trail[k].x, p.trail[k].y, p.trail[k].z);
                }
                glEnd();
                glEnable(GL_LIGHTING);
            }
        }

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    gluDeleteQuadric(quad);
    glfwDestroyWindow(window);
    glfwTerminate();

    if (g_stop_requested) {
        std::cout << "\nSimulation interrupted via Ctrl+C. Exiting cleanly (code 130)...\n";
        return 130;
    }
    return 0;
}
