// pso_ackley_gl.cpp
// Dependencies: GLFW, GLEW, OpenGL (compatibility profile / GLU)
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <GL/glu.h>

#include <algorithm>
#include <cmath>
#include <csignal>
#include <deque>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <vector>

volatile std::sig_atomic_t g_stop_requested = 0;
void handle_sigint(int) { g_stop_requested = 1; }

struct Vec3 {
    double x = 0.0, y = 0.0, z = 0.0;

    Vec3 operator+(const Vec3& o) const { return {x + o.x, y + o.y, z + o.z}; }
    Vec3 operator-(const Vec3& o) const { return {x - o.x, y - o.y, z - o.z}; }
    Vec3 operator*(double s) const { return {x * s, y * s, z * s}; }

    Vec3& operator+=(const Vec3& o) { x += o.x; y += o.y; z += o.z; return *this; }

    double length() const { return std::sqrt(x * x + y * y + z * z); }

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
    Vec3 pbest_pos;
    double pbest_val = std::numeric_limits<double>::infinity();
    Color3 color;
    std::deque<Vec3> trail;
};

struct PSOParams {
    size_t num_particles = 40;
    size_t trail_length = 4;      // Sliding window of last 4 positions
    double domain_limit = 5.0;    // Search bounds [-5, 5]^3
    double visual_scale = 1.8;    // Scale [-5, 5] -> [-9, 9] for display
    double sphere_radius = 0.24;

    // Standard Constriction / Inertia PSO Hyperparameters
    double w_inertia = 0.72;      // Inertia weight
    double c1_cognitive = 1.49;   // Personal best attraction
    double c2_social = 1.49;      // Global best attraction
    double max_vel = 0.35;        // Velocity clamp in search space
};

// 3D Ackley Benchmark Function (Global minimum = 0.0 at (0, 0, 0))
double ackley_3d(const Vec3& p) {
    constexpr double a = 20.0;
    constexpr double b = 0.2;
    constexpr double c = 2.0 * M_PI;

    double sum_sq = (p.x * p.x + p.y * p.y + p.z * p.z) / 3.0;
    double sum_cos = (std::cos(c * p.x) + std::cos(c * p.y) + std::cos(c * p.z)) / 3.0;

    return -a * std::exp(-b * std::sqrt(sum_sq)) - std::exp(sum_cos) + a + std::exp(1.0);
}

class PSOSwarm {
public:
    explicit PSOSwarm(const PSOParams& params)
        : params_(params), rng_(std::random_device{}()) {
        reset();
    }

    void reset() {
        std::uniform_real_distribution<double> pos_dist(-params_.domain_limit, params_.domain_limit);
        std::uniform_real_distribution<double> vel_dist(-0.2, 0.2);

        gbest_val_ = std::numeric_limits<double>::infinity();
        gbest_pos_ = {0.0, 0.0, 0.0};
        iteration_ = 0;

        particles_.assign(params_.num_particles, Particle{});
        for (size_t i = 0; i < params_.num_particles; ++i) {
            particles_[i].pos = {pos_dist(rng_), pos_dist(rng_), pos_dist(rng_)};
            particles_[i].vel = {vel_dist(rng_), vel_dist(rng_), vel_dist(rng_)};
            particles_[i].pbest_pos = particles_[i].pos;
            particles_[i].pbest_val = ackley_3d(particles_[i].pos);
            particles_[i].trail.push_back(particles_[i].pos);

            if (particles_[i].pbest_val < gbest_val_) {
                gbest_val_ = particles_[i].pbest_val;
                gbest_pos_ = particles_[i].pbest_pos;
            }

            float t = static_cast<float>(i) / std::max<size_t>(1, params_.num_particles - 1);
            particles_[i].color = {
                0.10f + 0.78f * t,
                0.22f + 0.45f * (1.0f - std::abs(2.0f * t - 1.0f)),
                0.85f - 0.68f * t
            };
        }
    }

    void step() {
        std::uniform_real_distribution<double> uni01(0.0, 1.0);

        for (auto& p : particles_) {
            double r1 = uni01(rng_);
            double r2 = uni01(rng_);

            Vec3 cognitive = (p.pbest_pos - p.pos) * (params_.c1_cognitive * r1);
            Vec3 social    = (gbest_pos_ - p.pos) * (params_.c2_social * r2);

            p.vel = (p.vel * params_.w_inertia + cognitive + social).limited(params_.max_vel);
            p.pos += p.vel;

            // Clamp particle position to search domain [-domain_limit, domain_limit]
            p.pos.x = std::clamp(p.pos.x, -params_.domain_limit, params_.domain_limit);
            p.pos.y = std::clamp(p.pos.y, -params_.domain_limit, params_.domain_limit);
            p.pos.z = std::clamp(p.pos.z, -params_.domain_limit, params_.domain_limit);

            // Evaluate fitness and update personal & global bests
            double fitness = ackley_3d(p.pos);
            if (fitness < p.pbest_val) {
                p.pbest_val = fitness;
                p.pbest_pos = p.pos;

                if (fitness < gbest_val_) {
                    gbest_val_ = fitness;
                    gbest_pos_ = p.pos;
                }
            }

            p.trail.push_back(p.pos);
            if (p.trail.size() > params_.trail_length) {
                p.trail.pop_front();
            }
        }
        ++iteration_;
    }

    const std::vector<Particle>& particles() const { return particles_; }
    const Vec3& gbest_pos() const { return gbest_pos_; }
    double gbest_val() const { return gbest_val_; }
    int iteration() const { return iteration_; }
    const PSOParams& params() const { return params_; }

private:
    PSOParams params_;
    std::vector<Particle> particles_;
    Vec3 gbest_pos_{0.0, 0.0, 0.0};
    double gbest_val_ = std::numeric_limits<double>::infinity();
    int iteration_ = 0;
    std::mt19937 rng_;
};

// Interactive Orbit Camera
struct Camera {
    float yaw = 40.0f;
    float pitch = 26.0f;
    float distance = 30.0f;
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

    GLFWwindow* window = glfwCreateWindow(1024, 768, "3D Particle Swarm Optimization (Ackley Function) - Press R to Reset", nullptr, nullptr);
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

    PSOParams params;
    PSOSwarm swarm(params);

    bool r_was_pressed = false;
    std::cout << std::fixed << std::setprecision(6);

    while (!glfwWindowShouldClose(window)) {
        if (g_stop_requested || glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
            break;
        }

        // Press 'R' to re-scatter particles across the search space
        bool r_pressed = (glfwGetKey(window, GLFW_KEY_R) == GLFW_PRESS);
        if (r_pressed && !r_was_pressed) {
            swarm.reset();
            std::cout << "--- Swarm Reset ---\n";
        }
        r_was_pressed = r_pressed;

        swarm.step();

        if (swarm.iteration() % 30 == 0) {
            const auto& gb = swarm.gbest_pos();
            std::cout << "Iter " << std::setw(4) << swarm.iteration()
                      << " | Global Best f(x): " << swarm.gbest_val()
                      << " at (" << gb.x << ", " << gb.y << ", " << gb.z << ")\n";
        }

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

        const double s = params.visual_scale;
        draw_wireframe_box(params.domain_limit * s);

        // 1. Render True Global Minimum Target at Origin (0,0,0) as a subtle green marker
        glPushMatrix();
        glColor4f(0.10f, 0.72f, 0.32f, 0.45f);
        gluSphere(quad, params.sphere_radius * 1.4, 16, 12);
        glPopMatrix();

        // 2. Render Current Swarm Global Best (gbest) in Bright Gold
        const Vec3& gb = swarm.gbest_pos();
        glPushMatrix();
        glTranslated(gb.x * s, gb.y * s, gb.z * s);
        glColor3f(0.95f, 0.72f, 0.05f);
        gluSphere(quad, params.sphere_radius * 1.25, 16, 12);
        glPopMatrix();

        // 3. Render Swarm Particles & 4-Point Trails
        for (const auto& p : swarm.particles()) {
            glPushMatrix();
            glTranslated(p.pos.x * s, p.pos.y * s, p.pos.z * s);
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
                    glVertex3d(p.trail[k].x * s, p.trail[k].y * s, p.trail[k].z * s);
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
