// boids_gl_white.cpp
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

struct Boid {
    Vec3 pos;
    Vec3 vel;
    Color3 color;
    std::deque<Vec3> trail;
};

struct SimParams {
    size_t num_boids = 25;
    size_t trail_length = 15;
    double box_size = 10.0;
    double sphere_radius = 0.2;
    double max_speed = 0.25;
    double max_force = 0.025;
    double perception_radius = 3.2;
    double separation_radius = 1.3;

    double w_separation = 1.6;
    double w_alignment = 1.0;
    double w_cohesion = 1.0;
    double w_noise = 0.12;
};

class Flock {
public:
    explicit Flock(const SimParams& params)
        : params_(params), rng_(std::random_device{}()) {
        std::uniform_real_distribution<double> pos_dist(-params_.box_size * 0.7, params_.box_size * 0.7);
        std::normal_distribution<double> vel_dist(0.0, 0.08);

        boids_.resize(params_.num_boids);
        for (size_t i = 0; i < params_.num_boids; ++i) {
            boids_[i].pos = {pos_dist(rng_), pos_dist(rng_), pos_dist(rng_)};
            boids_[i].vel = {vel_dist(rng_), vel_dist(rng_), vel_dist(rng_)};
            boids_[i].trail.push_back(boids_[i].pos);

            // Rich saturated palette that pops against a white background
            float t = static_cast<float>(i) / std::max<size_t>(1, params_.num_boids - 1);
            boids_[i].color = {0.1f + 0.75f * t, 0.2f + 0.4f * (1.0f - std::abs(2.0f * t - 1.0f)), 0.85f - 0.7f * t};
        }
    }

    void step() {
        std::normal_distribution<double> noise_dist(0.0, 1.0);
        std::vector<Vec3> next_vel(boids_.size());

        const double perc_sq = params_.perception_radius * params_.perception_radius;
        const double sep_sq = params_.separation_radius * params_.separation_radius;
        const double margin = params_.box_size * 0.75;
        const double turn_factor = 0.035;

        for (size_t i = 0; i < boids_.size(); ++i) {
            Vec3 sep_force{0, 0, 0};
            Vec3 avg_vel{0, 0, 0};
            Vec3 center_of_mass{0, 0, 0};
            int neighbor_count = 0;

            for (size_t j = 0; j < boids_.size(); ++j) {
                if (i == j) continue;

                Vec3 diff = boids_[j].pos - boids_[i].pos;
                double d_sq = diff.length_sq();

                if (d_sq < sep_sq && d_sq > 1e-8) {
                    sep_force -= diff / d_sq;
                }
                if (d_sq < perc_sq) {
                    avg_vel += boids_[j].vel;
                    center_of_mass += boids_[j].pos;
                    ++neighbor_count;
                }
            }

            Vec3 align_force{0, 0, 0};
            Vec3 coh_force{0, 0, 0};
            if (neighbor_count > 0) {
                double inv_n = 1.0 / static_cast<double>(neighbor_count);
                align_force = (avg_vel * inv_n) - boids_[i].vel;
                coh_force = (center_of_mass * inv_n) - boids_[i].pos;
            }

            Vec3 bound_force{0, 0, 0};
            if (boids_[i].pos.x < -margin) bound_force.x += turn_factor;
            if (boids_[i].pos.x >  margin) bound_force.x -= turn_factor;
            if (boids_[i].pos.y < -margin) bound_force.y += turn_factor;
            if (boids_[i].pos.y >  margin) bound_force.y -= turn_factor;
            if (boids_[i].pos.z < -margin) bound_force.z += turn_factor;
            if (boids_[i].pos.z >  margin) bound_force.z -= turn_factor;

            Vec3 noise_force{noise_dist(rng_), noise_dist(rng_), noise_dist(rng_)};

            Vec3 total_force =
                (sep_force.limited(params_.max_force) * params_.w_separation) +
                (align_force.limited(params_.max_force) * params_.w_alignment) +
                (coh_force.limited(params_.max_force) * params_.w_cohesion) +
                (noise_force.limited(params_.max_force) * params_.w_noise) +
                bound_force;

            next_vel[i] = (boids_[i].vel + total_force).limited(params_.max_speed);
        }

        for (size_t i = 0; i < boids_.size(); ++i) {
            boids_[i].vel = next_vel[i];
            boids_[i].pos += boids_[i].vel;

            boids_[i].trail.push_back(boids_[i].pos);
            if (boids_[i].trail.size() > params_.trail_length) {
                boids_[i].trail.pop_front();
            }
        }
    }

    const std::vector<Boid>& boids() const { return boids_; }
    const SimParams& params() const { return params_; }

private:
    SimParams params_;
    std::vector<Boid> boids_;
    std::mt19937 rng_;
};

// Interactive Orbit Camera State
struct Camera {
    float yaw = 45.0f;
    float pitch = 25.0f;
    float distance = 32.0f;
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
    g_cam.distance = std::clamp(g_cam.distance - static_cast<float>(yoffset) * 1.5f, 10.0f, 80.0f);
}

void draw_wireframe_box(double s) {
    glDisable(GL_LIGHTING);
    // Dark slate gray lines so the bounding box stays visible on a white background
    glColor4f(0.25f, 0.28f, 0.35f, 0.45f);
    glLineWidth(1.2f);
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

    GLFWwindow* window = glfwCreateWindow(1024, 768, "3D Boids Flocking Simulation", nullptr, nullptr);
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

    // OpenGL Depth, Blending & Bright White Studio Lighting Setup
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_LIGHTING);
    glEnable(GL_LIGHT0);
    glEnable(GL_COLOR_MATERIAL);
    glColorMaterial(GL_FRONT, GL_AMBIENT_AND_DIFFUSE);

    // Global scene ambient white light + bright overhead key light + specular highlights
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

    // Subtle specular shine on the spheres
    GLfloat mat_spec[] = {0.5f, 0.5f, 0.5f, 1.0f};
    glMaterialfv(GL_FRONT, GL_SPECULAR, mat_spec);
    glMaterialf(GL_FRONT, GL_SHININESS, 32.0f);

    GLUquadric* quad = gluNewQuadric();
    gluQuadricNormals(quad, GLU_SMOOTH);

    SimParams params;
    Flock flock(params);

    while (!glfwWindowShouldClose(window)) {
        if (g_stop_requested || glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
            break;
        }

        flock.step();

        int width, height;
        glfwGetFramebufferSize(window, &width, &height);
        glViewport(0, 0, width, height);

        // Clean pure white background
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

        for (const auto& b : flock.boids()) {
            glPushMatrix();
            glTranslated(b.pos.x, b.pos.y, b.pos.z);
            glColor3f(b.color.r, b.color.g, b.color.b);
            gluSphere(quad, params.sphere_radius, 16, 12);
            glPopMatrix();

            if (b.trail.size() > 1) {
                glDisable(GL_LIGHTING);
                glLineWidth(2.2f);
                glBegin(GL_LINE_STRIP);
                for (size_t k = 0; k < b.trail.size(); ++k) {
                    float alpha = static_cast<float>(k + 1) / static_cast<float>(b.trail.size());
                    glColor4f(b.color.r, b.color.g, b.color.b, alpha * 0.9f);
                    glVertex3d(b.trail[k].x, b.trail[k].y, b.trail[k].z);
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
