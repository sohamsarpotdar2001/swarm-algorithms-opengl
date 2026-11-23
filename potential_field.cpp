// apf_3d_gl.cpp
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

    Vec3 normalized() const {
        double len = length();
        return (len > 1e-8) ? (*this / len) : Vec3{0.0, 0.0, 0.0};
    }

    Vec3 limited(double max_val) const {
        double len = length();
        return (len > max_val && len > 1e-8) ? (*this * (max_val / len)) : *this;
    }
};

struct Color3 {
    float r, g, b;
};

struct Obstacle {
    Vec3 center;
    double radius;
};

struct Agent {
    Vec3 pos;
    Vec3 vel;
    Vec3 goal;
    Color3 color;
    std::deque<Vec3> trail;
};

struct APFParams {
    size_t num_agents = 15;
    size_t trail_length = 4;
    double box_size = 10.0;
    double agent_radius = 0.26;

    // Artificial Potential Field Gains & Thresholds
    double k_att = 0.045;           // Attractive potential gain
    double d_goal_star = 2.5;       // Quadratic-to-conic transition distance
    double k_rep = 0.35;            // Obstacle repulsive gain
    double influence_dist = 2.2;    // Obstacle detection clearance (Q*)
    double k_agent_rep = 0.12;      // Inter-agent collision avoidance gain
    double agent_clearance = 1.1;   // Minimum separation between agents
    double k_damp = 0.22;           // Velocity damping for smooth arrival

    double max_speed = 0.24;
    double max_force = 0.045;
};

class APFSimulation {
public:
    explicit APFSimulation(const APFParams& params)
        : params_(params), rng_(std::random_device{}()) {
        // Place spherical obstacles in the middle of the 3D workspace
        obstacles_ = {
            {{ 0.0,  0.0,  0.0}, 1.8},
            {{-3.2,  1.8, -1.5}, 1.4},
            {{ 3.0, -1.6,  1.8}, 1.5},
            {{-1.5, -2.5,  2.5}, 1.2},
            {{ 2.2,  2.4, -2.6}, 1.3}
        };

        std::uniform_real_distribution<double> spread(-5.5, 5.5);
        agents_.resize(params_.num_agents);

        for (size_t i = 0; i < params_.num_agents; ++i) {
            // Spawn agents on the -X side and set goals on the +X side (forcing obstacle traversal)
            agents_[i].pos = {-7.5, spread(rng_), spread(rng_)};
            agents_[i].goal = { 7.5, -agents_[i].pos.y * 0.8, -agents_[i].pos.z * 0.8};
            agents_[i].vel = {0.0, 0.0, 0.0};
            agents_[i].trail.push_back(agents_[i].pos);

            float t = static_cast<float>(i) / std::max<size_t>(1, params_.num_agents - 1);
            agents_[i].color = {
                0.10f + 0.78f * t,
                0.22f + 0.45f * (1.0f - std::abs(2.0f * t - 1.0f)),
                0.85f - 0.68f * t
            };
        }
    }

    void reset_goals_if_reached() {
        for (auto& a : agents_) {
            if ((a.goal - a.pos).length() < 0.45) {
                // Swap start and goal sides so the simulation runs continuously
                a.goal.x = -a.goal.x;
                a.goal.y = -a.goal.y;
                a.goal.z = -a.goal.z;
            }
        }
    }

    void step() {
        reset_goals_if_reached();

        const size_t n = agents_.size();
        std::vector<Vec3> next_vel(n);

        for (size_t i = 0; i < n; ++i) {
            const Vec3& p = agents_[i].pos;

            // 1. Hybrid Quadratic / Conic Attractive Force toward Goal
            Vec3 to_goal = agents_[i].goal - p;
            double dist_goal = to_goal.length();
            Vec3 f_att{0.0, 0.0, 0.0};

            if (dist_goal > 1e-8) {
                if (dist_goal <= params_.d_goal_star) {
                    // Quadratic potential bowl near goal: F_att = -k_att * (p - p_goal)
                    f_att = to_goal * params_.k_att;
                } else {
                    // Conic potential far from goal: constant pull magnitude
                    f_att = to_goal.normalized() * (params_.k_att * params_.d_goal_star);
                }
            }

            // 2. Repulsive Force from Spherical Obstacles
            Vec3 f_rep{0.0, 0.0, 0.0};
            for (const auto& obs : obstacles_) {
                Vec3 from_obs = p - obs.center;
                double center_dist = from_obs.length();
                // Surface-to-surface clearance
                double clearance = center_dist - (obs.radius + params_.agent_radius);

                if (clearance < params_.influence_dist) {
                    double safe_d = std::max(0.08, clearance);
                    Vec3 grad_d = (center_dist > 1e-8) ? (from_obs / center_dist) : Vec3{0.0, 1.0, 0.0};

                    // Standard Khatib APF repulsive gradient
                    double rep_mag = params_.k_rep * (1.0 / safe_d - 1.0 / params_.influence_dist)
                                     * (1.0 / (safe_d * safe_d));
                    f_rep += grad_d * rep_mag;

                    // Tangential curl term to break local minima symmetry directly behind obstacles
                    Vec3 tangent{-grad_d.z, 0.15, grad_d.x};
                    f_rep += tangent.normalized() * (rep_mag * 0.25);
                }
            }

            // 3. Inter-Agent Repulsive Force (prevents agent-agent collisions)
            Vec3 f_inter{0.0, 0.0, 0.0};
            for (size_t j = 0; j < n; ++j) {
                if (i == j) continue;
                Vec3 diff = p - agents_[j].pos;
                double d = diff.length();
                if (d < params_.agent_clearance && d > 1e-8) {
                    double mag = params_.k_agent_rep * (1.0 / d - 1.0 / params_.agent_clearance) / (d * d);
                    f_inter += (diff / d) * mag;
                }
            }

            // 4. Damping + Total Force Integration
            Vec3 f_damp = agents_[i].vel * (-params_.k_damp);
            Vec3 total_force = (f_att + f_rep + f_inter + f_damp).limited(params_.max_force);

            next_vel[i] = (agents_[i].vel + total_force).limited(params_.max_speed);
        }

        // Synchronous state update & 4-point trail update
        for (size_t i = 0; i < n; ++i) {
            agents_[i].vel = next_vel[i];
            agents_[i].pos += agents_[i].vel;

            agents_[i].trail.push_back(agents_[i].pos);
            if (agents_[i].trail.size() > params_.trail_length) {
                agents_[i].trail.pop_front();
            }
        }
    }

    const std::vector<Agent>& agents() const { return agents_; }
    const std::vector<Obstacle>& obstacles() const { return obstacles_; }
    const APFParams& params() const { return params_; }

private:
    APFParams params_;
    std::vector<Agent> agents_;
    std::vector<Obstacle> obstacles_;
    std::mt19937 rng_;
};

// Interactive Orbit Camera
struct Camera {
    float yaw = 20.0f;
    float pitch = 24.0f;
    float distance = 29.0f;
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

    GLFWwindow* window = glfwCreateWindow(1024, 768, "3D Artificial Potential Fields (APF) Path Planning", nullptr, nullptr);
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

    // Bright White Studio Lighting Setup
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

    APFParams params;
    APFSimulation sim(params);

    while (!glfwWindowShouldClose(window)) {
        if (g_stop_requested || glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
            break;
        }

        sim.step();

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

        // 1. Render Spherical Obstacles (Dark Slate Gray)
        for (const auto& obs : sim.obstacles()) {
            glPushMatrix();
            glTranslated(obs.center.x, obs.center.y, obs.center.z);
            glColor4f(0.28f, 0.32f, 0.38f, 0.92f);
            gluSphere(quad, obs.radius, 24, 18);
            glPopMatrix();
        }

        // 2. Render Agents, Goal Markers & 4-Point Trails
        for (const auto& a : sim.agents()) {
            // Draw small translucent goal marker sphere
            glPushMatrix();
            glTranslated(a.goal.x, a.goal.y, a.goal.z);
            glColor4f(a.color.r, a.color.g, a.color.b, 0.30f);
            gluSphere(quad, params.agent_radius * 0.55, 10, 8);
            glPopMatrix();

            // Draw Agent Sphere
            glPushMatrix();
            glTranslated(a.pos.x, a.pos.y, a.pos.z);
            glColor3f(a.color.r, a.color.g, a.color.b);
            gluSphere(quad, params.agent_radius, 16, 12);
            glPopMatrix();

            // Draw 4-Point Sliding Trail
            if (a.trail.size() > 1) {
                glDisable(GL_LIGHTING);
                glLineWidth(2.2f);
                glBegin(GL_LINE_STRIP);
                for (size_t k = 0; k < a.trail.size(); ++k) {
                    float alpha = static_cast<float>(k + 1) / static_cast<float>(a.trail.size());
                    glColor4f(a.color.r, a.color.g, a.color.b, alpha * 0.9f);
                    glVertex3d(a.trail[k].x, a.trail[k].y, a.trail[k].z);
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
