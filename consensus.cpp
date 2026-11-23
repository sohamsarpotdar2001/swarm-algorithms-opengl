// consensus_3d_gl.cpp
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
#include <numeric>
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

    double length_sq() const { return x * x + y * y + z * z; }
    double length() const { return std::sqrt(length_sq()); }
};

struct Color3 {
    float r, g, b;
};

struct Agent {
    Vec3 pos;
    Vec3 vel;
    Color3 color;
    std::deque<Vec3> trail;
};

struct ConsensusParams {
    size_t num_agents = 28;
    size_t trail_length = 4;      // Sliding window of last 4 trajectory points
    double box_size = 9.0;
    double sphere_radius = 0.26;

    // Consensus Protocol Parameters (Second-Order damped Laplacian)
    double comm_radius = 7.5;     // Communication range R_comm for graph edges
    double k_pos = 0.035;         // Position consensus coupling gain (alpha)
    double k_vel = 0.14;          // Velocity consensus / damping gain (beta)
    double max_speed = 0.25;
};

class ConsensusNetwork {
public:
    explicit ConsensusNetwork(const ConsensusParams& params)
        : params_(params), rng_(std::random_device{}()) {
        reset();
    }

    void reset() {
        std::uniform_real_distribution<double> pos_dist(-params_.box_size * 0.85, params_.box_size * 0.85);
        std::normal_distribution<double> vel_dist(0.0, 0.05);

        agents_.assign(params_.num_agents, Agent{});
        for (size_t i = 0; i < params_.num_agents; ++i) {
            agents_[i].pos = {pos_dist(rng_), pos_dist(rng_), pos_dist(rng_)};
            agents_[i].vel = {vel_dist(rng_), vel_dist(rng_), vel_dist(rng_)};
            agents_[i].trail.push_back(agents_[i].pos);

            float t = static_cast<float>(i) / std::max<size_t>(1, params_.num_agents - 1);
            agents_[i].color = {
                0.10f + 0.78f * t,
                0.22f + 0.45f * (1.0f - std::abs(2.0f * t - 1.0f)),
                0.85f - 0.68f * t
            };
        }

        compute_centroid();
        rebuild_graph();
        iteration_ = 0;
    }

    // Perturbs agents outward while keeping the network connected to test disturbance rejection
    void perturb() {
        std::uniform_real_distribution<double> kick(-0.35, 0.35);
        for (auto& a : agents_) {
            a.vel += Vec3{kick(rng_), kick(rng_), kick(rng_)};
        }
    }

    void step() {
        const size_t n = agents_.size();
        rebuild_graph();

        std::vector<Vec3> next_vel(n);
        for (size_t i = 0; i < n; ++i) {
            Vec3 pos_consensus{0.0, 0.0, 0.0};
            Vec3 vel_consensus{0.0, 0.0, 0.0};

            // Sum over neighbors j in N_i: u_i = sum_{j in N_i} [ k_pos*(x_j - x_i) + k_vel*(v_j - v_i) ]
            for (const auto& edge : edges_) {
                if (edge.first == i) {
                    size_t j = edge.second;
                    pos_consensus += (agents_[j].pos - agents_[i].pos);
                    vel_consensus += (agents_[j].vel - agents_[i].vel);
                } else if (edge.second == i) {
                    size_t j = edge.first;
                    pos_consensus += (agents_[j].pos - agents_[i].pos);
                    vel_consensus += (agents_[j].vel - agents_[i].vel);
                }
            }

            // Normalize by (N - 1) so step size stays stable regardless of swarm size
            double scale = (n > 1) ? (1.0 / static_cast<double>(n - 1)) : 1.0;
            Vec3 accel = (pos_consensus * params_.k_pos + vel_consensus * params_.k_vel) * scale
                         - agents_[i].vel * 0.08;  // Light local damping for asymptotic rest

            Vec3 v = agents_[i].vel + accel;
            double speed = v.length();
            if (speed > params_.max_speed && speed > 1e-8) {
                v = v * (params_.max_speed / speed);
            }
            next_vel[i] = v;
        }

        for (size_t i = 0; i < n; ++i) {
            agents_[i].vel = next_vel[i];
            agents_[i].pos += agents_[i].vel;

            agents_[i].trail.push_back(agents_[i].pos);
            if (agents_[i].trail.size() > params_.trail_length) {
                agents_[i].trail.pop_front();
            }
        }

        compute_centroid();
        ++iteration_;
    }

    // Computes maximum distance of any agent from the group consensus centroid
    double disagreement_error() const {
        double max_err = 0.0;
        for (const auto& a : agents_) {
            max_err = std::max(max_err, (a.pos - centroid_).length());
        }
        return max_err;
    }

    const std::vector<Agent>& agents() const { return agents_; }
    const std::vector<std::pair<size_t, size_t>>& edges() const { return edges_; }
    const Vec3& centroid() const { return centroid_; }
    int iteration() const { return iteration_; }
    const ConsensusParams& params() const { return params_; }

private:
    void compute_centroid() {
        Vec3 sum{0.0, 0.0, 0.0};
        for (const auto& a : agents_) sum += a.pos;
        centroid_ = agents_.empty() ? Vec3{0, 0, 0} : (sum / static_cast<double>(agents_.size()));
    }

    void rebuild_graph() {
        edges_.clear();
        const size_t n = agents_.size();
        const double r_sq = params_.comm_radius * params_.comm_radius;

        for (size_t i = 0; i < n; ++i) {
            for (size_t j = i + 1; j < n; ++j) {
                if ((agents_[i].pos - agents_[j].pos).length_sq() <= r_sq) {
                    edges_.emplace_back(i, j);
                }
            }
        }
    }

    ConsensusParams params_;
    std::vector<Agent> agents_;
    std::vector<std::pair<size_t, size_t>> edges_;
    Vec3 centroid_{0.0, 0.0, 0.0};
    int iteration_ = 0;
    std::mt19937 rng_;
};

// Interactive Orbit Camera
struct Camera {
    float yaw = 38.0f;
    float pitch = 25.0f;
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
    glColor4f(0.25f, 0.28f, 0.35f, 0.30f);
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

    GLFWwindow* window = glfwCreateWindow(1024, 768, "3D Multi-Agent Consensus - [Space] Perturb | [R] Reset", nullptr, nullptr);
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

    ConsensusParams params;
    ConsensusNetwork network(params);

    bool r_was_pressed = false;
    bool space_was_pressed = false;
    std::cout << std::fixed << std::setprecision(4);

    while (!glfwWindowShouldClose(window)) {
        if (g_stop_requested || glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
            break;
        }

        // 'R' resets agents to new random positions
        bool r_pressed = (glfwGetKey(window, GLFW_KEY_R) == GLFW_PRESS);
        if (r_pressed && !r_was_pressed) {
            network.reset();
            std::cout << "--- Network Reset ---\n";
        }
        r_was_pressed = r_pressed;

        // 'Space' injects velocity disturbances to test re-convergence
        bool space_pressed = (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS);
        if (space_pressed && !space_was_pressed) {
            network.perturb();
            std::cout << "--- Disturbance Injected ---\n";
        }
        space_was_pressed = space_pressed;

        network.step();

        if (network.iteration() % 40 == 0) {
            const auto& c = network.centroid();
            std::cout << "Step " << std::setw(4) << network.iteration()
                      << " | Active Links: " << std::setw(3) << network.edges().size()
                      << " | Max Disagreement: " << network.disagreement_error()
                      << " | Centroid: (" << c.x << ", " << c.y << ", " << c.z << ")\n";
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

        draw_wireframe_box(params.box_size);

        const auto& agents = network.agents();

        // 1. Render Active Communication Graph Edges (Laplacian Topology)
        glDisable(GL_LIGHTING);
        glColor4f(0.25f, 0.35f, 0.50f, 0.22f);
        glLineWidth(1.1f);
        glBegin(GL_LINES);
        for (const auto& edge : network.edges()) {
            const auto& p1 = agents[edge.first].pos;
            const auto& p2 = agents[edge.second].pos;
            glVertex3d(p1.x, p1.y, p1.z);
            glVertex3d(p2.x, p2.y, p2.z);
        }
        glEnd();
        glEnable(GL_LIGHTING);

        // 2. Render Consensus Equilibrium Centroid (Translucent Green Sphere)
        const Vec3& c = network.centroid();
        glPushMatrix();
        glTranslated(c.x, c.y, c.z);
        glColor4f(0.10f, 0.72f, 0.32f, 0.45f);
        gluSphere(quad, params.sphere_radius * 1.6, 18, 14);
        glPopMatrix();

        // 3. Render Agents & 4-Point Sliding Trails
        for (const auto& a : agents) {
            glPushMatrix();
            glTranslated(a.pos.x, a.pos.y, a.pos.z);
            glColor3f(a.color.r, a.color.g, a.color.b);
            gluSphere(quad, params.sphere_radius, 16, 12);
            glPopMatrix();

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
