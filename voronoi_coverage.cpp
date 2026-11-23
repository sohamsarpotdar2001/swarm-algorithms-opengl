// voronoi_coverage_3d_gl.cpp
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
    Vec3 operator/(double s) const { return {x / s, y / s, z / s}; }

    Vec3& operator+=(const Vec3& o) { x += o.x; y += o.y; z += o.z; return *this; }

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

struct Agent {
    Vec3 pos;
    Vec3 centroid;          // Target Voronoi cell centroid c_{V_i}
    double cell_mass = 0.0; // Integrated density weight over V_i
    Color3 color;
    std::deque<Vec3> trail;
};

struct VoronoiParams {
    size_t num_agents = 24;
    size_t trail_length = 4;      // Sliding window of last 4 trajectory points
    int grid_res = 26;            // 26^3 = 17,576 voxel quadrature points for fast 3D Voronoi
    double box_size = 8.0;        // Bounded workspace [-8, 8]^3
    double sphere_radius = 0.28;

    double k_gain = 0.14;         // Lloyd proportional centroid-tracking gain
    double max_speed = 0.25;
};

class VoronoiCoverage3D {
public:
    explicit VoronoiCoverage3D(const VoronoiParams& params)
        : params_(params), rng_(std::random_device{}()) {
        precompute_grid();
        reset();
    }

    void reset() {
        // Cluster initial agents in one octant to clearly showcase expansion into full 3D coverage
        std::uniform_real_distribution<double> cluster_dist(-params_.box_size * 0.85, -params_.box_size * 0.15);

        agents_.assign(params_.num_agents, Agent{});
        for (size_t i = 0; i < params_.num_agents; ++i) {
            agents_[i].pos = {cluster_dist(rng_), cluster_dist(rng_), cluster_dist(rng_)};
            agents_[i].centroid = agents_[i].pos;
            agents_[i].trail.push_back(agents_[i].pos);

            float t = static_cast<float>(i) / std::max<size_t>(1, params_.num_agents - 1);
            agents_[i].color = {
                0.10f + 0.78f * t,
                0.22f + 0.45f * (1.0f - std::abs(2.0f * t - 1.0f)),
                0.85f - 0.68f * t
            };
        }
        iteration_ = 0;
        compute_voronoi_centroids(0.0);
    }

    void toggle_hotspot() {
        use_hotspot_ = !use_hotspot_;
    }

    bool hotspot_enabled() const { return use_hotspot_; }
    const Vec3& hotspot_pos() const { return hotspot_; }

    void step(double sim_time) {
        // Move sensory density hotspot smoothly when enabled
        if (use_hotspot_) {
            hotspot_ = {
                4.2 * std::sin(0.55 * sim_time),
                2.8 * std::cos(0.40 * sim_time),
                4.2 * std::cos(0.55 * sim_time)
            };
        }

        compute_voronoi_centroids(sim_time);

        // Lloyd's continuous control law: p_dot = k * (c_{V_i} - p_i)
        for (auto& a : agents_) {
            Vec3 vel = ((a.centroid - a.pos) * params_.k_gain).limited(params_.max_speed);
            a.pos += vel;

            a.pos.x = std::clamp(a.pos.x, -params_.box_size, params_.box_size);
            a.pos.y = std::clamp(a.pos.y, -params_.box_size, params_.box_size);
            a.pos.z = std::clamp(a.pos.z, -params_.box_size, params_.box_size);

            a.trail.push_back(a.pos);
            if (a.trail.size() > params_.trail_length) {
                a.trail.pop_front();
            }
        }

        // Rebuild Delaunay-dual proximity edges for visual inspection
        rebuild_neighbor_links();
        ++iteration_;
    }

    // Coverage cost H(P) = sum_i int_{V_i} ||q - p_i||^2 phi(q) dq (monotonically decreases under Lloyd)
    double coverage_cost() const { return last_cost_; }
    int iteration() const { return iteration_; }
    const std::vector<Agent>& agents() const { return agents_; }
    const std::vector<std::pair<size_t, size_t>>& neighbor_links() const { return links_; }
    const VoronoiParams& params() const { return params_; }

private:
    void precompute_grid() {
        const int r = params_.grid_res;
        const double step = (2.0 * params_.box_size) / static_cast<double>(r);
        voxel_vol_ = step * step * step;

        grid_points_.reserve(static_cast<size_t>(r * r * r));
        for (int ix = 0; ix < r; ++ix) {
            double x = -params_.box_size + (ix + 0.5) * step;
            for (int iy = 0; iy < r; ++iy) {
                double y = -params_.box_size + (iy + 0.5) * step;
                for (int iz = 0; iz < r; ++iz) {
                    double z = -params_.box_size + (iz + 0.5) * step;
                    grid_points_.push_back({x, y, z});
                }
            }
        }
    }

    // Sensory importance density phi(q)
    double density(const Vec3& q) const {
        if (!use_hotspot_) return 1.0;
        double d_sq = (q - hotspot_).length_sq();
        // Sharp Gaussian peak concentrates agents around high-interest region
        return 0.15 + 6.0 * std::exp(-d_sq / (2.0 * 3.2 * 3.2));
    }

    void compute_voronoi_centroids(double) {
        const size_t n = agents_.size();
        std::vector<Vec3> weighted_sum(n, {0.0, 0.0, 0.0});
        std::vector<double> mass_sum(n, 0.0);
        double total_cost = 0.0;

        for (const auto& q : grid_points_) {
            size_t best_i = 0;
            double min_d_sq = std::numeric_limits<double>::infinity();

            for (size_t i = 0; i < n; ++i) {
                double d_sq = (q - agents_[i].pos).length_sq();
                if (d_sq < min_d_sq) {
                    min_d_sq = d_sq;
                    best_i = i;
                }
            }

            double w = density(q) * voxel_vol_;
            weighted_sum[best_i] += q * w;
            mass_sum[best_i] += w;
            total_cost += min_d_sq * w;
        }

        last_cost_ = total_cost;
        for (size_t i = 0; i < n; ++i) {
            agents_[i].cell_mass = mass_sum[i];
            if (mass_sum[i] > 1e-9) {
                agents_[i].centroid = weighted_sum[i] / mass_sum[i];
            } else {
                agents_[i].centroid = agents_[i].pos;
            }
        }
    }

    void rebuild_neighbor_links() {
        links_.clear();
        const size_t n = agents_.size();
        // Connect agents whose cells share close spatial proximity (approximate Delaunay dual)
        double thresh = (2.0 * params_.box_size) / std::cbrt(static_cast<double>(n)) * 1.35;
        double thresh_sq = thresh * thresh;

        for (size_t i = 0; i < n; ++i) {
            for (size_t j = i + 1; j < n; ++j) {
                if ((agents_[i].pos - agents_[j].pos).length_sq() <= thresh_sq) {
                    links_.emplace_back(i, j);
                }
            }
        }
    }

    VoronoiParams params_;
    std::vector<Agent> agents_;
    std::vector<Vec3> grid_points_;
    std::vector<std::pair<size_t, size_t>> links_;
    double voxel_vol_ = 1.0;
    double last_cost_ = 0.0;
    bool use_hotspot_ = false;
    Vec3 hotspot_{0.0, 0.0, 0.0};
    int iteration_ = 0;
    std::mt19937 rng_;
};

// Interactive Orbit Camera
struct Camera {
    float yaw = 38.0f;
    float pitch = 25.0f;
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
    glColor4f(0.25f, 0.28f, 0.35f, 0.38f);
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

    GLFWwindow* window = glfwCreateWindow(1024, 768, "3D Voronoi Coverage Control (Lloyd's Algorithm) - [G] Hotspot | [R] Reset", nullptr, nullptr);
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

    VoronoiParams params;
    VoronoiCoverage3D voronoi(params);

    bool r_was_pressed = false;
    bool g_was_pressed = false;
    double sim_time = 0.0;
    std::cout << std::fixed << std::setprecision(2);

    while (!glfwWindowShouldClose(window)) {
        if (g_stop_requested || glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
            break;
        }

        bool r_pressed = (glfwGetKey(window, GLFW_KEY_R) == GLFW_PRESS);
        if (r_pressed && !r_was_pressed) {
            voronoi.reset();
            std::cout << "--- Agents Re-Clustered in Corner ---\n";
        }
        r_was_pressed = r_pressed;

        bool g_pressed = (glfwGetKey(window, GLFW_KEY_G) == GLFW_PRESS);
        if (g_pressed && !g_was_pressed) {
            voronoi.toggle_hotspot();
            std::cout << "--- Density Mode: "
                      << (voronoi.hotspot_enabled() ? "Dynamic Gaussian Hotspot" : "Uniform Coverage")
                      << " ---\n";
        }
        g_was_pressed = g_pressed;

        sim_time += 0.016;
        voronoi.step(sim_time);

        if (voronoi.iteration() % 40 == 0) {
            std::cout << "Step " << std::setw(4) << voronoi.iteration()
                      << " | Locational Coverage Cost H(P): " << voronoi.coverage_cost() << "\n";
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

        const auto& agents = voronoi.agents();

        // 1. Render Moving Sensory Density Hotspot (when enabled via 'G')
        if (voronoi.hotspot_enabled()) {
            const Vec3& hs = voronoi.hotspot_pos();
            glPushMatrix();
            glTranslated(hs.x, hs.y, hs.z);
            glColor4f(0.10f, 0.72f, 0.32f, 0.30f);
            gluSphere(quad, 1.4, 20, 16);
            glPopMatrix();
        }

        // 2. Render Spatial Coverage Mesh & Vectors from Agents to Voronoi Centroids
        glDisable(GL_LIGHTING);
        glColor4f(0.25f, 0.35f, 0.50f, 0.22f);
        glLineWidth(1.1f);
        glBegin(GL_LINES);
        for (const auto& link : voronoi.neighbor_links()) {
            const auto& p1 = agents[link.first].pos;
            const auto& p2 = agents[link.second].pos;
            glVertex3d(p1.x, p1.y, p1.z);
            glVertex3d(p2.x, p2.y, p2.z);
        }
        glEnd();

        // Draw short tethers from each agent to its target Voronoi cell centroid
        glColor4f(0.85f, 0.45f, 0.10f, 0.45f);
        glLineWidth(1.2f);
        glBegin(GL_LINES);
        for (const auto& a : agents) {
            glVertex3d(a.pos.x, a.pos.y, a.pos.z);
            glVertex3d(a.centroid.x, a.centroid.y, a.centroid.z);
        }
        glEnd();
        glEnable(GL_LIGHTING);

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
