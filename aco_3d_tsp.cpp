// aco_3d_tsp_gl.cpp
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

    double length() const { return std::sqrt(x * x + y * y + z * z); }
};

struct Color3 {
    float r, g, b;
};

struct Ant {
    std::vector<int> tour;
    double tour_length = 0.0;
    Vec3 visual_pos;
    Color3 color;
    std::deque<Vec3> trail;
};

struct ACOParams {
    size_t num_cities = 18;
    size_t num_ants = 24;
    size_t trail_length = 4;     // Last 4 positions per ant
    double box_size = 8.5;
    double city_radius = 0.28;
    double ant_radius = 0.16;

    // Ant System / Elitist ACO Hyperparameters
    double alpha = 1.0;          // Pheromone importance
    double beta = 3.0;           // Distance heuristic importance (1 / d)
    double evaporation = 0.25;   // Pheromone evaporation rate (rho)
    double q_deposit = 50.0;     // Pheromone deposit constant
    double elitist_weight = 2.5; // Extra pheromone boost for global best tour
};

class AntColonyTSP {
public:
    explicit AntColonyTSP(const ACOParams& params)
        : params_(params), rng_(std::random_device{}()) {
        reset();
    }

    void reset() {
        const size_t n = params_.num_cities;
        std::uniform_real_distribution<double> coord(-params_.box_size * 0.85, params_.box_size * 0.85);

        cities_.resize(n);
        for (size_t i = 0; i < n; ++i) {
            cities_[i] = {coord(rng_), coord(rng_), coord(rng_)};
        }

        dist_.assign(n, std::vector<double>(n, 0.0));
        pheromone_.assign(n, std::vector<double>(n, 1.0));

        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < n; ++j) {
                if (i != j) {
                    dist_[i][j] = std::max(1e-4, (cities_[i] - cities_[j]).length());
                }
            }
        }

        best_tour_.clear();
        best_length_ = std::numeric_limits<double>::infinity();
        iteration_ = 0;

        ants_.assign(params_.num_ants, Ant{});
        for (size_t k = 0; k < params_.num_ants; ++k) {
            float t = static_cast<float>(k) / std::max<size_t>(1, params_.num_ants - 1);
            ants_[k].color = {
                0.10f + 0.78f * t,
                0.22f + 0.45f * (1.0f - std::abs(2.0f * t - 1.0f)),
                0.85f - 0.68f * t
            };
        }

        construct_tours();
        update_pheromones();
    }

    // Runs one full ACO tour construction + pheromone update cycle
    void step_iteration() {
        construct_tours();
        update_pheromones();
        ++iteration_;
    }

    // Interpolates ant spheres smoothly along their current tour for 3D visualization
    void update_ant_visuals(double phase_0_to_1) {
        const size_t n = cities_.size();
        if (n < 2) return;

        double scaled = phase_0_to_1 * static_cast<double>(n);
        size_t seg = std::min(n - 1, static_cast<size_t>(scaled));
        double local_t = scaled - static_cast<double>(seg);

        for (auto& ant : ants_) {
            if (ant.tour.size() < n) continue;
            int u = ant.tour[seg];
            int v = ant.tour[(seg + 1) % n];

            ant.visual_pos = cities_[u] * (1.0 - local_t) + cities_[v] * local_t;
            ant.trail.push_back(ant.visual_pos);
            if (ant.trail.size() > params_.trail_length) {
                ant.trail.pop_front();
            }
        }
    }

    const std::vector<Vec3>& cities() const { return cities_; }
    const std::vector<Ant>& ants() const { return ants_; }
    const std::vector<std::vector<double>>& pheromone() const { return pheromone_; }
    const std::vector<int>& best_tour() const { return best_tour_; }
    double best_length() const { return best_length_; }
    int iteration() const { return iteration_; }
    const ACOParams& params() const { return params_; }

private:
    int select_next_city(int current, const std::vector<bool>& visited) {
        const size_t n = cities_.size();
        std::vector<double> weights(n, 0.0);
        double total_weight = 0.0;

        for (size_t j = 0; j < n; ++j) {
            if (!visited[j]) {
                double tau = std::pow(pheromone_[current][j], params_.alpha);
                double eta = std::pow(1.0 / dist_[current][j], params_.beta);
                weights[j] = tau * eta;
                total_weight += weights[j];
            }
        }

        if (total_weight <= 1e-12) {
            for (size_t j = 0; j < n; ++j) {
                if (!visited[j]) return static_cast<int>(j);
            }
        }

        std::uniform_real_distribution<double> pick(0.0, total_weight);
        double r = pick(rng_);
        double cumulative = 0.0;
        for (size_t j = 0; j < n; ++j) {
            if (!visited[j]) {
                cumulative += weights[j];
                if (r <= cumulative) return static_cast<int>(j);
            }
        }
        for (size_t j = n; j-- > 0;) {
            if (!visited[j]) return static_cast<int>(j);
        }
        return 0;
    }

    void construct_tours() {
        const size_t n = cities_.size();
        std::uniform_int_distribution<int> start_dist(0, static_cast<int>(n - 1));

        for (auto& ant : ants_) {
            ant.tour.clear();
            ant.tour.reserve(n);
            std::vector<bool> visited(n, false);

            int start = start_dist(rng_);
            ant.tour.push_back(start);
            visited[start] = true;

            double length = 0.0;
            int current = start;
            for (size_t step = 1; step < n; ++step) {
                int next = select_next_city(current, visited);
                ant.tour.push_back(next);
                visited[next] = true;
                length += dist_[current][next];
                current = next;
            }
            // Close the TSP loop back to start
            length += dist_[current][start];
            ant.tour_length = length;

            if (length < best_length_) {
                best_length_ = length;
                best_tour_ = ant.tour;
            }
        }
    }

    void update_pheromones() {
        const size_t n = cities_.size();

        // 1. Evaporate existing pheromones
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < n; ++j) {
                pheromone_[i][j] = std::max(0.05, pheromone_[i][j] * (1.0 - params_.evaporation));
            }
        }

        // 2. Deposit pheromone proportional to 1 / tour_length
        for (const auto& ant : ants_) {
            double delta = params_.q_deposit / ant.tour_length;
            for (size_t s = 0; s < n; ++s) {
                int u = ant.tour[s];
                int v = ant.tour[(s + 1) % n];
                pheromone_[u][v] += delta;
                pheromone_[v][u] += delta;
            }
        }

        // 3. Elitist reinforcement along the global best tour
        if (!best_tour_.empty()) {
            double elite_delta = params_.elitist_weight * (params_.q_deposit / best_length_);
            for (size_t s = 0; s < n; ++s) {
                int u = best_tour_[s];
                int v = best_tour_[(s + 1) % n];
                pheromone_[u][v] += elite_delta;
                pheromone_[v][u] += elite_delta;
            }
        }
    }

    ACOParams params_;
    std::vector<Vec3> cities_;
    std::vector<std::vector<double>> dist_;
    std::vector<std::vector<double>> pheromone_;
    std::vector<Ant> ants_;
    std::vector<int> best_tour_;
    double best_length_ = std::numeric_limits<double>::infinity();
    int iteration_ = 0;
    std::mt19937 rng_;
};

// Interactive Orbit Camera
struct Camera {
    float yaw = 38.0f;
    float pitch = 24.0f;
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

    GLFWwindow* window = glfwCreateWindow(1024, 768, "3D Ant Colony Optimization (TSP) - Press R for New Cities", nullptr, nullptr);
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

    ACOParams params;
    AntColonyTSP aco(params);

    bool r_was_pressed = false;
    double tour_phase = 0.0;
    std::cout << std::fixed << std::setprecision(3);

    while (!glfwWindowShouldClose(window)) {
        if (g_stop_requested || glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
            break;
        }

        // Press 'R' to generate a fresh random 3D city layout
        bool r_pressed = (glfwGetKey(window, GLFW_KEY_R) == GLFW_PRESS);
        if (r_pressed && !r_was_pressed) {
            aco.reset();
            tour_phase = 0.0;
            std::cout << "--- New 3D City Graph Generated ---\n";
        }
        r_was_pressed = r_pressed;

        // Advance ant visual traversal; trigger a new ACO iteration each full tour
        tour_phase += 0.015;
        if (tour_phase >= 1.0) {
            tour_phase -= 1.0;
            aco.step_iteration();
            std::cout << "Iter " << std::setw(3) << aco.iteration()
                      << " | Best 3D Tour Length: " << aco.best_length() << "\n";
        }
        aco.update_ant_visuals(tour_phase);

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

        const auto& cities = aco.cities();
        const auto& pher = aco.pheromone();
        const size_t n = cities.size();

        // 1. Render Pheromone Trails Between Cities (Opacity proportional to pheromone strength)
        double max_pher = 0.0;
        for (size_t i = 0; i < n; ++i)
            for (size_t j = i + 1; j < n; ++j)
                max_pher = std::max(max_pher, pher[i][j]);

        glDisable(GL_LIGHTING);
        glLineWidth(1.2f);
        glBegin(GL_LINES);
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = i + 1; j < n; ++j) {
                float norm_p = static_cast<float>(pher[i][j] / std::max(1e-6, max_pher));
                if (norm_p > 0.12f) {
                    glColor4f(0.25f, 0.35f, 0.55f, norm_p * 0.45f);
                    glVertex3d(cities[i].x, cities[i].y, cities[i].z);
                    glVertex3d(cities[j].x, cities[j].y, cities[j].z);
                }
            }
        }
        glEnd();

        // 2. Render Global Best Tour in Bright Gold
        const auto& best = aco.best_tour();
        if (best.size() == n) {
            glColor4f(0.95f, 0.65f, 0.05f, 0.95f);
            glLineWidth(3.2f);
            glBegin(GL_LINE_LOOP);
            for (int idx : best) {
                glVertex3d(cities[idx].x, cities[idx].y, cities[idx].z);
            }
            glEnd();
        }
        glEnable(GL_LIGHTING);

        // 3. Render 3D City Nodes (Dark Slate Spheres)
        for (const auto& c : cities) {
            glPushMatrix();
            glTranslated(c.x, c.y, c.z);
            glColor3f(0.22f, 0.26f, 0.32f);
            gluSphere(quad, params.city_radius, 18, 14);
            glPopMatrix();
        }

        // 4. Render Traveling Ants & 4-Point Sliding Trails
        for (const auto& ant : aco.ants()) {
            glPushMatrix();
            glTranslated(ant.visual_pos.x, ant.visual_pos.y, ant.visual_pos.z);
            glColor3f(ant.color.r, ant.color.g, ant.color.b);
            gluSphere(quad, params.ant_radius, 12, 10);
            glPopMatrix();

            if (ant.trail.size() > 1) {
                glDisable(GL_LIGHTING);
                glLineWidth(2.0f);
                glBegin(GL_LINE_STRIP);
                for (size_t k = 0; k < ant.trail.size(); ++k) {
                    float alpha = static_cast<float>(k + 1) / static_cast<float>(ant.trail.size());
                    glColor4f(ant.color.r, ant.color.g, ant.color.b, alpha * 0.85f);
                    glVertex3d(ant.trail[k].x, ant.trail[k].y, ant.trail[k].z);
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
