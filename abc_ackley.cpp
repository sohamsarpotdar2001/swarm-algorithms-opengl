// abc_ackley_gl.cpp
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

    double& operator[](int idx) { return (idx == 0) ? x : (idx == 1 ? y : z); }
    double operator[](int idx) const { return (idx == 0) ? x : (idx == 1 ? y : z); }
};

struct Color3 {
    float r, g, b;
};

struct FoodSource {
    Vec3 pos;             // Current logical food source position in search space
    Vec3 visual_pos;      // Smoothly interpolated 3D position for rendering
    double obj_val = 0.0; // Ackley objective value f(x)
    double fitness = 0.0; // Nectar quality: 1 / (1 + f(x))
    int trials = 0;       // Stagnation counter for Scout abandonment
    Color3 color;
    std::deque<Vec3> trail;
};

struct ABCParams {
    size_t colony_size = 50;      // Total bees (SN = colony_size / 2 food sources)
    size_t trail_length = 4;      // Sliding window of last 4 trajectory points
    int abandon_limit = 45;       // Max trials before Scout bee abandons a source
    double domain_limit = 5.0;    // Search bounds [-5, 5]^3
    double visual_scale = 1.8;    // Scale [-5, 5] -> [-9, 9] for display
    double sphere_radius = 0.25;
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

// Converts minimization objective f(x) >= 0 into maximization nectar fitness
double compute_fitness(double obj) {
    return (obj >= 0.0) ? (1.0 / (1.0 + obj)) : (1.0 + std::abs(obj));
}

class ArtificialBeeColony {
public:
    explicit ArtificialBeeColony(const ABCParams& params)
        : params_(params), rng_(std::random_device{}()) {
        reset();
    }

    void reset() {
        const size_t sn = std::max<size_t>(2, params_.colony_size / 2);
        std::uniform_real_distribution<double> pos_dist(-params_.domain_limit, params_.domain_limit);

        best_val_ = std::numeric_limits<double>::infinity();
        best_pos_ = {0.0, 0.0, 0.0};
        iteration_ = 0;

        sources_.assign(sn, FoodSource{});
        for (size_t i = 0; i < sn; ++i) {
            sources_[i].pos = {pos_dist(rng_), pos_dist(rng_), pos_dist(rng_)};
            sources_[i].visual_pos = sources_[i].pos;
            sources_[i].obj_val = ackley_3d(sources_[i].pos);
            sources_[i].fitness = compute_fitness(sources_[i].obj_val);
            sources_[i].trials = 0;
            sources_[i].trail.push_back(sources_[i].visual_pos);

            if (sources_[i].obj_val < best_val_) {
                best_val_ = sources_[i].obj_val;
                best_pos_ = sources_[i].pos;
            }

            float t = static_cast<float>(i) / std::max<size_t>(1, sn - 1);
            sources_[i].color = {
                0.10f + 0.78f * t,
                0.22f + 0.45f * (1.0f - std::abs(2.0f * t - 1.0f)),
                0.85f - 0.68f * t
            };
        }
    }

    // Mutates food source i along one random dimension using partner k != i
    void explore_neighbor(size_t i) {
        const size_t sn = sources_.size();
        std::uniform_int_distribution<size_t> partner_dist(0, sn - 2);
        std::uniform_int_distribution<int> dim_dist(0, 2);
        std::uniform_real_distribution<double> phi_dist(-1.0, 1.0);

        size_t k = partner_dist(rng_);
        if (k >= i) ++k;

        int j = dim_dist(rng_);
        double phi = phi_dist(rng_);

        Vec3 candidate = sources_[i].pos;
        candidate[j] = sources_[i].pos[j] + phi * (sources_[i].pos[j] - sources_[k].pos[j]);
        candidate[j] = std::clamp(candidate[j], -params_.domain_limit, params_.domain_limit);

        double cand_obj = ackley_3d(candidate);
        double cand_fit = compute_fitness(cand_obj);

        // Greedy selection between current food source and mutated candidate
        if (cand_fit > sources_[i].fitness) {
            sources_[i].pos = candidate;
            sources_[i].obj_val = cand_obj;
            sources_[i].fitness = cand_fit;
            sources_[i].trials = 0;

            if (cand_obj < best_val_) {
                best_val_ = cand_obj;
                best_pos_ = candidate;
            }
        } else {
            ++sources_[i].trials;
        }
    }

    void step_iteration() {
        const size_t sn = sources_.size();

        // 1. Employed Bees Phase: every source undergoes local exploration
        for (size_t i = 0; i < sn; ++i) {
            explore_neighbor(i);
        }

        // 2. Onlooker Bees Phase: roulette-wheel selection weighted by nectar fitness
        double total_fitness = 0.0;
        for (const auto& s : sources_) total_fitness += s.fitness;

        std::uniform_real_distribution<double> uni01(0.0, 1.0);
        size_t onlookers_dispatched = 0;
        size_t idx = 0;
        while (onlookers_dispatched < sn) {
            double prob = (total_fitness > 1e-12) ? (sources_[idx].fitness / total_fitness) : (1.0 / sn);
            if (uni01(rng_) < prob) {
                explore_neighbor(idx);
                ++onlookers_dispatched;
            }
            idx = (idx + 1) % sn;
        }

        // 3. Scout Bees Phase: abandon exhausted food sources exceeding abandon_limit
        std::uniform_real_distribution<double> pos_dist(-params_.domain_limit, params_.domain_limit);
        for (size_t i = 0; i < sn; ++i) {
            if (sources_[i].trials >= params_.abandon_limit) {
                sources_[i].pos = {pos_dist(rng_), pos_dist(rng_), pos_dist(rng_)};
                sources_[i].obj_val = ackley_3d(sources_[i].pos);
                sources_[i].fitness = compute_fitness(sources_[i].obj_val);
                sources_[i].trials = 0;
                sources_[i].trail.clear();  // Reset trail on scout teleportation

                if (sources_[i].obj_val < best_val_) {
                    best_val_ = sources_[i].obj_val;
                    best_pos_ = sources_[i].pos;
                }
            }
        }

        ++iteration_;
    }

    void update_visuals() {
        for (auto& s : sources_) {
            // Smooth exponential interpolation toward current food source coordinates
            s.visual_pos = s.visual_pos + (s.pos - s.visual_pos) * 0.35;
            s.trail.push_back(s.visual_pos);
            if (s.trail.size() > params_.trail_length) {
                s.trail.pop_front();
            }
        }
    }

    const std::vector<FoodSource>& sources() const { return sources_; }
    const Vec3& best_pos() const { return best_pos_; }
    double best_val() const { return best_val_; }
    int iteration() const { return iteration_; }
    const ABCParams& params() const { return params_; }

private:
    ABCParams params_;
    std::vector<FoodSource> sources_;
    Vec3 best_pos_{0.0, 0.0, 0.0};
    double best_val_ = std::numeric_limits<double>::infinity();
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

    GLFWwindow* window = glfwCreateWindow(1024, 768, "3D Artificial Bee Colony (ABC) Optimization - Press R to Reset", nullptr, nullptr);
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

    ABCParams params;
    ArtificialBeeColony abc(params);

    bool r_was_pressed = false;
    int frame_counter = 0;
    std::cout << std::fixed << std::setprecision(6);

    while (!glfwWindowShouldClose(window)) {
        if (g_stop_requested || glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
            break;
        }

        bool r_pressed = (glfwGetKey(window, GLFW_KEY_R) == GLFW_PRESS);
        if (r_pressed && !r_was_pressed) {
            abc.reset();
            std::cout << "--- Colony Reset ---\n";
        }
        r_was_pressed = r_pressed;

        // Run one ABC foraging cycle every 2 frames so axial movements are clearly visible
        if (++frame_counter % 2 == 0) {
            abc.step_iteration();
            if (abc.iteration() % 25 == 0) {
                const auto& bp = abc.best_pos();
                std::cout << "Cycle " << std::setw(4) << abc.iteration()
                          << " | Best Nectar f(x): " << abc.best_val()
                          << " at (" << bp.x << ", " << bp.y << ", " << bp.z << ")\n";
            }
        }
        abc.update_visuals();

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

        // 1. True Analytical Global Minimum at (0, 0, 0) in Translucent Green
        glPushMatrix();
        glColor4f(0.10f, 0.72f, 0.32f, 0.45f);
        gluSphere(quad, params.sphere_radius * 1.45, 16, 12);
        glPopMatrix();

        // 2. Best Food Source Found by Colony in Bright Gold
        const Vec3& bp = abc.best_pos();
        glPushMatrix();
        glTranslated(bp.x * s, bp.y * s, bp.z * s);
        glColor3f(0.95f, 0.72f, 0.05f);
        gluSphere(quad, params.sphere_radius * 1.30, 16, 12);
        glPopMatrix();

        // 3. Active Food Sources (Employed / Onlooker / Scout Bees) & 4-Point Trails
        for (const auto& src : abc.sources()) {
            glPushMatrix();
            glTranslated(src.visual_pos.x * s, src.visual_pos.y * s, src.visual_pos.z * s);
            glColor3f(src.color.r, src.color.g, src.color.b);
            gluSphere(quad, params.sphere_radius, 16, 12);
            glPopMatrix();

            if (src.trail.size() > 1) {
                glDisable(GL_LIGHTING);
                glLineWidth(2.2f);
                glBegin(GL_LINE_STRIP);
                for (size_t k = 0; k < src.trail.size(); ++k) {
                    float alpha = static_cast<float>(k + 1) / static_cast<float>(src.trail.size());
                    glColor4f(src.color.r, src.color.g, src.color.b, alpha * 0.9f);
                    glVertex3d(src.trail[k].x * s, src.trail[k].y * s, src.trail[k].z * s);
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
