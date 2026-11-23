// gwo_ackley_gl.cpp
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
    Vec3 operator/(double s) const { return {x / s, y / s, z / s}; }

    double& operator[](int idx) { return (idx == 0) ? x : (idx == 1 ? y : z); }
    double operator[](int idx) const { return (idx == 0) ? x : (idx == 1 ? y : z); }
};

struct Color3 {
    float r, g, b;
};

struct Wolf {
    Vec3 pos;             // Current logical position in search space
    Vec3 visual_pos;      // Smoothly interpolated 3D position for rendering
    double fitness = std::numeric_limits<double>::infinity();
    Color3 color;
    std::deque<Vec3> trail;
};

struct GWOParams {
    size_t pack_size = 35;        // Number of wolves in the pack
    size_t trail_length = 4;      // Sliding window of last 4 trajectory points
    int max_iterations = 250;     // Hunt duration for linear decay a: 2 -> 0
    double domain_limit = 5.0;    // Search bounds [-5, 5]^3
    double visual_scale = 1.8;    // Scale [-5, 5] -> [-9, 9] for display
    double sphere_radius = 0.24;
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

class GreyWolfOptimizer {
public:
    explicit GreyWolfOptimizer(const GWOParams& params)
        : params_(params), rng_(std::random_device{}()) {
        reset();
    }

    void reset() {
        std::uniform_real_distribution<double> pos_dist(-params_.domain_limit, params_.domain_limit);

        alpha_score_ = std::numeric_limits<double>::infinity();
        beta_score_  = std::numeric_limits<double>::infinity();
        delta_score_ = std::numeric_limits<double>::infinity();

        alpha_pos_ = {0.0, 0.0, 0.0};
        beta_pos_  = {0.0, 0.0, 0.0};
        delta_pos_ = {0.0, 0.0, 0.0};
        iteration_ = 0;

        wolves_.assign(params_.pack_size, Wolf{});
        for (size_t i = 0; i < params_.pack_size; ++i) {
            wolves_[i].pos = {pos_dist(rng_), pos_dist(rng_), pos_dist(rng_)};
            wolves_[i].visual_pos = wolves_[i].pos;
            wolves_[i].trail.push_back(wolves_[i].visual_pos);

            float t = static_cast<float>(i) / std::max<size_t>(1, params_.pack_size - 1);
            wolves_[i].color = {
                0.15f + 0.55f * t,
                0.28f + 0.42f * (1.0f - std::abs(2.0f * t - 1.0f)),
                0.85f - 0.55f * t
            };
        }

        evaluate_and_rank();
    }

    void evaluate_and_rank() {
        for (auto& w : wolves_) {
            w.fitness = ackley_3d(w.pos);

            // Update Alpha, Beta, and Delta hierarchy
            if (w.fitness < alpha_score_) {
                delta_score_ = beta_score_;
                delta_pos_   = beta_pos_;

                beta_score_  = alpha_score_;
                beta_pos_    = alpha_pos_;

                alpha_score_ = w.fitness;
                alpha_pos_   = w.pos;
            } else if (w.fitness < beta_score_) {
                delta_score_ = beta_score_;
                delta_pos_   = beta_pos_;

                beta_score_  = w.fitness;
                beta_pos_    = w.pos;
            } else if (w.fitness < delta_score_) {
                delta_score_ = w.fitness;
                delta_pos_   = w.pos;
            }
        }
    }

    void step_iteration() {
        // Linearly decrease control parameter 'a' from 2.0 down to 0.0
        double progress = std::min(1.0, static_cast<double>(iteration_) / static_cast<double>(params_.max_iterations));
        double a = 2.0 * (1.0 - progress);

        std::uniform_real_distribution<double> uni01(0.0, 1.0);

        for (auto& w : wolves_) {
            Vec3 new_pos{0.0, 0.0, 0.0};

            for (int dim = 0; dim < 3; ++dim) {
                // Guidance from Alpha wolf
                double r1 = uni01(rng_), r2 = uni01(rng_);
                double A1 = 2.0 * a * r1 - a;
                double C1 = 2.0 * r2;
                double D_alpha = std::abs(C1 * alpha_pos_[dim] - w.pos[dim]);
                double X1 = alpha_pos_[dim] - A1 * D_alpha;

                // Guidance from Beta wolf
                r1 = uni01(rng_); r2 = uni01(rng_);
                double A2 = 2.0 * a * r1 - a;
                double C2 = 2.0 * r2;
                double D_beta = std::abs(C2 * beta_pos_[dim] - w.pos[dim]);
                double X2 = beta_pos_[dim] - A2 * D_beta;

                // Guidance from Delta wolf
                r1 = uni01(rng_); r2 = uni01(rng_);
                double A3 = 2.0 * a * r1 - a;
                double C3 = 2.0 * r2;
                double D_delta = std::abs(C3 * delta_pos_[dim] - w.pos[dim]);
                double X3 = delta_pos_[dim] - A3 * D_delta;

                new_pos[dim] = std::clamp((X1 + X2 + X3) / 3.0, -params_.domain_limit, params_.domain_limit);
            }

            w.pos = new_pos;
        }

        evaluate_and_rank();
        ++iteration_;
    }

    void update_visuals() {
        for (auto& w : wolves_) {
            w.visual_pos = w.visual_pos + (w.pos - w.visual_pos) * 0.35;
            w.trail.push_back(w.visual_pos);
            if (w.trail.size() > params_.trail_length) {
                w.trail.pop_front();
            }
        }
    }

    const std::vector<Wolf>& wolves() const { return wolves_; }
    const Vec3& alpha_pos() const { return alpha_pos_; }
    const Vec3& beta_pos() const { return beta_pos_; }
    const Vec3& delta_pos() const { return delta_pos_; }
    double alpha_score() const { return alpha_score_; }
    int iteration() const { return iteration_; }
    const GWOParams& params() const { return params_; }

private:
    GWOParams params_;
    std::vector<Wolf> wolves_;

    Vec3 alpha_pos_{0.0, 0.0, 0.0};
    Vec3 beta_pos_{0.0, 0.0, 0.0};
    Vec3 delta_pos_{0.0, 0.0, 0.0};

    double alpha_score_ = std::numeric_limits<double>::infinity();
    double beta_score_  = std::numeric_limits<double>::infinity();
    double delta_score_ = std::numeric_limits<double>::infinity();

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

    GLFWwindow* window = glfwCreateWindow(1024, 768, "3D Grey Wolf Optimizer (GWO) - Press R to Reset", nullptr, nullptr);
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

    GWOParams params;
    GreyWolfOptimizer gwo(params);

    bool r_was_pressed = false;
    int frame_counter = 0;
    std::cout << std::fixed << std::setprecision(6);

    while (!glfwWindowShouldClose(window)) {
        if (g_stop_requested || glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
            break;
        }

        bool r_pressed = (glfwGetKey(window, GLFW_KEY_R) == GLFW_PRESS);
        if (r_pressed && !r_was_pressed) {
            gwo.reset();
            std::cout << "--- Pack Reset ---\n";
        }
        r_was_pressed = r_pressed;

        if (++frame_counter % 2 == 0) {
            gwo.step_iteration();
            if (gwo.iteration() % 25 == 0) {
                const auto& ap = gwo.alpha_pos();
                std::cout << "Iter " << std::setw(4) << gwo.iteration()
                          << " | Alpha f(x): " << gwo.alpha_score()
                          << " at (" << ap.x << ", " << ap.y << ", " << ap.z << ")\n";
            }
        }
        gwo.update_visuals();

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

        // 2. Render Alpha (Gold), Beta (Crimson), and Delta (Teal) Leaders + Leadership Triangle
        const Vec3& ap = gwo.alpha_pos();
        const Vec3& bp = gwo.beta_pos();
        const Vec3& dp = gwo.delta_pos();

        glDisable(GL_LIGHTING);
        glColor4f(0.85f, 0.62f, 0.05f, 0.45f);
        glLineWidth(1.6f);
        glBegin(GL_LINE_LOOP);
        glVertex3d(ap.x * s, ap.y * s, ap.z * s);
        glVertex3d(bp.x * s, bp.y * s, bp.z * s);
        glVertex3d(dp.x * s, dp.y * s, dp.z * s);
        glEnd();
        glEnable(GL_LIGHTING);

        // Alpha Wolf (Gold)
        glPushMatrix();
        glTranslated(ap.x * s, ap.y * s, ap.z * s);
        glColor3f(0.95f, 0.72f, 0.05f);
        gluSphere(quad, params.sphere_radius * 1.35, 16, 12);
        glPopMatrix();

        // Beta Wolf (Crimson)
        glPushMatrix();
        glTranslated(bp.x * s, bp.y * s, bp.z * s);
        glColor3f(0.85f, 0.20f, 0.30f);
        gluSphere(quad, params.sphere_radius * 1.20, 16, 12);
        glPopMatrix();

        // Delta Wolf (Teal)
        glPushMatrix();
        glTranslated(dp.x * s, dp.y * s, dp.z * s);
        glColor3f(0.10f, 0.65f, 0.72f);
        gluSphere(quad, params.sphere_radius * 1.15, 16, 12);
        glPopMatrix();

        // 3. Render Omega Pack Wolves & 4-Point Trails
        for (const auto& w : gwo.wolves()) {
            glPushMatrix();
            glTranslated(w.visual_pos.x * s, w.visual_pos.y * s, w.visual_pos.z * s);
            glColor3f(w.color.r, w.color.g, w.color.b);
            gluSphere(quad, params.sphere_radius * 0.9, 14, 10);
            glPopMatrix();

            if (w.trail.size() > 1) {
                glDisable(GL_LIGHTING);
                glLineWidth(2.0f);
                glBegin(GL_LINE_STRIP);
                for (size_t k = 0; k < w.trail.size(); ++k) {
                    float alpha = static_cast<float>(k + 1) / static_cast<float>(w.trail.size());
                    glColor4f(w.color.r, w.color.g, w.color.b, alpha * 0.85f);
                    glVertex3d(w.trail[k].x * s, w.trail[k].y * s, w.trail[k].z * s);
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
