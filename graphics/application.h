#pragma once
#include "fluid.h"
#include "renderer.h"
#include "brush_input.h"
#include "control_panel.h"
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#include <memory>
#include <array>

namespace fluidgl {
class Application {
public:
    explicit Application(GLFWwindow* window,bool enable_panel=true);
    ~Application();
    void run(bool preview,bool demo);
    void smoke_test();
    void panel_test();
private:
    void reset();
    void sample_mouse(double elapsed);
    void update_controls(double elapsed);
    void set_paused(bool paused);
    void tick(double elapsed);
    void simulate(double dt);
    Canvas render();
    static void key(GLFWwindow*,int,int,int,int);
    GLFWwindow* window_;
    std::unique_ptr<Renderer> renderer_;
    // Reverse destruction order: unregister CUDA before deleting the GL texture.
    std::unique_ptr<FluidSim> fluid_;
    FixedStep clock_;
    bool paused_=false,single_=false,color_=true,demo_=false,dirty_=true;
    bool brush_=false;
    bool mouse_density_=true;
    bool reset_requested_=false;
    Point brush_position_,brush_velocity_;
    BrushInput brush_input_;
    unsigned long long steps_=0;
    float gain_=0.2f;
    PanelState panel_state_;
    std::unique_ptr<ControlPanel> panel_;
};
} // namespace fluidgl
