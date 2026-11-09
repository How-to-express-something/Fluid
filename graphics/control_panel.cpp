#include "control_panel.h"
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace fluidgl {
ControlPanel::ControlPanel(GLFWwindow* window) {
    IMGUI_CHECKVERSION();
    context_=ImGui::CreateContext();
    if (!context_) throw std::runtime_error("Cannot create parameter panel context");
    ImGui::SetCurrentContext(context_);
    try {
        auto& io=ImGui::GetIO();
        io.IniFilename=nullptr; // No working-directory state or stale off-screen placement.
        io.LogFilename=nullptr;
        io.ConfigFlags|=ImGuiConfigFlags_NavEnableKeyboard;
        ImGui::StyleColorsDark();
        auto& style=ImGui::GetStyle();
        style.WindowRounding=7; style.FrameRounding=4; style.WindowPadding={14,12};
        style.Colors[ImGuiCol_CheckMark]={0.25f,0.9f,0.85f,1};
        // Install after Application's callbacks: the backend chains and restores them.
        platform_=ImGui_ImplGlfw_InitForOpenGL(window,true);
        if (!platform_) throw std::runtime_error("Cannot initialize panel GLFW backend");
        renderer_=ImGui_ImplOpenGL3_Init("#version 330 core");
        if (!renderer_) throw std::runtime_error("Cannot initialize panel OpenGL backend");
    } catch (...) { release(); throw; }
}
void ControlPanel::release() noexcept {
    if (!context_) return;
    ImGui::SetCurrentContext(context_);
    if (frame_open_) ImGui::EndFrame();
    if (renderer_) ImGui_ImplOpenGL3_Shutdown();
    if (platform_) ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext(context_); context_=nullptr;
    renderer_=platform_=frame_open_=false;
}
ControlPanel::~ControlPanel() { release(); }
void ControlPanel::remember(PanelItem item) {
    const auto a=ImGui::GetItemRectMin(),b=ImGui::GetItemRectMax();
    items_[static_cast<unsigned>(item)]={a.x,a.y,b.x,b.y};
}
PanelActions ControlPanel::begin_frame(PanelState& s,const PanelStats& stats) {
    ImGui::SetCurrentContext(context_);
    ImGui_ImplOpenGL3_NewFrame(); ImGui_ImplGlfw_NewFrame(); ImGui::NewFrame();
    frame_open_=true;
    PanelActions actions;
    bool mouse_in_window=false;
    if (visible_) {
        const auto display=ImGui::GetIO().DisplaySize;
        ImGui::SetNextWindowPos({12,12},ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize({std::min(350.0f,std::max(180.0f,display.x-24)),
                                  std::min(760.0f,std::max(160.0f,display.y-24))},ImGuiCond_Always);
        if (ImGui::Begin("Fluid controls (F1)",nullptr,
                         ImGuiWindowFlags_NoCollapse|ImGuiWindowFlags_NoResize|ImGuiWindowFlags_NoFocusOnAppearing)) {
            ImGui::PushItemWidth(155);
            const auto pos=ImGui::GetWindowPos(),size=ImGui::GetWindowSize(),mouse=ImGui::GetIO().MousePos;
            mouse_in_window=mouse.x>=pos.x && mouse.x<pos.x+size.x && mouse.y>=pos.y && mouse.y<pos.y+size.y;
            ImGui::Text("%d x %d | fixed 60 Hz",s.params.gridW,s.params.gridH);
            ImGui::Text("Steps: %llu | sim: %.2f s",stats.steps,stats.steps/60.0);
            ImGui::Text("Frame interval: %.2f ms",stats.frame_interval_ms);
            ImGui::TextDisabled("Wall interval incl. swap; not GPU time");
            ImGui::Text("Density publishes: %u",stats.uploads);
            ImGui::Text("Dropped wall time: %.3f s",stats.dropped_seconds);
            ImGui::Separator();
            ImGui::Checkbox("Pause simulation (Space)",&s.paused); remember(PanelItem::Pause);
            ImGui::BeginDisabled(!s.paused);
            actions.single_step=ImGui::Button("Single step (N)"); remember(PanelItem::SingleStep);
            ImGui::EndDisabled();
            ImGui::SameLine(); actions.reset=ImGui::Button("Reset field (R)"); remember(PanelItem::Reset);
            ImGui::Checkbox("Mouse density + force (M)",&s.mouse_density); remember(PanelItem::MouseDensity);
            ImGui::TextDisabled(s.mouse_density?"LMB: inject and drag":"LMB: force only (no new density)");
            ImGui::Checkbox("Auto demo injection",&s.demo);
            ImGui::Checkbox("Fixed reset seed",&s.fixed_seed); remember(PanelItem::FixedSeed);
            ImGui::BeginDisabled(!s.fixed_seed);
            ImGui::InputInt("Reset seed",&s.seed); s.seed=std::max(0,s.seed);
            ImGui::EndDisabled();
            ImGui::SeparatorText("Simulation");
            constexpr auto clamp=ImGuiSliderFlags_AlwaysClamp;
            ImGui::SliderFloat("Brush radius",&s.params.forceRadius,0.01f,0.3f,"%.3f",clamp);
            ImGui::SliderFloat("Force strength",&s.params.forceStrength,0,1000,"%.0f",clamp);
            ImGui::SliderInt("Jacobi iterations",&s.params.jacobiIter,1,200,"%d",clamp); remember(PanelItem::Jacobi);
            ImGui::SliderFloat("Velocity retain",&s.params.velDissip,0,1,"%.4f",clamp);
            ImGui::SliderFloat("Density retain",&s.params.denDissip,0,1,"%.4f",clamp);
            ImGui::SliderFloat("Viscosity",&s.params.viscosity,0,5,"%.3f",clamp);
            ImGui::SliderInt("Diffuse iterations",&s.params.diffuseIter,1,100,"%d",clamp);
            ImGui::SliderFloat("Buoyancy",&s.params.buoyancy,-10,10,"%.2f",clamp);
            if (ImGui::Button("Restore simulation defaults")) {
                const int w=s.params.gridW,h=s.params.gridH;
                s.params=SimParams{}; s.params.gridW=w; s.params.gridH=h;
            }
            ImGui::SeparatorText("Display");
            ImGui::Checkbox("Nebula colors (1/2)",&s.color);
            ImGui::SliderFloat("Display gain",&s.gain,0.01f,1,"%.3f",clamp); remember(PanelItem::Gain);
            ImGui::TextWrapped("Simulation settings apply at the next step. Reset keeps settings and pause state.");
            ImGui::PopItemWidth();
        }
        ImGui::End();
    }
    // ImGui's initial fallback window may acquire navigation focus. Leave the
    // application focused on startup unless the user is already interacting.
    if (first_frame_) {
        if (!ImGui::IsAnyItemActive() && !ImGui::IsMouseDown(0)) ImGui::SetWindowFocus(nullptr);
        first_frame_=false;
    }
    // Also guard non-finite values typed into temporary numeric editors.
    const SimParams defaults;
    auto finite=[](float& value,float fallback) { if (!std::isfinite(value)) value=fallback; };
    finite(s.params.forceRadius,defaults.forceRadius); finite(s.params.forceStrength,defaults.forceStrength);
    finite(s.params.velDissip,defaults.velDissip); finite(s.params.denDissip,defaults.denDissip);
    finite(s.params.viscosity,defaults.viscosity); finite(s.params.buoyancy,defaults.buoyancy);
    finite(s.gain,0.2f);
    mouse_capture_=mouse_in_window || ImGui::GetIO().WantCaptureMouse
        || (visible_ && ImGui::IsWindowHovered(ImGuiHoveredFlags_AnyWindow))
        || ImGui::IsAnyItemActive();
    return actions;
}
void ControlPanel::render() {
    ImGui::SetCurrentContext(context_);
    ImGui::Render(); frame_open_=false;
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}
bool ControlPanel::captures_mouse() const { return mouse_capture_; }
bool ControlPanel::captures_keyboard() const {
    ImGui::SetCurrentContext(context_);
    return ImGui::GetIO().WantCaptureKeyboard;
}
} // namespace fluidgl
