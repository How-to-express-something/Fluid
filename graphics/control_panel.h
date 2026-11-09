#pragma once
#include "sim_types.h"
#include <array>

struct GLFWwindow;
struct ImGuiContext;

namespace fluidgl {
struct PanelState {
    SimParams params{};
    bool paused=false,mouse_density=true,demo=false,color=true;
    bool fixed_seed=false;
    int seed=123;
    float gain=0.2f;
};
struct PanelStats {
    unsigned long long steps=0;
    unsigned uploads=0;
    double frame_interval_ms=0,dropped_seconds=0;
};
struct PanelActions { bool single_step=false,reset=false; };
enum class PanelItem { Pause,SingleStep,Reset,MouseDensity,Jacobi,FixedSeed,Gain,Count };
struct PanelRect { float x1=0,y1=0,x2=0,y2=0; };

// Owns one ImGui context and its backends; requires a current GL context.
class ControlPanel {
public:
    explicit ControlPanel(GLFWwindow* window);
    ~ControlPanel();
    ControlPanel(const ControlPanel&)=delete;
    ControlPanel& operator=(const ControlPanel&)=delete;
    PanelActions begin_frame(PanelState&,const PanelStats&);
    void render();
    bool captures_mouse() const;
    bool captures_keyboard() const;
    void toggle() { visible_=!visible_; }
    bool visible() const { return visible_; }
    // Stable widget bounds allow actual UI input regression without screen coordinates.
    PanelRect item_rect(PanelItem item) const { return items_[static_cast<unsigned>(item)]; }
private:
    void remember(PanelItem);
    void release() noexcept;
    ImGuiContext* context_=nullptr;
    bool platform_=false,renderer_=false,visible_=true,frame_open_=false;
    bool first_frame_=true;
    bool mouse_capture_=false;
    std::array<PanelRect,static_cast<unsigned>(PanelItem::Count)> items_{};
};
} // namespace fluidgl
