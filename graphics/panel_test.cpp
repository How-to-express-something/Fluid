#include "application.h"
#include <imgui.h>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace fluidgl {
namespace {
void require(bool value,const char* message) { if (!value) throw std::runtime_error(message); }
}

// Real ImGui widgets/backends, synthetic input events, and the normal apply/tick/render path.
// This does not claim coverage of OS event delivery or physical high-DPI monitors.
void Application::panel_test() {
    require(panel_!=nullptr,"Panel regression requires a panel");
    glfwSwapInterval(0);
    auto& io=ImGui::GetIO();
    io.ConfigInputTrickleEventQueue=false;
    io.AddFocusEvent(true);
    set_paused(true);
    auto frame=[&]() {
        glfwPollEvents();
        update_controls(FixedStep::dt);
        if (reset_requested_) { reset_requested_=false; reset(); }
        brush_=false;
        tick(FixedStep::dt); render(); panel_->render();
        require(glGetError()==GL_NO_ERROR,"Panel rendering left an OpenGL error");
    };
    auto point=[&](PanelItem item,float fraction=0.45f) {
        const auto r=panel_->item_rect(item);
        require(r.x2>r.x1 && r.y2>r.y1,"Panel widget has no layout rectangle");
        io.AddMousePosEvent(r.x1+(r.x2-r.x1)*fraction,(r.y1+r.y2)*0.5f);
    };
    auto click=[&](PanelItem item,float fraction=0.45f) {
        point(item,fraction); frame();
        io.AddMouseButtonEvent(0,true); frame();
        io.AddMouseButtonEvent(0,false); frame();
    };
    frame(); frame();
    require(steps_==0,"Creating a paused panel advanced the simulation");
    require(!panel_->captures_keyboard(),"Panel stole keyboard focus before any interaction");
    key(window_,GLFW_KEY_M,0,GLFW_PRESS,0);
    require(!mouse_density_,"Application shortcut was captured before panel interaction");
    key(window_,GLFW_KEY_M,0,GLFW_PRESS,0);
    const auto field_before_display=fluid_->readbackDensity();
    const auto publishes_before_display=renderer_->updates();
    click(PanelItem::Gain,0.08f);
    require(gain_!=0.2f && gain_>=0.01f && gain_<=1,"Display gain slider did not update");
    require(fluid_->readbackDensity()==field_before_display && steps_==0
        && renderer_->updates()==publishes_before_display,"Display-only slider changed simulation or republished density");
    gain_=0.2f; frame();
    click(PanelItem::SingleStep);
    require(steps_==1 && paused_,"UI single step must advance exactly once and retain pause");
    const auto publishes=renderer_->updates();
    frame(); frame();
    require(steps_==1 && renderer_->updates()==publishes,"Paused panel frames republished or simulated density");
    click(PanelItem::Pause);
    require(!paused_ && steps_>1,"UI resume did not resume fixed-step simulation");
    click(PanelItem::Pause);
    require(paused_,"UI pause did not pause");
    const auto stopped=steps_;
    frame(); require(steps_==stopped,"UI pause left a simulation backlog");

    click(PanelItem::MouseDensity);
    require(!mouse_density_,"UI density switch did not enable force-only mode");
    click(PanelItem::Jacobi,0.08f);
    const auto selected=fluid_->params().jacobiIter;
    require(selected!=60 && selected>=1 && selected<=200,"UI Jacobi slider did not update simulation parameters");
    require(panel_->captures_keyboard(),"Focused panel did not capture keyboard input");
    key(window_,GLFW_KEY_R,0,GLFW_PRESS,0);
    key(window_,GLFW_KEY_SPACE,0,GLFW_PRESS,0);
    key(window_,GLFW_KEY_M,0,GLFW_PRESS,0);
    require(!reset_requested_ && paused_ && !mouse_density_,"Panel keyboard input triggered application shortcuts");
    click(PanelItem::SingleStep); // Executes the configured iteration count, including graph rebuild.
    require(fluid_->params().jacobiIter==selected,"Single step lost panel parameters");
    for (float value:fluid_->readbackDensity()) require(std::isfinite(value),"UI parameter step produced non-finite density");

    click(PanelItem::FixedSeed);
    require(panel_state_.fixed_seed,"Fixed seed checkbox did not update");
    click(PanelItem::Reset);
    require(paused_ && steps_==0 && !mouse_density_ && fluid_->params().jacobiIter==selected,
            "UI reset did not preserve settings or pause");
    require(fluid_->readbackDensity()==FluidSim::noiseDensity(fluid_->config(),123),"Seeded UI reset was not reproducible");

    // Drive the same brush policy used by sample_mouse with logical window coordinates.
    BrushInput probe;
    const auto rect=panel_->item_rect(PanelItem::Jacobi);
    auto sample=[&](Point cursor,bool down,bool capture) {
        return probe.sample(cursor,1024,720,1024,720,fluid_->gridW(),fluid_->gridH(),true,down,capture,FixedStep::dt);
    };
    point(PanelItem::Jacobi); frame();
    require(panel_->captures_mouse(),"Hovering the real panel did not capture mouse input");
    require(!sample({rect.x1,rect.y1},true,panel_->captures_mouse()).active,"Panel press generated a brush");
    require(!sample({700,350},true,false).active,"Dragging from UI into canvas generated a brush");
    sample({700,350},false,false);
    require(sample({700,350},true,false).active,"Releasing UI gesture did not restore canvas input");

    fluid_->resize(fluid_->gridW(),fluid_->gridH());
    fluid_->attachGLTexture(renderer_->texture()); dirty_=true;
    brush_=true; brush_position_={128,128}; brush_velocity_={60,0};
    single_=true; tick(FixedStep::dt); brush_=false;
    const auto empty=fluid_->readbackDensity();
    require(std::all_of(empty.begin(),empty.end(),[](float d){return d==0;}),"Force-only UI mode injected density");
    click(PanelItem::Reset);
    const auto noise=fluid_->readbackDensity();
    FluidSim unforced(fluid_->config(),noise); unforced.params()=fluid_->params(); unforced.step();
    const auto quiet=unforced.readbackDensity();
    brush_=true; brush_position_={128,128}; brush_velocity_={60,0};
    single_=true; tick(FixedStep::dt); brush_=false;
    const auto stirred=fluid_->readbackDensity();
    bool changed=false;
    for (std::size_t i=0;i<quiet.size();++i) changed|=std::abs(quiet[i]-stirred[i])>1e-4f;
    require(changed,"Force-only mode stopped applying force to existing density");
    click(PanelItem::Reset);

    // Clicking the canvas releases keyboard focus so application hotkeys work again.
    io.AddMousePosEvent(700,350); frame();
    io.AddMouseButtonEvent(0,true); frame(); io.AddMouseButtonEvent(0,false); frame();
    require(!panel_->captures_keyboard(),"Canvas click did not release panel keyboard focus");
    const auto before_hotkey=steps_;
    key(window_,GLFW_KEY_N,0,GLFW_PRESS,0); frame();
    require(steps_==before_hotkey+1,"Application shortcut did not resume outside UI");
    click(PanelItem::Reset);

    // Verify overlay visibility and next fluid render after ImGui restores GL state.
    frame();
    int fw=0,fh=0; glfwGetFramebufferSize(window_,&fw,&fh);
    const auto overlay=renderer_->read_rgb(fw,fh);
    key(window_,GLFW_KEY_F1,0,GLFW_PRESS,0); frame(); frame();
    require(!panel_->visible(),"F1 did not hide the panel");
    const auto hidden=renderer_->read_rgb(fw,fh);
    require(hidden!=overlay,"Panel hide did not change the framebuffer");
    for (int y=0;y<fh;++y) for (int x=500;x<fw;++x) for (int k=0;k<3;++k)
        require(hidden[(y*fw+x)*3+k]==overlay[(y*fw+x)*3+k],"UI altered the fluid rendering outside its overlay");
    key(window_,GLFW_KEY_F1,0,GLFW_PRESS,0); frame();
    require(panel_->visible(),"F1 did not restore the panel");
    glfwSetWindowSize(window_,640,480); frame(); frame();
    require(paused_ && steps_==0,"Resizing the panel/window advanced the paused simulation");
    glfwSetWindowSize(window_,1024,720); frame(); frame();
    glfwGetFramebufferSize(window_,&fw,&fh);
    const auto rgb=renderer_->read_rgb(fw,fh);
    std::ofstream out("fluid_panel.ppm",std::ios::binary);
    out << "P6\n" << fw << ' ' << fh << "\n255\n";
    for (int y=fh-1;y>=0;--y) out.write(reinterpret_cast<const char*>(rgb.data()+std::size_t(y)*fw*3),fw*3);
    require(static_cast<bool>(out),"Cannot save panel regression screenshot");
    set_paused(false);
    run(true,true); // Exercise the actual interactive loop, not only the test frame driver.
    require(steps_>0 && demo_,"Interactive preview loop did not advance with the panel enabled");
    for (float value:fluid_->readbackDensity())
        require(std::isfinite(value) && value>=0,"Panel preview loop produced invalid density");
    std::cout << "PASS: real panel pause/resume/single-step, parameter apply/graph rebuild, seeded reset, "
                 "UI input capture, force-only injection, hide/show, GL state and interactive preview loop\n";
}
} // namespace fluidgl
