#include "application.h"
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <random>

namespace fluidgl {
namespace {
void require(bool ok,const char* message) { if (!ok) throw std::runtime_error(message); }
void save(const char* file,const std::vector<unsigned char>& rgb,int w,int h) {
    std::ofstream out(file,std::ios::binary);
    out << "P6\n" << w << ' ' << h << "\n255\n";
    for (int y=h-1;y>=0;--y)
        out.write(reinterpret_cast<const char*>(rgb.data()+static_cast<std::size_t>(y)*w*3),w*3);
    require(static_cast<bool>(out),"Cannot save Fluid screenshot");
}
float sample(const float* data,int w,int h,double u,double v) {
    const double x=u*w-0.5,y=v*h-0.5;
    const int ix=static_cast<int>(std::floor(x)),iy=static_cast<int>(std::floor(y));
    const double fx=x-ix,fy=y-iy;
    auto at=[&](int xx,int yy) {return data[std::clamp(yy,0,h-1)*w+std::clamp(xx,0,w-1)];};
    return static_cast<float>((1-fy)*((1-fx)*at(ix,iy)+fx*at(ix+1,iy))
        +fy*((1-fx)*at(ix,iy+1)+fx*at(ix+1,iy+1)));
}
}
Application::Application(GLFWwindow* window,bool enable_panel):window_(window) {
    const SimConfig config{};
    renderer_=std::make_unique<Renderer>(config.gridW,config.gridH);
    fluid_=std::make_unique<FluidSim>(config,FluidSim::noiseDensity(config,std::random_device{}()));
    fluid_->attachGLTexture(renderer_->texture());
    glfwSetWindowUserPointer(window_,this); glfwSetKeyCallback(window_,key);
    try { if (enable_panel) panel_=std::make_unique<ControlPanel>(window_); }
    catch (...) { glfwSetKeyCallback(window_,nullptr); glfwSetWindowUserPointer(window_,nullptr); throw; }
}
Application::~Application() {
    panel_.reset(); // Backend restores chained callbacks before Application clears them.
    glfwSetKeyCallback(window_,nullptr); glfwSetWindowUserPointer(window_,nullptr);
}
void Application::key(GLFWwindow* window,int k,int,int action,int) {
    if (action!=GLFW_PRESS) return;
    auto& app=*static_cast<Application*>(glfwGetWindowUserPointer(window));
    if (k==GLFW_KEY_F1 && app.panel_) { app.panel_->toggle(); app.brush_input_.reset(); return; }
    if (app.panel_ && app.panel_->captures_keyboard()) return;
    if (k==GLFW_KEY_ESCAPE) glfwSetWindowShouldClose(window,GLFW_TRUE);
    if (k==GLFW_KEY_SPACE) {
        app.set_paused(!app.paused_);
    }
    if (k==GLFW_KEY_N && app.paused_) app.single_=true;
    if (k==GLFW_KEY_M) app.mouse_density_=!app.mouse_density_;
    // Defer reset to the main loop: never throw through a C callback.
    if (k==GLFW_KEY_R) app.reset_requested_=true;
    if (k==GLFW_KEY_1) app.color_=false;
    if (k==GLFW_KEY_2) app.color_=true;
}
void Application::reset() {
    const auto config=fluid_->config();
    const auto params=fluid_->params();
    const auto seed=panel_state_.fixed_seed ? static_cast<std::uint32_t>(panel_state_.seed) : std::random_device{}();
    auto next=std::make_unique<FluidSim>(config,FluidSim::noiseDensity(config,seed));
    next->params()=params;
    fluid_->attachGLTexture(0);
    next->attachGLTexture(renderer_->texture());
    fluid_=std::move(next);
    clock_=FixedStep{}; steps_=0; single_=false; dirty_=true;
    brush_=false; brush_velocity_={}; brush_input_.reset();
}
void Application::set_paused(bool paused) {
    paused_=paused; single_=false; clock_.accumulator=0;
    brush_input_.reset(); brush_velocity_={};
}
void Application::update_controls(double elapsed) {
    if (!panel_) return;
    panel_state_.params=fluid_->params(); panel_state_.paused=paused_;
    panel_state_.mouse_density=mouse_density_; panel_state_.demo=demo_;
    panel_state_.color=color_; panel_state_.gain=gain_;
    const auto actions=panel_->begin_frame(panel_state_,
        {steps_,renderer_->updates(),elapsed*1000,clock_.dropped_seconds});
    if (paused_!=panel_state_.paused) set_paused(panel_state_.paused);
    fluid_->params()=panel_state_.params;
    mouse_density_=panel_state_.mouse_density; demo_=panel_state_.demo;
    color_=panel_state_.color; gain_=panel_state_.gain;
    if (actions.single_step && paused_) single_=true;
    if (actions.reset) reset_requested_=true; // Resource changes happen in the main loop.
}
void Application::sample_mouse(double elapsed) {
    int ww=0,wh=0,fw=0,fh=0;
    glfwGetWindowSize(window_,&ww,&wh); glfwGetFramebufferSize(window_,&fw,&fh);
    Point cursor; glfwGetCursorPos(window_,&cursor.x,&cursor.y);
    const auto sample=brush_input_.sample(cursor,ww,wh,fw,fh,fluid_->gridW(),fluid_->gridH(),
        glfwGetWindowAttrib(window_,GLFW_FOCUSED)!=0,
        glfwGetMouseButton(window_,GLFW_MOUSE_BUTTON_LEFT)==GLFW_PRESS,
        panel_ && panel_->captures_mouse(),elapsed);
    brush_=sample.active; brush_position_=sample.position; brush_velocity_=sample.velocity;
}
void Application::simulate(double dt) {
    fluid_->params().dt=static_cast<float>(dt);
    auto inject=[&](Point p,Point velocity,float density) {
        fluid_->inject({static_cast<float>(p.x),static_cast<float>(p.y),
                        static_cast<float>(velocity.x)*0.10f,static_cast<float>(velocity.y)*0.10f,density});
    };
    if (brush_) inject(brush_position_,brush_velocity_,mouse_density_ ? 300.0f : 0.0f);
    if (demo_) {
        const double t=steps_*FixedStep::dt;
        inject({fluid_->gridW()*(0.5+0.26*std::sin(t)),fluid_->gridH()*(0.5+0.26*std::cos(t))},
               {60*std::cos(t),-60*std::sin(t)},300.0f);
    }
    fluid_->step(); // All solver work and publishToGL use stream 0.
    ++steps_; dirty_=true;
}
void Application::tick(double elapsed) {
    clock_.advance(elapsed,paused_,single_,[&](double dt){simulate(dt);}); single_=false;
}
Canvas Application::render() {
    if (dirty_) { renderer_->update_density(*fluid_); dirty_=false; }
    int fw=0,fh=0; glfwGetFramebufferSize(window_,&fw,&fh);
    return renderer_->draw(fw,fh,color_,gain_,static_cast<float>(steps_*FixedStep::dt));
}
void Application::run(bool preview,bool demo) {
    demo_=demo || preview;
    glfwSwapInterval(1);
    double last=glfwGetTime(),start=last,title_time=last;
    std::cout << "F1: parameter panel | LMB: inject/drag | M: toggle mouse density (force stays on) | Space: pause | N: single step (paused) | R: reset | 1/2: gray/color | Esc: quit\n";
    while (!glfwWindowShouldClose(window_)) {
        glfwPollEvents();
        if (glfwWindowShouldClose(window_)) break;
        double now=glfwGetTime();
        double elapsed=now-last; last=now;
        int fw=0,fh=0; glfwGetFramebufferSize(window_,&fw,&fh);
        if (fw<=0 || fh<=0 || glfwGetWindowAttrib(window_,GLFW_ICONIFIED)) {
            clock_.accumulator=0; brush_input_.reset(); brush_=false;
            glfwWaitEventsTimeout(0.05); last=glfwGetTime(); continue;
        }
        update_controls(elapsed);
        if (reset_requested_) { reset_requested_=false; reset(); elapsed=0; }
        sample_mouse(elapsed); tick(elapsed); render();
        if (panel_) panel_->render();
        glfwSwapBuffers(window_);
        if (now-title_time>0.25) {
            std::string title="Fluid | "+std::to_string(fluid_->gridW())+"x"
                +std::to_string(fluid_->gridH())+" | fixed 60 Hz | steps "+std::to_string(steps_)
                +(paused_ ? " | PAUSED" : " | RUNNING")
                +(mouse_density_ ? " | M: DENSITY + FORCE" : " | M: FORCE ONLY");
            glfwSetWindowTitle(window_,title.c_str()); title_time=now;
        }
        if (preview && now-start>=3) break;
    }
    std::cout << "Simulation steps: " << steps_ << ", simulated seconds: " << steps_*FixedStep::dt
              << ", dropped wall seconds: " << clock_.dropped_seconds
              << ", texture allocations: 1, density uploads: " << renderer_->updates() << '\n';
}
void Application::smoke_test() {
    const int gw=fluid_->gridW(),gh=fluid_->gridH();
    auto check_zero_texture=[&]() {
        int w=0,h=0; glfwGetFramebufferSize(window_,&w,&h);
        const auto canvas=render();
        const auto rgb=renderer_->read_rgb(w,h);
        for (int y=canvas.y;y<canvas.y+canvas.height;++y)
            for (int x=canvas.x;x<canvas.x+canvas.width;++x)
                for (int k=0;k<3;++k)
                    require(rgb[(y*w+x)*3+k]==0,"Initial/reset interop texture is not zero");
    };
    // Move an attached simulation: the CUDA registration must have exactly one owner.
    fluid_=std::make_unique<FluidSim>(std::move(*fluid_));
    fluid_->attachGLTexture(renderer_->texture()); // Same attachment is a no-op.
    render(); // Publish before any diagnostic synchronization.
    auto density=fluid_->readbackDensity();
    require(*std::max_element(density.begin(),density.end())>0,"Noise initialization is empty");
    std::vector<float> texture_density(gw*gh);
    glBindTexture(GL_TEXTURE_2D,renderer_->texture());
    glGetTexImage(GL_TEXTURE_2D,0,GL_RED,GL_FLOAT,texture_density.data());
    require(glGetError()==GL_NO_ERROR && texture_density==density,"Initial noise did not reach GL texture exactly");
    {
        int w=0,h=0; glfwGetFramebufferSize(window_,&w,&h);
        save("fluid_nebula.ppm",renderer_->read_rgb(w,h),w,h);
    }
    GLuint temporary=0; glGenTextures(1,&temporary); glBindTexture(GL_TEXTURE_2D,temporary);
    glTexImage2D(GL_TEXTURE_2D,0,GL_R32F,gw-1,gh,0,GL_RED,GL_FLOAT,nullptr);
    bool rejected=false;
    try { fluid_->attachGLTexture(temporary); } catch (const std::invalid_argument&) { rejected=true; }
    require(rejected,"Mismatched interop texture dimensions were accepted");
    glTexImage2D(GL_TEXTURE_2D,0,GL_R32F,gw,gh,0,GL_RED,GL_FLOAT,nullptr);
    fluid_->attachGLTexture(temporary); fluid_->publishToGL();
    fluid_->attachGLTexture(renderer_->texture());
    glDeleteTextures(1,&temporary);
    std::cout << "PASS: noise texture equality, registered move, texture validation/replacement\n";
    // Start injection/orientation tests with a clean field.
    fluid_->resize(gw,gh); fluid_->attachGLTexture(renderer_->texture()); dirty_=true;
    check_zero_texture();
    std::cout << "PASS: initial device density reaches registered GL texture\n";
    // Use the SAME cursor mapping and injection path as interactive input.
    int ww=0,wh=0,fw=0,fh=0;
    glfwGetWindowSize(window_,&ww,&wh); glfwGetFramebufferSize(window_,&fw,&fh);
    auto c=fit_canvas(fw,fh,gw,gh);
    Point cursor{(c.x+c.width*0.25)*ww/fw,(fh-c.y-c.height+c.height*0.2)*wh/fh};
    auto p=cursor_to_grid(cursor,ww,wh,fw,fh,c,gw,gh);
    require(p.has_value(),"Test injection cursor is outside canvas");
    brush_=true; brush_position_=*p; tick(FixedStep::dt); brush_=false;
    render(); // Exercise solver -> interop ordering BEFORE any diagnostic CPU synchronization.
    density=fluid_->readbackDensity();
    double sum=0; int peak=0;
    for (int i=0;i<gw*gh;++i) {
        require(std::isfinite(density[i]),"Non-finite density returned by CUDA");
        sum+=density[i]; if (density[i]>density[peak]) peak=i;
    }
    require(sum>0 && std::abs(peak%gw-p->x)<2 && std::abs(peak/gw-p->y)<2,
            "CUDA injection did not appear at mapped top-left grid position");
    std::cout << "PASS: real CUDA injection and density readback, sum=" << sum << '\n';
    auto check_pixels=[&](int width,int height,bool color) {
        glfwSetWindowSize(window_,width,height); glfwPollEvents();
        glfwGetFramebufferSize(window_,&fw,&fh); color_=false; c=render();
        auto rgb=renderer_->read_rgb(fw,fh);
        for (int j=0;j<23;++j) for (int i=0;i<31;++i) {
            const int x=c.x+i*(c.width-1)/30,y=c.y+j*(c.height-1)/22;
            const float d=std::clamp(sample(density.data(),gw,gh,
                (x+0.5-c.x)/c.width,1-(y+0.5-c.y)/c.height)*gain_,0.0f,1.0f);
            for (int k=0;k<3;++k) require(std::abs(static_cast<int>(rgb[(y*fw+x)*3+k])-std::lround(d*255))<=3,
                "Rendered gray pixel differs from CPU density sample (UV/aspect)");
        }
        // At least one bar is present at both tested aspect ratios.
        require((c.x>0 || c.y>0) && std::abs(static_cast<int>(rgb[0])-9)<=1
            && std::abs(static_cast<int>(rgb[1])-11)<=1 && std::abs(static_cast<int>(rgb[2])-15)<=1,
            "Letterbox clear or viewport is incorrect");
        if (color) {
            const auto gray=rgb;
            color_=true; render(); rgb=renderer_->read_rgb(fw,fh);
            require(rgb!=gray,"Nebula mode is indistinguishable from gray mode");
            int chroma=0;
            for (int y=c.y;y<c.y+c.height;++y) for (int x=c.x;x<c.x+c.width;++x) {
                const auto i=(y*fw+x)*3;
                chroma=std::max(chroma,std::abs(int(rgb[i])-int(rgb[i+2])));
            }
            require(chroma>30,"Nebula palette has no visible chromatic contrast");
            render();
            require(renderer_->read_rgb(fw,fh)==rgb,"Rendering at fixed simulation time must be stable");
            renderer_->draw(fw,fh,true,gain_,static_cast<float>(steps_*FixedStep::dt)+5.0f);
            require(renderer_->read_rgb(fw,fh)!=rgb,"Nebula color animation does not advance");
            render();
            save("fluid_smoke.ppm",rgb,fw,fh);
        }
        std::cout << "PASS: " << fw << 'x' << fh << ", 713 pixel samples, "
                  << (color ? "color" : "gray") << ", letterbox/orientation\n";
    };
    check_pixels(1024,720,false); check_pixels(1200,500,true);
    const std::vector<float> frozen(density.data(),density.data()+gw*gh);
    const auto updates=renderer_->updates();
    const auto frozen_pixels=renderer_->read_rgb(fw,fh);
    const auto before=steps_; paused_=true; tick(2);
    render();
    require(renderer_->read_rgb(fw,fh)==frozen_pixels,"Pause advanced the nebula animation");
    require(renderer_->updates()==updates,"Paused render unnecessarily updated interop texture");
    density=fluid_->readbackDensity();
    require(steps_==before && std::equal(frozen.begin(),frozen.end(),density.data()),"Pause advanced the simulation");
    single_=true; tick(2); require(steps_==before+1,"Single step advanced more than once");
    std::cout << "PASS: pause and exactly one fixed simulation step\n";
    reset(); tick(1); require(steps_==0,"Reset failed to retain pause");
    density=fluid_->readbackDensity();
    require(*std::max_element(density.begin(),density.end())>0,"Reset did not regenerate noise");
    render();
    fluid_->resize(gw,gh); fluid_->attachGLTexture(renderer_->texture()); dirty_=true;
    check_zero_texture();
    single_=true; tick(0);
    density=fluid_->readbackDensity();
    require(std::all_of(density.data(),density.data()+gw*gh,[](float v){return v==0;}),
            "Reset did not clear CUDA simulation state");
    check_zero_texture(); std::cout << "PASS: reset noise and resize clearing reach interop texture\n";
    auto transported_center=[&](float vx,float vy) {
        FluidSim probe({gw,gh},std::vector<float>(gw*gh,0));
        probe.inject({gw*0.5f,gh*0.5f,vx,vy,300});
        for (int i=0;i<12;++i) probe.step();
        const auto snapshot=probe.readbackDensity();
        double mass=0,mx=0,my=0;
        for (int y=0;y<gh;++y) for (int x=0;x<gw;++x) {
            const float d=snapshot[y*gw+x];
            require(std::isfinite(d),"Non-finite density in velocity direction check");
            mass+=d; mx+=d*(x+0.5); my+=d*(y+0.5);
        }
        require(mass>0,"Velocity probe lost all density");
        return Point{mx/mass,my/mass};
    };
    const auto right=transported_center(60,0),left=transported_center(-60,0);
    const auto down=transported_center(0,60),up=transported_center(0,-60);
    require(right.x>left.x+0.1 && down.y>up.y+0.1,"Injected velocity advects density in the wrong direction");
    std::cout << "PASS: drag velocity direction, x=" << left.x << " -> " << right.x
              << ", y=" << up.y << " -> " << down.y << '\n';


}
} // namespace fluidgl
