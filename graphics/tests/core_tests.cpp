#include "../canvas.h"
#include "../brush_input.h"
#include <iostream>
#include <stdexcept>

using namespace fluidgl;
void require(bool v,const char* message) { if (!v) throw std::runtime_error(message); }
bool near(double a,double b) { return std::abs(a-b)<1e-8; }
int main() {
    try {
        for (auto dims : {Point{1024,720},Point{1200,500},Point{500,1200}}) {
            const int fw=static_cast<int>(dims.x),fh=static_cast<int>(dims.y);
            const auto c=fit_canvas(fw,fh,256,163);
            require(c.width<=fw && c.height<=fh,"Canvas exceeds framebuffer");
            require(std::abs(static_cast<double>(c.width)/c.height-256.0/163)<0.004,"Canvas aspect changed");
            // Synthetic 1x, 1.5x and 2x DPI, including different X/Y scale factors.
            for (auto dpi : {Point{1,1},Point{1.5,1.5},Point{2,2},Point{2,1.5}}) {
                const int ww=static_cast<int>(fw/dpi.x),wh=static_cast<int>(fh/dpi.y);
                for (auto uv : {Point{0,0},Point{0.5,0.5},Point{0.99,0.99},Point{0.2,0.8}}) {
                    const Point cursor{(c.x+c.width*uv.x)*ww/fw,
                        (fh-c.y-c.height+c.height*uv.y)*wh/fh};
                    auto p=cursor_to_grid(cursor,ww,wh,fw,fh,c,256,163);
                    require(p && near(p->x,uv.x*256) && near(p->y,uv.y*163),"High DPI cursor mapping failed");
                }
                require(!cursor_to_grid({-1,-1},ww,wh,fw,fh,c,256,163),"Outside cursor was accepted");
                if (c.x>0 || c.y>0)
                    require(!cursor_to_grid({0,0},ww,wh,fw,fh,c,256,163),"Letterbox cursor was accepted");
            }
        }
        require(fit_canvas(0,0,256,163).width==0,"Minimized canvas should be empty");
        require(!cursor_to_grid({0,0},0,0,0,0,{},256,163),"Minimized cursor accepted");
        BrushInput input;
        auto brush=[&](Point cursor,bool down,bool ui,bool focus=true,int fw=1024,int fh=720) {
            return input.sample(cursor,1024,720,fw,fh,256,256,focus,down,ui,1.0/60);
        };
        require(!brush({200,150},true,true).active,"UI press reached the fluid");
        require(!brush({700,350},true,false).active,"UI drag leaving the panel reached the fluid");
        brush({700,350},false,false);
        auto first=brush({700,350},true,false);
        require(first.active && first.velocity.x==0 && first.velocity.y==0,"New canvas press inherited UI drag velocity");
        const auto dragged=brush({710,360},true,false);
        require(dragged.active && dragged.velocity.x>0 && std::hypot(dragged.velocity.x,dragged.velocity.y)<=200.00001,
                "Canvas drag direction or speed cap changed");
        require(!brush({710,360},true,true).active,"Canvas drag crossing UI remained active");
        require(!brush({700,350},true,false).active,"Crossing UI did not cancel the gesture");
        brush({700,350},false,false);
        brush({700,350},true,false);
        require(!brush({700,350},true,false,false).active,"Unfocused window injected");
        require(!brush({700,350},true,false,true).active,"Focus return resumed a held gesture");
        brush({700,350},false,false);
        brush({700,350},true,false);
        auto resized=brush({700,350},true,false,true,2048,1440);
        require(resized.active && resized.velocity.x==0 && resized.velocity.y==0,"DPI/geometry change created a velocity spike");
        for (int fps : {30,60,120,144}) {
            FixedStep timer; int count=0;
            for (int i=0;i<fps;++i) timer.advance(1.0/fps,false,false,[&](double dt){
                require(near(dt,1.0/60),"Simulation dt depends on render FPS"); ++count; });
            require(count==60,"Simulation speed depends on render FPS");
            timer.advance(5,true,false,[&](double){++count;}); require(count==60,"Pause advanced");
            timer.advance(5,true,true,[&](double){++count;}); require(count==61,"Single step failed");
            timer.advance(0,false,false,[&](double){++count;}); require(count==61,"Pause left a backlog");
        }
        FixedStep overloaded; int count=0;
        overloaded.advance(10,false,false,[&](double){++count;});
        require(count==8 && overloaded.dropped_seconds>9,"Catch-up limit failed");
        std::cout << "PASS: letterbox, DPI mapping, outside rejection, fixed timestep, pause/step, catch-up limit\n";
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
