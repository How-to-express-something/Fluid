#pragma once
#include "canvas.h"
#include <array>

namespace fluidgl {
struct BrushSample { bool active=false; Point position{},velocity{}; };

// OS-independent policy shared by interactive input and regression checks.
class BrushInput {
public:
    void reset() { previous_.reset(); }
    BrushSample sample(Point cursor,int ww,int wh,int fw,int fh,int gw,int gh,
                       bool focused,bool left_down,bool ui_capture,double elapsed) {
        const std::array<int,4> geometry{ww,wh,fw,fh};
        if (geometry!=geometry_) { reset(); geometry_=geometry; }
        if (!left_down) blocked_=false;
        // Crossing UI or losing focus cancels the gesture until the next press.
        if (left_down && (ui_capture || !focused)) blocked_=true;
        const auto mapped=cursor_to_grid(cursor,ww,wh,fw,fh,fit_canvas(fw,fh,gw,gh),gw,gh);
        if (!focused || !left_down || ui_capture || blocked_ || !mapped) { reset(); return {}; }
        BrushSample result{true,*mapped,{}};
        if (previous_ && elapsed>1e-6) {
            result.velocity={(mapped->x-previous_->x)/elapsed,(mapped->y-previous_->y)/elapsed};
            const double speed=std::hypot(result.velocity.x,result.velocity.y);
            if (speed>200) { result.velocity.x*=200/speed; result.velocity.y*=200/speed; }
        }
        previous_=mapped;
        return result;
    }
private:
    std::optional<Point> previous_;
    std::array<int,4> geometry_{};
    bool blocked_=false;
};
} // namespace fluidgl
