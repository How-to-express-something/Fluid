#pragma once
#include <algorithm>
#include <cmath>
#include <optional>

namespace fluidgl {
struct Point { double x=0, y=0; };
// OpenGL viewport: framebuffer pixels, origin at bottom left.
struct Canvas { int x=0,y=0,width=0,height=0; };
inline Canvas fit_canvas(int fw,int fh,int gw,int gh) {
    if (fw<=0 || fh<=0 || gw<=0 || gh<=0) return {};
    int w=fw,h=fh;
    if (static_cast<double>(fw)/fh > static_cast<double>(gw)/gh)
        w=std::max(1,static_cast<int>(static_cast<double>(fh)*gw/gh));
    else h=std::max(1,static_cast<int>(static_cast<double>(fw)*gh/gw));
    return {(fw-w)/2,(fh-h)/2,w,h};
}
// GLFW cursor: window units, origin at top left. Grid row 0 is also at TOP.
inline std::optional<Point> cursor_to_grid(Point cursor,int ww,int wh,int fw,int fh,
                                          Canvas c,int gw,int gh) {
    if (ww<=0 || wh<=0 || fw<=0 || fh<=0 || c.width<=0 || c.height<=0) return {};
    const double px=cursor.x*fw/ww, py=cursor.y*fh/wh;
    const double top=fh-(c.y+c.height);
    const double u=(px-c.x)/c.width, v=(py-top)/c.height;
    if (!std::isfinite(u) || !std::isfinite(v) || u<0 || u>=1 || v<0 || v>=1) return {};
    return Point{u*gw,v*gh};
}
struct FixedStep {
    static constexpr double dt=1.0/60.0;
    double accumulator=0, dropped_seconds=0;
    template<class Step> int advance(double elapsed,bool paused,bool single,Step step) {
        if (paused) {
            accumulator=0;
            if (single) { step(dt); return 1; }
            return 0;
        }
        elapsed=std::max(0.0,elapsed);
        dropped_seconds+=std::max(0.0,elapsed-0.25);
        accumulator+=std::min(elapsed,0.25);
        int count=0;
        while (accumulator+1e-12>=dt && count<8) {
            step(dt); accumulator-=dt; ++count;
        }
        if (accumulator>=dt) {
            const double keep=std::fmod(accumulator,dt);
            dropped_seconds+=accumulator-keep;
            accumulator=keep;
        }
        accumulator=std::max(0.0,accumulator);
        return count;
    }
};
} // namespace fluidgl
