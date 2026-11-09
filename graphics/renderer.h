#pragma once
#include <glad/gl.h>
#include "canvas.h"
#include <vector>

class FluidSim;

namespace fluidgl {
// Owns GL resources. FluidSim borrows/registers the texture.
class Renderer {
public:
    // Call with a current GL context, BEFORE allocating any Fluid CUDA memory.
    static void select_cuda_device();
    Renderer(int width,int height);
    ~Renderer();
    Renderer(const Renderer&)=delete;
    Renderer& operator=(const Renderer&)=delete;
    void update_density(FluidSim& fluid);
    GLuint texture() const { return texture_; }
    Canvas draw(int framebuffer_width,int framebuffer_height,bool color=true,float gain=0.2f,
                float visual_time=0.0f);
    std::vector<unsigned char> read_rgb(int fw,int fh) const;
    unsigned updates() const { return uploads_; }
private:
    void release();
    int width_,height_;
    GLuint program_=0,vao_=0,vbo_=0,texture_=0;
    GLint color_=-1,gain_=-1,time_=-1;
    unsigned uploads_=0;
};
} // namespace fluidgl
