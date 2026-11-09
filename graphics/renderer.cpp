#include "renderer.h"
#include "fluid.h"
#include <cuda_gl_interop.h>
#include <iostream>
#include <stdexcept>
#include <string>

namespace fluidgl {
namespace {
constexpr const char* vs=R"GLSL(#version 330 core
layout(location=0) in vec2 position;
layout(location=1) in vec2 texcoord;
out vec2 uv;
void main() { gl_Position=vec4(position,0,1); uv=texcoord; }
)GLSL";
constexpr const char* fs=R"GLSL(#version 330 core
in vec2 uv;
uniform sampler2D densityTexture;
uniform int useColor;
uniform float gain;
uniform float visualTime;
out vec4 color;
float density(vec2 p) {
    return 1.0-exp(-max(texture(densityTexture,p).r*gain,0.0)*0.9);
}
// Palette stops are display-space colors; convert once before lighting.
vec3 palette(float t) {
    vec3 c=mix(vec3(0.018,0.008,0.070),vec3(0.30,0.025,0.48),smoothstep(0.0,0.20,t));
    c=mix(c,vec3(0.40,0.055,0.92),smoothstep(0.17,0.38,t));
    c=mix(c,vec3(0.035,0.28,1.0),smoothstep(0.34,0.57,t));
    c=mix(c,vec3(0.015,0.82,0.98),smoothstep(0.53,0.77,t));
    c=mix(c,vec3(0.58,0.95,1.0),smoothstep(0.74,0.93,t));
    return mix(c,vec3(1.0,0.98,0.93),smoothstep(0.91,1.0,t));
}
float brightDensity(vec2 p) { return smoothstep(0.54,0.80,density(p)); }
void main() {
    if (useColor==0) {
        float d=clamp(texture(densityTexture,uv).r*gain,0.0,1.0);
        color=vec4(vec3(d),1.0);
        return;
    }
    vec2 texel=1.0/vec2(textureSize(densityTexture,0));
    float d=density(uv);
    float l=density(uv-vec2(texel.x,0)),r=density(uv+vec2(texel.x,0));
    float t=density(uv-vec2(0,texel.y)),b=density(uv+vec2(0,texel.y));
    vec2 gradient=0.5*vec2(r-l,b-t);
    float coverage=smoothstep(0.015,0.70,d);
    float drift=0.014*sin(d*9.0+visualTime*0.25);
    float tone=clamp(d*1.55+drift*coverage,0.0,1.0);
    vec3 base=pow(palette(tone),vec3(2.2));
    vec3 normal=normalize(vec3(-gradient.x*9.0,gradient.y*9.0,1.0));
    vec3 light=normalize(vec3(-0.45,0.60,0.85));
    float diffuse=max(dot(normal,light),0.0);
    float specular=pow(max(dot(normal,normalize(light+vec3(0,0,1))),0.0),48.0);
    // Keep low density dark and confine neutral highlights to dense cores.
    float brightness=pow(coverage,1.35);
    vec3 lit=base*(0.78+0.22*diffuse)*brightness*1.35;
    float rim=1.0-exp(-length(gradient)*8.0);
    lit+=base*rim*0.10*brightness;
    lit+=vec3(1.0,0.98,0.94)*specular*0.16*smoothstep(0.68,0.97,tone)*brightness;
    // Only spill light out of bright regions; never overlay a purple wash.
    vec2 halo=texel*2.5;
    float nearby=0.25*(brightDensity(uv+vec2(halo.x,0))+brightDensity(uv-vec2(halo.x,0))
                      +brightDensity(uv+vec2(0,halo.y))+brightDensity(uv-vec2(0,halo.y)));
    float glow=max(nearby-brightDensity(uv),0.0);
    lit+=vec3(0.30,0.75,1.0)*glow*0.10;
    // A shared shoulder preserves RGB ratios instead of bleaching every channel.
    float peak=max(lit.r,max(lit.g,lit.b));
    vec3 mapped=lit/(1.0+max(peak-0.85,0.0));
    color=vec4(pow(clamp(mapped,0.0,1.0),vec3(1.0/2.2)),1.0);
}

)GLSL";
GLuint compile(GLenum type,const char* source) {
    GLuint shader=glCreateShader(type);
    if (!shader) throw std::runtime_error("glCreateShader failed");
    glShaderSource(shader,1,&source,nullptr); glCompileShader(shader);
    GLint ok=0,n=0;
    glGetShaderiv(shader,GL_COMPILE_STATUS,&ok);
    glGetShaderiv(shader,GL_INFO_LOG_LENGTH,&n);
    if (n>1) {
        std::vector<char> log(n); glGetShaderInfoLog(shader,n,nullptr,log.data());
        std::cerr << "Shader log: " << log.data() << '\n';
    }
    if (!ok) { glDeleteShader(shader); throw std::runtime_error("Shader compilation failed"); }
    return shader;
}
GLuint link_program() {
    GLuint vertex=0,fragment=0,program=0;
    try {
        vertex=compile(GL_VERTEX_SHADER,vs); fragment=compile(GL_FRAGMENT_SHADER,fs);
        program=glCreateProgram();
        if (!program) throw std::runtime_error("glCreateProgram failed");
        glAttachShader(program,vertex); glAttachShader(program,fragment); glLinkProgram(program);
        GLint ok=0,n=0;
        glGetProgramiv(program,GL_LINK_STATUS,&ok); glGetProgramiv(program,GL_INFO_LOG_LENGTH,&n);
        if (n>1) {
            std::vector<char> log(n); glGetProgramInfoLog(program,n,nullptr,log.data());
            std::cerr << "Program log: " << log.data() << '\n';
        }
        if (!ok) throw std::runtime_error("Program linking failed");
        glDetachShader(program,vertex); glDetachShader(program,fragment);
        glDeleteShader(vertex); glDeleteShader(fragment);
        return program;
    } catch (...) {
        if (program) glDeleteProgram(program);
        if (vertex) glDeleteShader(vertex);
        if (fragment) glDeleteShader(fragment);
        throw;
    }
}
void check_gl(const char* stage) {
    if (glGetError()!=GL_NO_ERROR) throw std::runtime_error(stage);
}
void check_cuda(cudaError_t error,const char* stage) {
    if (error!=cudaSuccess)
        throw std::runtime_error(std::string(stage)+": "+cudaGetErrorString(error));
}

}
void Renderer::select_cuda_device() {
    unsigned count=0;
    int devices[8]{};
    check_cuda(cudaGLGetDevices(&count,devices,8,cudaGLDeviceListAll),
               "CUDA/OpenGL device query failed; use an NVIDIA OpenGL GPU");
    if (count!=1)
        throw std::runtime_error("CUDA/OpenGL renderer requires exactly one associated CUDA GPU");
    check_cuda(cudaSetDevice(devices[0]),"CUDA/OpenGL device selection failed");
}
Renderer::Renderer(int width,int height):width_(width),height_(height) {
    if (width<=0 || height<=0) throw std::invalid_argument("Invalid density dimensions");
    try {
        program_=link_program();
        // x,y,u,v: first density row is displayed at the TOP, matching input mapping.
        constexpr float quad[]={-1,-1,0,1, 1,1,1,0, -1,1,0,0,
                                -1,-1,0,1, 1,-1,1,1, 1,1,1,0};
        glGenVertexArrays(1,&vao_); glBindVertexArray(vao_);
        glGenBuffers(1,&vbo_); glBindBuffer(GL_ARRAY_BUFFER,vbo_);
        glBufferData(GL_ARRAY_BUFFER,sizeof(quad),quad,GL_STATIC_DRAW);
        glVertexAttribPointer(0,2,GL_FLOAT,GL_FALSE,4*sizeof(float),nullptr);
        glVertexAttribPointer(1,2,GL_FLOAT,GL_FALSE,4*sizeof(float),reinterpret_cast<void*>(2*sizeof(float)));
        glEnableVertexAttribArray(0); glEnableVertexAttribArray(1);
        glActiveTexture(GL_TEXTURE0); glGenTextures(1,&texture_);
        glBindTexture(GL_TEXTURE_2D,texture_);
        glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_S,GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_T,GL_CLAMP_TO_EDGE);
        glPixelStorei(GL_UNPACK_ALIGNMENT,1);
        // Allocate once. FluidSim registers and updates the borrowed texture.
        glTexImage2D(GL_TEXTURE_2D,0,GL_R32F,width,height,0,GL_RED,GL_FLOAT,nullptr);
        check_gl("Density texture allocation failed");
        glUseProgram(program_);
        const GLint sampler=glGetUniformLocation(program_,"densityTexture");
        color_=glGetUniformLocation(program_,"useColor"); gain_=glGetUniformLocation(program_,"gain");
        time_=glGetUniformLocation(program_,"visualTime");
        if (sampler<0 || color_<0 || gain_<0 || time_<0) throw std::runtime_error("Field uniforms missing");
        glUniform1i(sampler,0);
        check_gl("Renderer initialization failed");
    } catch (...) { release(); throw; }
}
void Renderer::release() {
    // Application destroys FluidSim (unregister) before deleting GL resources.
    if (texture_) glDeleteTextures(1,&texture_);
    if (vbo_) glDeleteBuffers(1,&vbo_);
    if (vao_) glDeleteVertexArrays(1,&vao_);
    if (program_) glDeleteProgram(program_);
}
Renderer::~Renderer() { release(); }
void Renderer::update_density(FluidSim& fluid) {
    fluid.publishToGL();
    ++uploads_;
}
Canvas Renderer::draw(int fw,int fh,bool color,float gain,float visual_time) {
    const Canvas canvas=fit_canvas(fw,fh,width_,height_);
    if (canvas.width<=0) return canvas;
    glDisable(GL_SCISSOR_TEST); glDisable(GL_DEPTH_TEST); glDisable(GL_BLEND);
    glViewport(0,0,fw,fh);
    glClearColor(0.035f,0.045f,0.06f,1); glClear(GL_COLOR_BUFFER_BIT);
    glViewport(canvas.x,canvas.y,canvas.width,canvas.height);
    glUseProgram(program_); glUniform1i(color_,color ? 1 : 0); glUniform1f(gain_,gain);
    glUniform1f(time_,visual_time);
    glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D,texture_);
    glBindVertexArray(vao_); glDrawArrays(GL_TRIANGLES,0,6);
    glActiveTexture(GL_TEXTURE0);
    check_gl("Density draw failed"); return canvas;
}
std::vector<unsigned char> Renderer::read_rgb(int fw,int fh) const {
    std::vector<unsigned char> pixels(static_cast<std::size_t>(fw)*fh*3);
    glReadBuffer(GL_BACK); glPixelStorei(GL_PACK_ALIGNMENT,1);
    glReadPixels(0,0,fw,fh,GL_RGB,GL_UNSIGNED_BYTE,pixels.data());
    check_gl("Framebuffer readback failed"); return pixels;
}
} // namespace fluidgl
