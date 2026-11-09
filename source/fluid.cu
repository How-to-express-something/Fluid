#include "fluid.h"
#include "fluid_kernel.h"
#include <cuda_gl_interop.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>

namespace {
void check(cudaError_t error) {
    if (error!=cudaSuccess) throw std::runtime_error(std::string("CUDA: ")+cudaGetErrorString(error));
}
void cleanup(cudaError_t error) noexcept {
    if (error!=cudaSuccess) std::fprintf(stderr,"CUDA cleanup: %s\n",cudaGetErrorString(error));
}
void validateGrid(SimConfig c) {
    if (c.gridW<2 || c.gridH<2 || c.gridW>FluidSim::MAX_GRID || c.gridH>FluidSim::MAX_GRID)
        throw std::invalid_argument("Grid dimensions must be in [2, 2048]");
}
void validateParams(SimParams p,SimConfig c) {
    if (p.gridW!=c.gridW || p.gridH!=c.gridH) throw std::invalid_argument("Use resize() to change grid dimensions");
    if (p.jacobiIter<1 || p.jacobiIter>10000 || p.diffuseIter<1 || p.diffuseIter>10000)
        throw std::invalid_argument("Iteration counts must be in [1, 10000]");
    const float values[]={p.dt,p.velDissip,p.denDissip,p.viscosity,p.forceRadius,p.forceStrength,p.buoyancy};
    for (float v:values) if (!std::isfinite(v)) throw std::invalid_argument("Non-finite simulation parameter");
    if (p.dt<=0 || p.velDissip<0 || p.velDissip>1 || p.denDissip<0 || p.denDissip>1
        || p.viscosity<0 || p.forceRadius<=0 || p.forceRadius>1 || p.forceStrength<0)
        throw std::invalid_argument("Invalid simulation parameter range");
}
}
FluidSim::FluidSim(const SimConfig& cfg,const std::vector<float>& density):config_(cfg) {
    validateGrid(cfg);
    if (density.size()!=static_cast<std::size_t>(cfg.gridW)*cfg.gridH)
        throw std::invalid_argument("Initial density size does not match grid");
    for (float v:density) if (!std::isfinite(v) || v<0) throw std::invalid_argument("Invalid initial density");
    params_.gridW=cfg.gridW; params_.gridH=cfg.gridH;
    try {
        allocate();
        check(cudaMemcpy(denA_,density.data(),density.size()*sizeof(float),cudaMemcpyHostToDevice));
        buildPressureGraph();
    } catch (...) { release(); throw; }
}
void FluidSim::allocate() {
    const auto bytes=static_cast<std::size_t>(gridW())*gridH()*sizeof(float);
    check(cudaMalloc(&velA_,2*bytes)); check(cudaMalloc(&velB_,2*bytes)); check(cudaMalloc(&velRhs_,2*bytes));
    check(cudaMalloc(&denA_,bytes)); check(cudaMalloc(&denB_,bytes));
    check(cudaMalloc(&pA_,bytes)); check(cudaMalloc(&pB_,bytes)); check(cudaMalloc(&div_,bytes));
    check(cudaMemset(velA_,0,2*bytes)); check(cudaMemset(velB_,0,2*bytes)); check(cudaMemset(velRhs_,0,2*bytes));
    check(cudaMemset(denA_,0,bytes)); check(cudaMemset(denB_,0,bytes));
    check(cudaMemset(pA_,0,bytes)); check(cudaMemset(pB_,0,bytes)); check(cudaMemset(div_,0,bytes));
}
FluidSim::~FluidSim() { release(); }
void FluidSim::buildPressureGraph() {
    // Only grid size and iteration count affect jacobi. Resize replaces all resources.
    // Keep the old executable valid until the replacement is fully constructed.
    cudaStream_t stream=nullptr;
    cudaGraph_t graph=nullptr;
    cudaGraphExec_t exec=nullptr;
    bool capturing=false;
    float* src=pA_;
    float* dst=pB_;
    try {
        check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
        check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
        capturing=true;
        for (int i=0;i<params_.jacobiIter;++i) {
            fluidkernels::pressure(src,div_,dst,params_,stream);
            check(cudaGetLastError());
            std::swap(src,dst); // CPU recording logic; replay does not execute this swap.
        }
        const auto endError=cudaStreamEndCapture(stream,&graph);
        capturing=false;
        check(endError);
        check(cudaGraphInstantiate(&exec,graph,nullptr,nullptr,0));
        // All graph launches use stream 0. Rebuilding is exceptional, not per-frame.
        if (pressureGraphExec_) check(cudaStreamSynchronize(0));
    } catch (...) {
        if (capturing) cleanup(cudaStreamEndCapture(stream,&graph));
        if (exec) cleanup(cudaGraphExecDestroy(exec));
        if (graph) cleanup(cudaGraphDestroy(graph));
        if (stream) cleanup(cudaStreamDestroy(stream));
        throw;
    }
    if (pressureGraphExec_) cleanup(cudaGraphExecDestroy(pressureGraphExec_));
    pressureGraphExec_=exec;
    pressureResult_=src;
    pressureGraphIterations_=params_.jacobiIter;
    cleanup(cudaGraphDestroy(graph)); // Executable retains its own instantiated nodes.
    cleanup(cudaStreamDestroy(stream)); // Capture stream is not needed for replay.
}
void FluidSim::release() noexcept {
    if (mapped_) { cleanup(cudaGraphicsUnmapResources(1,&res_,0)); mapped_=false; }
    if (res_) { cleanup(cudaGraphicsUnregisterResource(res_)); res_=nullptr; }
    tex_=0;
    if (pressureGraphExec_) {
        cleanup(cudaStreamSynchronize(0));
        cleanup(cudaGraphExecDestroy(pressureGraphExec_));
        pressureGraphExec_=nullptr;
    }
    pressureResult_=nullptr; pressureGraphIterations_=0;
    for (void* p:{static_cast<void*>(velA_),static_cast<void*>(velB_),static_cast<void*>(velRhs_),
        static_cast<void*>(denA_),static_cast<void*>(denB_),static_cast<void*>(pA_),static_cast<void*>(pB_),static_cast<void*>(div_)})
        if (p) cleanup(cudaFree(p));
    velA_=velB_=velRhs_=nullptr; denA_=denB_=pA_=pB_=div_=nullptr;
    injections_.clear();
}
void FluidSim::swapState(FluidSim& other) noexcept {
    using std::swap;
    swap(config_,other.config_); swap(params_,other.params_);
    swap(velA_,other.velA_); swap(velB_,other.velB_); swap(velRhs_,other.velRhs_);
    swap(denA_,other.denA_); swap(denB_,other.denB_); swap(pA_,other.pA_); swap(pB_,other.pB_); swap(div_,other.div_);
    swap(pressureGraphExec_,other.pressureGraphExec_);
    swap(pressureResult_,other.pressureResult_); swap(pressureGraphIterations_,other.pressureGraphIterations_);
    swap(injections_,other.injections_); swap(tex_,other.tex_); swap(res_,other.res_); swap(mapped_,other.mapped_);
}
FluidSim::FluidSim(FluidSim&& other) noexcept { swapState(other); }
FluidSim& FluidSim::operator=(FluidSim&& other) noexcept {
    if (this!=&other) { release(); swapState(other); } return *this;
}
void FluidSim::attachGLTexture(GLuint tex) {
    if (mapped_) throw std::logic_error("Texture still mapped after a CUDA failure");
    if (tex==tex_) return;
    if (!tex) {
        if (res_) check(cudaGraphicsUnregisterResource(res_));
        res_=nullptr; tex_=0; return;
    }
    if (!glIsTexture(tex)) throw std::invalid_argument("Not an OpenGL texture");
    GLint previous=0,w=0,h=0,format=0;
    glGetIntegerv(GL_TEXTURE_BINDING_2D,&previous);
    glBindTexture(GL_TEXTURE_2D,tex);
    glGetTexLevelParameteriv(GL_TEXTURE_2D,0,GL_TEXTURE_WIDTH,&w);
    glGetTexLevelParameteriv(GL_TEXTURE_2D,0,GL_TEXTURE_HEIGHT,&h);
    glGetTexLevelParameteriv(GL_TEXTURE_2D,0,GL_TEXTURE_INTERNAL_FORMAT,&format);
    glBindTexture(GL_TEXTURE_2D,static_cast<GLuint>(previous));
    if (glGetError()!=GL_NO_ERROR || w!=gridW() || h!=gridH() || format!=GL_R32F)
        throw std::invalid_argument("Expected matching GL_TEXTURE_2D / GL_R32F texture");
    cudaGraphicsResource* next=nullptr;
    check(cudaGraphicsGLRegisterImage(&next,tex,GL_TEXTURE_2D,cudaGraphicsRegisterFlagsWriteDiscard));
    if (res_) {
        const auto error=cudaGraphicsUnregisterResource(res_);
        if (error!=cudaSuccess) { cleanup(cudaGraphicsUnregisterResource(next)); check(error); }
    }
    res_=next; tex_=tex;
}
void FluidSim::publishToGL() {
    if (!res_) throw std::logic_error("Attach an OpenGL texture before publishToGL()");
    if (mapped_) throw std::logic_error("Texture still mapped after a CUDA failure");
    check(cudaGraphicsMapResources(1,&res_,0)); mapped_=true;
    try {
        // This is a view of the GL image, not an additional CUDA allocation.
        cudaArray_t array=nullptr;
        check(cudaGraphicsSubResourceGetMappedArray(&array,res_,0,0));
        const auto pitch=static_cast<std::size_t>(gridW())*sizeof(float);
        check(cudaMemcpy2DToArrayAsync(array,0,0,denA_,pitch,pitch,gridH(),cudaMemcpyDeviceToDevice,0));
    } catch (...) {
        const auto error=cudaGraphicsUnmapResources(1,&res_,0);
        if (error==cudaSuccess) mapped_=false;
        cleanup(error); throw;
    }
    check(cudaGraphicsUnmapResources(1,&res_,0)); mapped_=false;
}
void FluidSim::inject(const FluidInjection& f) {
    if (!velA_) throw std::logic_error("Cannot inject a moved-from FluidSim");
    const float values[]={f.x,f.y,f.vx,f.vy,f.density};
    for (float v:values) if (!std::isfinite(v)) throw std::invalid_argument("Non-finite injection");
    if (f.density<0) throw std::invalid_argument("Negative injection density");
    if (injections_.size()>=64) throw std::length_error("Too many injections before step()");
    injections_.push_back(f);
}
void FluidSim::step() {
    if (!velA_) throw std::logic_error("Cannot step a moved-from FluidSim");
    validateParams(params_,config_);
    const SimParams p=params_; // One consistent snapshot for the whole step.

    const auto bytes=static_cast<std::size_t>(gridW())*gridH()*sizeof(float);
    if (pressureGraphIterations_!=p.jacobiIter) buildPressureGraph();
    using namespace fluidkernels;

    for (const auto& f:injections_) { addForces(velA_,denA_,p,f,true); check(cudaGetLastError()); }
    injections_.clear();
    addForces(velA_,denA_,p,{},false); check(cudaGetLastError());
    advectVelocity(velA_,velB_,p); check(cudaGetLastError());
    std::swap(velA_,velB_); // A always denotes the latest result.
    if (p.viscosity>0) {
        check(cudaMemcpyAsync(velRhs_,velA_,2*bytes,cudaMemcpyDeviceToDevice,0));
        for (int i=0;i<p.diffuseIter;++i) {
            diffuseVelocity(velA_,velRhs_,velB_,p); check(cudaGetLastError());
            std::swap(velA_,velB_);
        }
    }
    divergence(velA_,div_,p); check(cudaGetLastError());
    check(cudaMemsetAsync(pA_,0,bytes,0)); check(cudaMemsetAsync(pB_,0,bytes,0));
    check(cudaGraphLaunch(pressureGraphExec_,0));

    project(velA_,pressureResult_,velB_,p); check(cudaGetLastError());
    std::swap(velA_,velB_);
    advectDensity(denA_,velA_,denB_,p); check(cudaGetLastError());
    std::swap(denA_,denB_);
}
std::vector<float> FluidSim::readbackDensity() const {
    if (!denA_) throw std::logic_error("Cannot read a moved-from FluidSim");
    std::vector<float> result(static_cast<std::size_t>(gridW())*gridH());
    check(cudaMemcpy(result.data(),denA_,result.size()*sizeof(float),cudaMemcpyDeviceToHost));
    return result;
}
void FluidSim::resize(int w,int h) {
    validateGrid({w,h});
    FluidSim next({w,h},std::vector<float>(static_cast<std::size_t>(w)*h,0));
    next.params_=params_; next.params_.gridW=w; next.params_.gridH=h;
    validateParams(next.params_,next.config_);
    if (next.pressureGraphIterations_!=next.params_.jacobiIter) next.buildPressureGraph();
    attachGLTexture(0);
    swapState(next);
}
std::vector<float> FluidSim::noiseDensity(const SimConfig& c,std::uint32_t seed) {
    validateGrid(c);
    std::mt19937 rng(seed);
    std::vector<float> result(static_cast<std::size_t>(c.gridW)*c.gridH,0);
    for (int octave=0;octave<3;++octave) {
        const int n=4<<octave;
        std::vector<float> lattice((n+1)*(n+1));
        for (float& v:lattice) v=static_cast<float>(rng()>>8)*(1.0f/16777216.0f);
        for (int y=0;y<c.gridH;++y) for (int x=0;x<c.gridW;++x) {
            float px=float(x)*n/c.gridW,py=float(y)*n/c.gridH;
            int ix=int(px),iy=int(py); float u=px-ix,v=py-iy;
            u=u*u*(3-2*u); v=v*v*(3-2*v);
            auto lerp=[](float a,float b,float t){return a+(b-a)*t;};
            result[y*c.gridW+x]+=std::ldexp(1.0f,-octave)*lerp(
                lerp(lattice[iy*(n+1)+ix],lattice[iy*(n+1)+ix+1],u),
                lerp(lattice[(iy+1)*(n+1)+ix],lattice[(iy+1)*(n+1)+ix+1],u),v);
        }
    }
    for (float& v:result) v=5.0f*std::clamp((v/1.75f-0.25f)*2.0f,0.0f,1.0f);
    return result;
}
