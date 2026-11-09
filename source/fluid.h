#pragma once
#include <glad/gl.h>
#include <cuda_runtime.h>
#include "sim_types.h"
#include <cstdint>
#include <vector>

struct cudaGraphicsResource;

class FluidSim {
public:
    static constexpr int MAX_GRID = 2048;
    explicit FluidSim(const SimConfig& cfg, const std::vector<float>& initDensity);
    ~FluidSim();
    FluidSim(const FluidSim&) = delete;
    FluidSim& operator=(const FluidSim&) = delete;
    FluidSim(FluidSim&&) noexcept;
    FluidSim& operator=(FluidSim&&) noexcept;

    // Borrow an allocated GL_TEXTURE_2D / GL_R32F matching the grid; 0 detaches.
    // The owning GL context must be current for attachment and destruction.
    void attachGLTexture(GLuint tex);
    void step();
    void publishToGL();
    void inject(const FluidInjection& injection);
    std::vector<float> readbackDensity() const; // Blocking diagnostics only.
    static std::vector<float> noiseDensity(const SimConfig&, std::uint32_t seed);
    SimParams& params() { return params_; }
    const SimParams& params() const { return params_; }
    const SimConfig& config() const { return config_; }
    int gridW() const { return config_.gridW; }
    int gridH() const { return config_.gridH; }
    // Clears simulation state and detaches GL texture; reattach a matching texture.
    void resize(int newW, int newH);
private:
    void release() noexcept;
    void allocate();
    void buildPressureGraph();
    void swapState(FluidSim&) noexcept;
    SimConfig config_;
    SimParams params_;
    // Packed MAC grid: .x = right face; .y = bottom face. Units: cells/second.
    float2 *velA_ = nullptr, *velB_ = nullptr, *velRhs_ = nullptr;
    float *denA_ = nullptr, *denB_ = nullptr;
    float *pA_ = nullptr, *pB_ = nullptr, *div_ = nullptr;
    cudaGraphExec_t pressureGraphExec_ = nullptr;
    float* pressureResult_ = nullptr; // Captured final output; pA_/pB_ stay fixed.
    int pressureGraphIterations_ = 0;
    std::vector<FluidInjection> injections_;
    GLuint tex_ = 0;
    cudaGraphicsResource* res_ = nullptr;
    bool mapped_ = false;
};
