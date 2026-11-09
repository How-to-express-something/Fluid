#pragma once
#include <cuda_runtime.h>
#include "sim_types.h"

// Host launch wrappers. Stream 0 by default; pressure also supports graph capture.
namespace fluidkernels {
void addForces(float2* velocity, float* density, SimParams, FluidInjection, bool brush);
void advectVelocity(const float2* src, float2* dst, SimParams);
void diffuseVelocity(const float2* src, const float2* rhs, float2* dst, SimParams);
void divergence(const float2* velocity, float* div, SimParams);
void pressure(const float* src, const float* div, float* dst, SimParams, cudaStream_t stream = 0);
void project(const float2* src, const float* pressure, float2* dst, SimParams);
void advectDensity(const float* src, const float2* velocity, float* dst, SimParams);
}
