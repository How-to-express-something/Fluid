#include "fluid_kernel.h"
#include <cmath>

namespace fluidkernels {
namespace {
__device__ int index(int x, int y, SimParams p) {
    return max(0,min(y,p.gridH-1))*p.gridW+max(0,min(x,p.gridW-1));
}
__device__ float sample(const float* f,int x,int y,SimParams p) { return f[index(x,y,p)]; }
__device__ float face(const float2* f,int x,int y,int c,SimParams p) {
    // Closed normal faces; clamped tangential values implement free-slip walls.
    if ((c==0 && (x<0 || x>=p.gridW-1)) || (c==1 && (y<0 || y>=p.gridH-1))) return 0;
    float2 v=f[index(x,y,p)]; return c==0 ? v.x : v.y;
}
__device__ float lerp(float a,float b,float t) { return a+(b-a)*t; }
__device__ float bilerp(const float* f,float x,float y,SimParams p) {
    x=fminf(fmaxf(x,0),float(p.gridW-1)); y=fminf(fmaxf(y,0),float(p.gridH-1));
    int ix=int(floorf(x)),iy=int(floorf(y)); float u=x-ix,v=y-iy;
    return lerp(lerp(sample(f,ix,iy,p),sample(f,ix+1,iy,p),u),
                lerp(sample(f,ix,iy+1,p),sample(f,ix+1,iy+1,p),u),v);
}
__device__ float bilerp(const float2* f,float x,float y,int c,SimParams p) {
    x=fminf(fmaxf(x,-1),float(p.gridW)); y=fminf(fmaxf(y,-1),float(p.gridH));
    int ix=int(floorf(x)),iy=int(floorf(y)); float u=x-ix,v=y-iy;
    return lerp(lerp(face(f,ix,iy,c,p),face(f,ix+1,iy,c,p),u),
                lerp(face(f,ix,iy+1,c,p),face(f,ix+1,iy+1,c,p),u),v);
}
// Cell centers have integer coordinates. u lives at (x+0.5,y), v at (x,y+0.5).
__device__ float2 velocity(const float2* f,float x,float y,SimParams p) {
    return make_float2(bilerp(f,x-0.5f,y,0,p),bilerp(f,x,y-0.5f,1,p));
}
__device__ float weight(float x,float y,FluidInjection f,SimParams p) {
    float r=p.forceRadius*min(p.gridW,p.gridH),dx=x-f.x,dy=y-f.y;
    return expf(-(dx*dx+dy*dy)/(2*r*r));
}
__global__ void addForce(float2* vel,float* density,SimParams p,FluidInjection f,bool brush) {
    int x=blockIdx.x*blockDim.x+threadIdx.x,y=blockIdx.y*blockDim.y+threadIdx.y;
    if (x>=p.gridW || y>=p.gridH) return;
    int i=y*p.gridW+x; float2 v=vel[i];
    if (brush) {
        density[i]+=p.dt*f.density*weight(float(x),float(y),f,p);
        v.x+=p.dt*p.forceStrength*f.vx*weight(x+0.5f,float(y),f,p);
        v.y+=p.dt*p.forceStrength*f.vy*weight(float(x),y+0.5f,f,p);
    } else {
        v.y-=p.dt*p.buoyancy*0.5f*(density[i]+sample(density,x,y+1,p));
    }
    if (x==p.gridW-1) v.x=0;
    if (y==p.gridH-1) v.y=0;
    vel[i]=v;
}
__global__ void advectVector2(const float2* src,float2* dst,SimParams p) {
    int x=blockIdx.x*blockDim.x+threadIdx.x,y=blockIdx.y*blockDim.y+threadIdx.y;
    if (x>=p.gridW || y>=p.gridH) return;
    float2 u=velocity(src,x+0.5f,float(y),p),v=velocity(src,float(x),y+0.5f,p);
    dst[y*p.gridW+x]=make_float2(x==p.gridW-1 ? 0 : p.velDissip*bilerp(src,x-p.dt*u.x,y-p.dt*u.y,0,p),
        y==p.gridH-1 ? 0 : p.velDissip*bilerp(src,x-p.dt*v.x,y-p.dt*v.y,1,p));
}
__global__ void diffuse(const float2* src,const float2* rhs,float2* dst,SimParams p) {
    int x=blockIdx.x*blockDim.x+threadIdx.x,y=blockIdx.y*blockDim.y+threadIdx.y;
    if (x>=p.gridW || y>=p.gridH) return;
    int i=y*p.gridW+x; float a=p.dt*p.viscosity; float2 v;
    v.x=(rhs[i].x+a*(face(src,x-1,y,0,p)+face(src,x+1,y,0,p)+face(src,x,y-1,0,p)+face(src,x,y+1,0,p)))/(1+4*a);
    v.y=(rhs[i].y+a*(face(src,x-1,y,1,p)+face(src,x+1,y,1,p)+face(src,x,y-1,1,p)+face(src,x,y+1,1,p)))/(1+4*a);
    if (x==p.gridW-1) v.x=0;
    if (y==p.gridH-1) v.y=0;
    dst[i]=v;
}
__global__ void divergenceKernel(const float2* vel,float* div,SimParams p) {
    int x=blockIdx.x*blockDim.x+threadIdx.x,y=blockIdx.y*blockDim.y+threadIdx.y;
    if (x>=p.gridW || y>=p.gridH) return;
    div[y*p.gridW+x]=face(vel,x,y,0,p)-face(vel,x-1,y,0,p)
        +face(vel,x,y,1,p)-face(vel,x,y-1,1,p);
}
__global__ void jacobi(const float* src,const float* div,float* dst,SimParams p) {
    int x=blockIdx.x*blockDim.x+threadIdx.x,y=blockIdx.y*blockDim.y+threadIdx.y;
    if (x>=p.gridW || y>=p.gridH) return;
    // Neumann pressure boundaries match divergence and gradientSubtract.
    dst[y*p.gridW+x]=0.25f*(sample(src,x-1,y,p)+sample(src,x+1,y,p)
        +sample(src,x,y-1,p)+sample(src,x,y+1,p)-div[y*p.gridW+x]);
}
__global__ void gradientSubtract(const float2* src,const float* pressure,float2* dst,SimParams p) {
    int x=blockIdx.x*blockDim.x+threadIdx.x,y=blockIdx.y*blockDim.y+threadIdx.y;
    if (x>=p.gridW || y>=p.gridH) return;
    int i=y*p.gridW+x;
    dst[i]=make_float2(x==p.gridW-1 ? 0 : src[i].x-(sample(pressure,x+1,y,p)-pressure[i]),
        y==p.gridH-1 ? 0 : src[i].y-(sample(pressure,x,y+1,p)-pressure[i]));
}
__global__ void advectScalar(const float* src,const float2* vel,float* dst,SimParams p) {
    int x=blockIdx.x*blockDim.x+threadIdx.x,y=blockIdx.y*blockDim.y+threadIdx.y;
    if (x>=p.gridW || y>=p.gridH) return;
    float2 v=velocity(vel,float(x),float(y),p);
    dst[y*p.gridW+x]=p.denDissip*bilerp(src,x-p.dt*v.x,y-p.dt*v.y,p);
}
dim3 grid(SimParams p) { return dim3((p.gridW+15)/16,(p.gridH+15)/16); }
}
void addForces(float2* v,float* d,SimParams p,FluidInjection f,bool brush) { addForce<<<grid(p),dim3(16,16)>>>(v,d,p,f,brush); }
void advectVelocity(const float2* a,float2* b,SimParams p) { advectVector2<<<grid(p),dim3(16,16)>>>(a,b,p); }
void diffuseVelocity(const float2* a,const float2* rhs,float2* b,SimParams p) { diffuse<<<grid(p),dim3(16,16)>>>(a,rhs,b,p); }
void divergence(const float2* v,float* d,SimParams p) { divergenceKernel<<<grid(p),dim3(16,16)>>>(v,d,p); }
void pressure(const float* a,const float* d,float* b,SimParams p,cudaStream_t stream) { jacobi<<<grid(p),dim3(16,16),0,stream>>>(a,d,b,p); }
void project(const float2* a,const float* pressure,float2* b,SimParams p) { gradientSubtract<<<grid(p),dim3(16,16)>>>(a,pressure,b,p); }
void advectDensity(const float* a,const float2* v,float* b,SimParams p) { advectScalar<<<grid(p),dim3(16,16)>>>(a,v,b,p); }
}
