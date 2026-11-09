#include "fluid.h"
#include "fluid_kernel.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <utility>

void require(bool ok,const char* message) { if (!ok) throw std::runtime_error(message); }
void check(cudaError_t e) { if (e!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
template<class T> struct Buffer {
    T* p=nullptr; std::size_t count;
    explicit Buffer(std::size_t n):count(n) { check(cudaMalloc(&p,n*sizeof(T))); check(cudaMemset(p,0,n*sizeof(T))); }
    ~Buffer() { cudaFree(p); }
    void upload(const std::vector<T>& v) { check(cudaMemcpy(p,v.data(),count*sizeof(T),cudaMemcpyHostToDevice)); }
    std::vector<T> read() const { std::vector<T> v(count); check(cudaMemcpy(v.data(),p,count*sizeof(T),cudaMemcpyDeviceToHost)); return v; }
};
void numerics(int w,int h) {
    SimParams s; s.gridW=w; s.gridH=h; s.viscosity=3; s.dt=0.1f;
    const int n=w*h;
    Buffer<float2> a(n),b(n),rhs(n); Buffer<float> div(n),pa(n),pb(n);
    std::vector<float2> v(n);
    for (int y=0;y<h;++y) for (int x=0;x<w;++x)
        v[y*w+x]=make_float2(x==w-1 ? 0 : std::sin(float(x*3+y)),y==h-1 ? 0 : std::cos(float(x+y*2)));
    a.upload(v); rhs.upload(v);
    fluidkernels::divergence(a.p,div.p,s);
    auto d=div.read(); double before=0; for (float f:d) before+=f*f;
    double sum=0; for (float f:d) sum+=f;
    require(std::abs(sum)<1e-4,"Closed-wall divergence must telescope to zero");
    for (int iter=0;iter<1001;++iter) { fluidkernels::pressure(pa.p,div.p,pb.p,s); std::swap(pa.p,pb.p); }
    fluidkernels::project(a.p,pa.p,b.p,s);
    fluidkernels::divergence(b.p,div.p,s);
    d=div.read(); double after=0; for (float f:d) after+=f*f;
    require(after<before*0.01,"Pressure projection failed to reduce divergence");
    auto projected=b.read();
    for (int y=0;y<h;++y) require(projected[y*w+w-1].x==0,"Right wall leaked");
    for (int x=0;x<w;++x) require(projected[(h-1)*w+x].y==0,"Bottom wall leaked");
    // Independent CPU Jacobi reference; odd/even swaps and immutable RHS both matter.
    std::vector<float2> cpu=v,next(n);
    auto at=[&](int x,int y,int c) {
        if ((c==0 && (x<0 || x>=w-1)) || (c==1 && (y<0 || y>=h-1))) return 0.0f;
        const auto q=cpu[std::clamp(y,0,h-1)*w+std::clamp(x,0,w-1)]; return c==0?q.x:q.y;
    };
    for (int iter=0;iter<5;++iter) {
        fluidkernels::diffuseVelocity(a.p,rhs.p,b.p,s); std::swap(a.p,b.p);
        const float alpha=s.dt*s.viscosity;
        for (int y=0;y<h;++y) for (int x=0;x<w;++x) {
            const int i=y*w+x;
            next[i].x=x==w-1 ? 0 : (v[i].x+alpha*(at(x-1,y,0)+at(x+1,y,0)+at(x,y-1,0)+at(x,y+1,0)))/(1+4*alpha);
            next[i].y=y==h-1 ? 0 : (v[i].y+alpha*(at(x-1,y,1)+at(x+1,y,1)+at(x,y-1,1)+at(x,y+1,1)))/(1+4*alpha);
        }
        cpu.swap(next);
        auto gpu=a.read();
        for (int i=0;i<n;++i) require(std::abs(cpu[i].x-gpu[i].x)<1e-5 && std::abs(cpu[i].y-gpu[i].y)<1e-5,"Diffusion differs from CPU reference");
    }
    check(cudaGetLastError()); check(cudaDeviceSynchronize());
    std::cout << "PASS: " << w << 'x' << h << " projection energy " << before << " -> " << after << ", diffusion CPU reference\n";
}

void advectionAndForces() {
    SimParams p; p.gridW=31; p.gridH=19; p.dt=0.2f; p.denDissip=1; p.velDissip=1;
    const int w=p.gridW,h=p.gridH,n=w*h;
    Buffer<float> src(n),dst(n); Buffer<float2> vel(n),out(n);
    std::vector<float> data(n);
    for (int y=0;y<h;++y) for (int x=0;x<w;++x) data[y*w+x]=0.1f*x+0.2f*y+0.003f*x*y;
    src.upload(data);
    fluidkernels::advectDensity(src.p,vel.p,dst.p,p);
    auto result=dst.read();
    for (int i=0;i<n;++i) require(std::abs(result[i]-data[i])<1e-6,"Zero velocity must preserve every density sample, including edges");
    std::vector<float2> field(n);
    for (int y=0;y<h;++y) for (int x=0;x<w;++x)
        field[y*w+x]=make_float2(x==w-1?0:0.1f*(x+0.5f)+0.03f*y,y==h-1?0:-0.02f*x+0.04f*(y+0.5f));
    vel.upload(field);
    fluidkernels::advectDensity(src.p,vel.p,dst.p,p); result=dst.read();
    // Analytic cell-center reconstruction of the linear face velocity field.
    for (int y=2;y<h-2;++y) for (int x=2;x<w-2;++x) {
        const float px=x-p.dt*(0.1f*x+0.03f*y),py=y-p.dt*(-0.02f*x+0.04f*y);
        const float expected=0.1f*px+0.2f*py+0.003f*px*py;
        require(std::abs(result[y*w+x]-expected)<2e-5,"Advection must use local velocity and correct cell-center coordinates");
    }
    // Large displacements must clamp safely, including at the last row/column.
    for (int y=0;y<h;++y) for (int x=0;x<w;++x)
        field[y*w+x]=make_float2(x==w-1?0:1e6f,y==h-1?0:-1e6f);
    vel.upload(field);
    fluidkernels::advectDensity(src.p,vel.p,dst.p,p);
    for (float d:dst.read()) require(std::isfinite(d) && d>=data.front()-1e-5 && d<=data.back()+1e-5,"Clamped advection escaped source range");
    fluidkernels::advectVelocity(vel.p,out.p,p);
    for (auto v:out.read()) require(std::isfinite(v.x) && std::isfinite(v.y),"Velocity boundary sampling is not finite");

    check(cudaMemset(vel.p,0,n*sizeof(float2))); check(cudaMemset(src.p,0,n*sizeof(float)));
    p.forceRadius=0.1f; p.forceStrength=3;
    const FluidInjection f{15,9,2,-1,10};
    fluidkernels::addForces(vel.p,src.p,p,f,true);
    auto density=src.read(); auto force=vel.read();
    const float radius=p.forceRadius*std::min(w,h);
    for (int y=0;y<h;++y) for (int x=0;x<w;++x) {
        const float dx=x-f.x,dy=y-f.y;
        const float expected=p.dt*f.density*std::exp(-(dx*dx+dy*dy)/(2*radius*radius));
        require(std::abs(density[y*w+x]-expected)<1e-5,"Brush radius/coordinate/dt does not match Gaussian");
    }
    const int center=9*w+15;
    require(force[center].x>0 && force[center].y<0,"Mouse force has wrong sign");
    const float expectedFx=p.dt*p.forceStrength*f.vx*std::exp(-0.25f/(2*radius*radius));
    require(std::abs(force[center].x-expectedFx)<1e-5,"forceStrength or face position ignored");
    check(cudaMemset(vel.p,0,n*sizeof(float2))); src.upload(std::vector<float>(n,2));
    p.buoyancy=4;
    fluidkernels::addForces(vel.p,src.p,p,{},false); force=vel.read();
    for (int y=0;y<h;++y) for (int x=0;x<w;++x)
        require(force[y*w+x].x==0 && std::abs(force[y*w+x].y-(y==h-1?0:-1.6f))<1e-5,"Buoyancy direction/dt or wall condition is wrong");
    check(cudaGetLastError()); check(cudaDeviceSynchronize());
    std::cout << "PASS: zero/local-velocity advection, edge clamping, Gaussian radius, force strength, buoyancy\n";
}

// Ordinary launches are an independent execution path for graph ordering/parity tests.
struct ReferenceSim {
    SimParams params;
    Buffer<float2> va,vb,rhs;
    Buffer<float> da,db,pa,pb,div;
    ReferenceSim(SimConfig c,const std::vector<float>& density)
        :va(c.gridW*c.gridH),vb(c.gridW*c.gridH),rhs(c.gridW*c.gridH),
         da(c.gridW*c.gridH),db(c.gridW*c.gridH),pa(c.gridW*c.gridH),pb(c.gridW*c.gridH),div(c.gridW*c.gridH) {
        params.gridW=c.gridW; params.gridH=c.gridH; da.upload(density);
    }
    void step(FluidInjection f) {
        using namespace fluidkernels;
        const auto bytes=da.count*sizeof(float);
        addForces(va.p,da.p,params,f,true);
        addForces(va.p,da.p,params,{},false);
        advectVelocity(va.p,vb.p,params); std::swap(va.p,vb.p);
        if (params.viscosity>0) {
            check(cudaMemcpyAsync(rhs.p,va.p,2*bytes,cudaMemcpyDeviceToDevice,0));
            for (int i=0;i<params.diffuseIter;++i) {
                diffuseVelocity(va.p,rhs.p,vb.p,params); std::swap(va.p,vb.p);
            }
        }
        divergence(va.p,div.p,params);
        check(cudaMemsetAsync(pa.p,0,bytes,0)); check(cudaMemsetAsync(pb.p,0,bytes,0));
        for (int i=0;i<params.jacobiIter;++i) {
            pressure(pa.p,div.p,pb.p,params); std::swap(pa.p,pb.p);
        }
        project(va.p,pa.p,vb.p,params); std::swap(va.p,vb.p);
        advectDensity(da.p,va.p,db.p,params); std::swap(da.p,db.p);
        check(cudaGetLastError());
    }
};
void compareGraphSteps(FluidSim& sim,ReferenceSim& reference) {
    // Repeated counts exercise cached replay, changes exercise rebuilds, both parities matter.
    const int counts[]={60,60,1,1,2,2,7,7,8,8,61,61};
    for (int frame=0;frame<12;++frame) {
        auto& p=sim.params();
        p.jacobiIter=counts[frame]; p.dt=0.08f; p.forceStrength=45;
        p.viscosity=frame<6 ? 0.0f : 0.7f; p.diffuseIter=3;
        p.buoyancy=0.4f; p.denDissip=0.97f;
        const FluidInjection f{0.35f*p.gridW+0.13f*frame,0.6f*p.gridH,
                               2.0f,-1.0f,3.0f};
        reference.params=p;
        sim.inject(f); sim.step(); reference.step(f);
        const auto actual=sim.readbackDensity(),expected=reference.da.read();
        for (std::size_t i=0;i<actual.size();++i)
            require(std::isfinite(actual[i]) && std::abs(actual[i]-expected[i])<2e-5f,
                    "Graph step differs from ordinary launches: ordering, parity or rebuild failure");
    }
}
void pressureGraphSteps() {
    const SimConfig c{23,17};
    const auto noise=FluidSim::noiseDensity(c,321);
    FluidSim sim(c,noise); ReferenceSim reference(c,noise);
    compareGraphSteps(sim,reference);
    FluidSim moved(std::move(sim));
    compareGraphSteps(moved,reference);
    FluidSim assigned(c,noise); assigned=std::move(moved);
    compareGraphSteps(assigned,reference);
    assigned.resize(13,9);
    ReferenceSim resized({13,9},std::vector<float>(13*9,0));
    compareGraphSteps(assigned,resized);
    std::cout << "PASS: graph versus ordinary launches, nonzero forces, cached replay, odd/even rebuild, move, resize\n";
}

int main() {
    try {
        numerics(2,3); numerics(31,19); advectionAndForces(); pressureGraphSteps();
        SimConfig c{31,19}; const auto noise=FluidSim::noiseDensity(c,123);
        require(noise==FluidSim::noiseDensity(c,123),"Seeded noise must be reproducible");
        FluidSim sim(c,noise);
        sim.params().denDissip=0.8f; sim.params().viscosity=2; sim.params().diffuseIter=3; sim.params().jacobiIter=7;
        for (int i=0;i<3;++i) sim.step();
        auto d=sim.readbackDensity();
        for (std::size_t i=0;i<d.size();++i) require(std::abs(d[i]-noise[i]*0.512f)<1e-5,"Step failed configured dissipation / ping-pong");
        sim.params().denDissip=0.5f; sim.params().diffuseIter=4; sim.params().jacobiIter=8; sim.step();
        d=sim.readbackDensity();
        for (std::size_t i=0;i<d.size();++i) require(std::abs(d[i]-noise[i]*0.256f)<1e-5,"Runtime parameter change was ignored");
        FluidSim moved(std::move(sim));
        FluidSim assigned(c,noise); assigned=std::move(moved); assigned.step();
        assigned.resize(7,11); d=assigned.readbackDensity();
        require(d.size()==77 && std::all_of(d.begin(),d.end(),[](float v){return v==0;}),"Resize did not clear state");
        assigned.params().gridW=8;
        bool rejected=false; try { assigned.step(); } catch (const std::invalid_argument&) { rejected=true; }
        require(rejected,"Inconsistent dimensions were accepted");
        std::cout << "PASS: noise, SimParams, odd/even swaps, move ownership, resize, validation\n";
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
