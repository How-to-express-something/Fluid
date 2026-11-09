#include "benchmark.h"
#include "benchmark_stats.h"
#include "fluid.h"
#include "renderer.h"
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#include <array>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace fluidgl {
namespace {
using Clock=std::chrono::steady_clock;
double milliseconds(Clock::time_point start) {
    return std::chrono::duration<double,std::milli>(Clock::now()-start).count();
}
void check(cudaError_t e) { if (e!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
void check_gl() { if (glGetError()!=GL_NO_ERROR) throw std::runtime_error("Benchmark GL operation failed"); }
std::string json(const std::string& s) {
    std::ostringstream out; out << '"';
    for (unsigned char c:s) {
        if (c=='"' || c=='\\') out << '\\' << char(c);
        else if (c<32) out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << unsigned(c) << std::dec;
        else out << char(c);
    }
    out << '"'; return out.str();
}
struct Sample {
    double step_cpu_ms=0,step_gpu_ms=0,publish_cpu_ms=0,draw_submit_cpu_ms=0;
    double draw_gpu_ms=0,swap_cpu_ms=0,recycle_wait_cpu_ms=0,frame_cpu_ms=0;
};
// A bounded pool avoids an unconditional GPU synchronization on every step.
// Reusing a busy slot may block; that wait is measured and not hidden.
struct TimerSlot {
    cudaEvent_t start=nullptr,stop=nullptr;
    GLuint query=0;
    int sample=-1;
    explicit TimerSlot(bool pipeline) {
        try {
            check(cudaEventCreate(&start)); check(cudaEventCreate(&stop));
            if (pipeline) { glGenQueries(1,&query); check_gl(); }
        } catch (...) { release(); throw; }
    }
    void release() noexcept {
        if (query) glDeleteQueries(1,&query);
        if (stop) cudaEventDestroy(stop);
        if (start) cudaEventDestroy(start);
        query=0; start=stop=nullptr;
    }
    ~TimerSlot() { release(); }
    TimerSlot(const TimerSlot&)=delete;
    TimerSlot& operator=(const TimerSlot&)=delete;
    void collect(std::vector<Sample>& samples) {
        if (sample<0) return;
        check(cudaEventSynchronize(stop));
        float elapsed=0; check(cudaEventElapsedTime(&elapsed,start,stop));
        samples[static_cast<std::size_t>(sample)].step_gpu_ms=elapsed;
        if (query) {
            GLuint64 ns=0; glGetQueryObjectui64v(query,GL_QUERY_RESULT,&ns); check_gl();
            samples[static_cast<std::size_t>(sample)].draw_gpu_ms=static_cast<double>(ns)/1e6;
        }
        sample=-1;
    }
};
void scripted_input(FluidSim& fluid,int step) {
    const float t=step*fluid.params().dt;
    const float scale=static_cast<float>(std::min(fluid.gridW(),fluid.gridH()))/256.0f;
    fluid.inject({fluid.gridW()*(0.5f+0.24f*std::sin(t)),
                  fluid.gridH()*(0.5f+0.24f*std::cos(t)),
                  6.0f*scale*std::cos(t),-6.0f*scale*std::sin(t),300.0f});
}
SimParams configured_params(const BenchmarkOptions& o) {
    SimParams p;
    p.gridW=o.grid.gridW; p.gridH=o.grid.gridH;
    p.jacobiIter=o.jacobi; p.diffuseIter=o.diffuse; p.viscosity=o.viscosity;
    p.dt=1.0f/60.0f;
    return p;
}
void check_window(GLFWwindow* window,int w,int h) {
    glfwPollEvents();
    int fw=0,fh=0; glfwGetFramebufferSize(window,&fw,&fh);
    if (glfwWindowShouldClose(window) || glfwGetWindowAttrib(window,GLFW_ICONIFIED) || fw!=w || fh!=h)
        throw std::runtime_error("Benchmark interrupted: window closed, minimized, or framebuffer resized");
}
struct RunInfo { double wall_ms,drain_ms,mass,checksum,minimum,maximum; };
}
void run_benchmark(GLFWwindow* window,const BenchmarkOptions& o) {
    namespace fs=std::filesystem;
    const fs::path prefix=fs::u8path(o.output);
    const fs::path raw_path=fs::u8path(o.output+"_samples.csv");
    const fs::path summary_path=fs::u8path(o.output+"_summary.csv");
    const fs::path meta_path=fs::u8path(o.output+"_metadata.json");
    for (const auto& path:{raw_path,summary_path,meta_path})
        if (fs::exists(path)) throw std::runtime_error("Report already exists; choose a new --output prefix: "+path.u8string());
    if (!prefix.parent_path().empty()) fs::create_directories(prefix.parent_path());
    std::ofstream raw(raw_path),summary(summary_path);
    if (!raw || !summary) throw std::runtime_error("Cannot create benchmark reports");
    raw << std::setprecision(12);
    summary << std::setprecision(12);
    raw << "run,sample,simulation_step,step_cpu_ms,step_gpu_ms,publish_cpu_ms,draw_submit_cpu_ms,draw_gpu_ms,swap_cpu_ms,recycle_wait_cpu_ms,frame_cpu_ms\n";
    summary << "run,metric,unit,count,mean,p50,p95,max\n";
    int device=0,runtime=0,driver=0;
    check(cudaGetDevice(&device)); check(cudaRuntimeGetVersion(&runtime)); check(cudaDriverGetVersion(&driver));
    cudaDeviceProp properties{}; check(cudaGetDeviceProperties(&properties,device));
    int fw=0,fh=0; glfwGetFramebufferSize(window,&fw,&fh);
    if (fw<=0 || fh<=0) throw std::runtime_error("Invalid benchmark framebuffer");
    glfwSwapInterval(0);
    const auto initial=FluidSim::noiseDensity(o.grid,o.seed);
    const auto params=configured_params(o);
    const int pool_size=std::min(o.inflight,o.frames);
    std::vector<RunInfo> runs;
    runs.reserve(o.repeats);
    for (int run=0;run<o.repeats;++run) {
        // Declaration order matters: FluidSim unregisters before Renderer deletes its texture.
        std::unique_ptr<Renderer> renderer;
        if (o.pipeline) renderer=std::make_unique<Renderer>(o.grid.gridW,o.grid.gridH);
        FluidSim fluid(o.grid,initial); fluid.params()=params;
        if (renderer) fluid.attachGLTexture(renderer->texture());
        std::vector<std::unique_ptr<TimerSlot>> pool;
        for (int i=0;i<pool_size;++i) pool.push_back(std::make_unique<TimerSlot>(o.pipeline));
        std::vector<Sample> samples(static_cast<std::size_t>(o.frames));
        // Warm-up follows the same trajectory; initialization/JIT/allocations are outside timing.
        for (int i=0;i<o.warmup;++i) {
            check_window(window,fw,fh);
            scripted_input(fluid,i); fluid.step();
            if (renderer) {
                renderer->update_density(fluid);
                renderer->draw(fw,fh,o.color,0.2f,(i+1)*params.dt);
                glfwSwapBuffers(window);
            }
        }
        check(cudaDeviceSynchronize());
        if (renderer) { glFinish(); check_gl(); }
        const auto run_start=Clock::now();
        for (int i=0;i<o.frames;++i) {
            auto& s=samples[static_cast<std::size_t>(i)];
            auto& slot=*pool[static_cast<std::size_t>(i%pool_size)];
            const auto frame_start=Clock::now();
            const auto recycle_start=Clock::now();
            slot.collect(samples);
            s.recycle_wait_cpu_ms=milliseconds(recycle_start);
            check_window(window,fw,fh);
            scripted_input(fluid,o.warmup+i);
            check(cudaEventRecord(slot.start,0));
            auto begin=Clock::now();
            fluid.step();
            s.step_cpu_ms=milliseconds(begin);
            check(cudaEventRecord(slot.stop,0));
            if (renderer) {
                begin=Clock::now();
                renderer->update_density(fluid);
                s.publish_cpu_ms=milliseconds(begin);
                begin=Clock::now();
                glBeginQuery(GL_TIME_ELAPSED,slot.query);
                renderer->draw(fw,fh,o.color,0.2f,(o.warmup+i+1)*params.dt);
                glEndQuery(GL_TIME_ELAPSED);
                s.draw_submit_cpu_ms=milliseconds(begin);
                begin=Clock::now();
                glfwSwapBuffers(window);
                s.swap_cpu_ms=milliseconds(begin);
            }
            slot.sample=i;
            s.frame_cpu_ms=milliseconds(frame_start);
        }
        const auto drain_start=Clock::now();
        for (auto& slot:pool) slot->collect(samples);
        const double drain_ms=milliseconds(drain_start);
        const double wall_ms=milliseconds(run_start);
        // Explicit CPU readback is after timing. It detects invalid fields and supports repeat checks.
        const auto density=fluid.readbackDensity();
        double mass=0,checksum=0,minimum=density.front(),maximum=minimum;
        for (std::size_t i=0;i<density.size();++i) {
            const float d=density[i];
            if (!std::isfinite(d) || d<0) throw std::runtime_error("Benchmark produced invalid density");
            mass+=d; checksum+=d*static_cast<double>(1+i%97);
            minimum=std::min(minimum,double(d)); maximum=std::max(maximum,double(d));
        }
        runs.push_back({wall_ms,drain_ms,mass,checksum,minimum,maximum});
        for (int i=0;i<o.frames;++i) {
            const auto& s=samples[static_cast<std::size_t>(i)];
            raw << run << ',' << i << ',' << o.warmup+i+1 << ',' << s.step_cpu_ms << ',' << s.step_gpu_ms << ',';
            if (renderer) raw << s.publish_cpu_ms << ',' << s.draw_submit_cpu_ms << ',' << s.draw_gpu_ms << ',' << s.swap_cpu_ms;
            else raw << ",,,"; // Empty columns denote unmeasured stages, not zero cost.
            raw << ',' << s.recycle_wait_cpu_ms << ',' << s.frame_cpu_ms << '\n';
        }
        auto report=[&](const char* name,double Sample::*member) {
            std::vector<double> values; values.reserve(samples.size());
            for (const auto& s:samples) values.push_back(s.*member);
            const auto stats=benchmark_stats(std::move(values));
            summary << run << ',' << name << ",ms," << stats.count << ',' << stats.mean << ',' << stats.p50 << ',' << stats.p95 << ',' << stats.maximum << '\n';
        };
        report("step_cpu",&Sample::step_cpu_ms); report("step_gpu_interval",&Sample::step_gpu_ms);
        report("recycle_wait_cpu",&Sample::recycle_wait_cpu_ms); report("frame_cpu",&Sample::frame_cpu_ms);
        if (renderer) {
            report("publish_cpu",&Sample::publish_cpu_ms); report("draw_submit_cpu",&Sample::draw_submit_cpu_ms);
            report("draw_gpu",&Sample::draw_gpu_ms); report("swap_cpu",&Sample::swap_cpu_ms);
        }
        auto scalar=[&](const char* name,const char* unit,double value) {
            summary << run << ',' << name << ',' << unit << ",1," << value << ',' << value << ',' << value << ',' << value << '\n';
        };
        scalar("completed_run_wall","ms",wall_ms); scalar("final_drain_cpu","ms",drain_ms);
        scalar("completed_steps_per_second","steps/s",o.frames*1000.0/wall_ms);
        scalar("density_mass","density",mass); scalar("density_checksum","weighted_density",checksum);
        scalar("density_min","density",minimum); scalar("density_max","density",maximum);
        std::cout << "Benchmark " << (o.pipeline?"pipeline":"solver") << " run " << run
                  << ": " << wall_ms << " ms, " << o.frames*1000.0/wall_ms << " completed steps/s\n";
    }
    raw.close(); summary.close();
    if (!raw || !summary) throw std::runtime_error("Failed writing benchmark CSV");
    // A metadata file is written only after all runs finish successfully.
    std::ofstream meta(meta_path); if (!meta) throw std::runtime_error("Cannot create benchmark metadata");
    meta << std::setprecision(12) << std::boolalpha;
    meta << "{\n  \"schema_version\": 1,\n  \"mode\": " << json(o.pipeline?"pipeline":"solver")
         << ",\n  \"label\": " << json(o.label)
         << ",\n  \"pressure_backend\": " << json(FLUID_BENCHMARK_VARIANT)
#ifdef FLUID_SOURCE_SNAPSHOT
         << ",\n  \"source_snapshot_id\": " << json(FLUID_SOURCE_SNAPSHOT)
#endif
         << ",\n  \"device_name\": " << json(properties.name)
         << ",\n  \"compute_capability\": " << json(std::to_string(properties.major)+"."+std::to_string(properties.minor))
         << ",\n  \"cuda_runtime_version\": " << runtime << ",\n  \"cuda_driver_version\": " << driver
         << ",\n  \"gl_renderer\": " << json(reinterpret_cast<const char*>(glGetString(GL_RENDERER)))
         << ",\n  \"gl_version\": " << json(reinterpret_cast<const char*>(glGetString(GL_VERSION)))
#ifdef FLUID_BUILD_CONFIG
         << ",\n  \"build_config\": " << json(FLUID_BUILD_CONFIG)
#endif
         << ",\n  \"build_timestamp\": " << json(__DATE__ " " __TIME__)
         << ",\n  \"grid_width\": " << o.grid.gridW << ",\n  \"grid_height\": " << o.grid.gridH
         << ",\n  \"requested_window_width\": " << o.window_width << ",\n  \"requested_window_height\": " << o.window_height
         << ",\n  \"framebuffer_width\": " << fw << ",\n  \"framebuffer_height\": " << fh
         << ",\n  \"visible\": " << o.visible << ",\n  \"vsync_requested\": 0,\n  \"color\": " << o.color
         << ",\n  \"seed\": " << o.seed << ",\n  \"frames\": " << o.frames << ",\n  \"warmup\": " << o.warmup
         << ",\n  \"repeats\": " << o.repeats << ",\n  \"timer_pool_size\": " << pool_size
         << ",\n  \"trajectory\": \"circle_v1_scaled_to_grid\",\n  \"dt\": " << params.dt
         << ",\n  \"jacobi_iterations\": " << params.jacobiIter << ",\n  \"diffuse_iterations\": " << params.diffuseIter
         << ",\n  \"viscosity\": " << params.viscosity << ",\n  \"velocity_dissipation\": " << params.velDissip
         << ",\n  \"density_dissipation\": " << params.denDissip << ",\n  \"force_radius\": " << params.forceRadius
         << ",\n  \"force_strength\": " << params.forceStrength << ",\n  \"buoyancy\": " << params.buoyancy
         << ",\n  \"display_gain\": 0.2,\n  \"solver_buffer_bytes\": "
         << static_cast<std::size_t>(o.grid.gridW)*o.grid.gridH*(3*sizeof(float2)+5*sizeof(float))
         << ",\n  \"density_texture_payload_bytes\": " << (o.pipeline?static_cast<std::size_t>(o.grid.gridW)*o.grid.gridH*sizeof(float):0)
         << ",\n  \"percentile_method\": \"linear interpolation at q*(N-1)\",\n  \"results\": [\n";
    for (std::size_t i=0;i<runs.size();++i) {
        const auto& r=runs[i];
        meta << "    {\"run\": " << i << ", \"wall_ms\": " << r.wall_ms << ", \"final_drain_ms\": " << r.drain_ms
             << ", \"density_mass\": " << r.mass << ", \"density_checksum\": " << r.checksum
             << ", \"density_min\": " << r.minimum << ", \"density_max\": " << r.maximum << '}'
             << (i+1==runs.size()?"\n":",\n");
    }
    meta << "  ]\n}\n"; meta.close();
    if (!meta) throw std::runtime_error("Failed writing benchmark metadata");
    std::cout << "Reports: " << prefix.u8string() << "_{samples.csv,summary.csv,metadata.json}\n";
}
}
