#pragma once
#include "sim_types.h"
#include <charconv>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace fluidgl {
struct BenchmarkOptions {
    bool pipeline=false, color=true, visible=false;
    SimConfig grid{};
    int frames=1200, warmup=120, repeats=3, inflight=4;
    int jacobi=60, diffuse=20, window_width=1024, window_height=720;
    float viscosity=0;
    std::uint32_t seed=123;
    std::string output="benchmarks/fluid", label="unlabeled";
};
inline BenchmarkOptions parse_benchmark(int argc,const char* const* argv) {
    if (argc<3) throw std::invalid_argument("Expected --benchmark solver|pipeline");
    BenchmarkOptions o;
    const std::string mode=argv[2];
    if (mode=="pipeline") o.pipeline=true;
    else if (mode!="solver") throw std::invalid_argument("Benchmark mode must be solver or pipeline");
    auto argument=[&](int& i) -> std::string {
        if (++i>=argc) throw std::invalid_argument("Missing benchmark option value");
        return argv[i];
    };
    auto integer=[](const std::string& s,int lo,int hi) {
        int v=0; const auto r=std::from_chars(s.data(),s.data()+s.size(),v);
        if (r.ec!=std::errc{} || r.ptr!=s.data()+s.size() || v<lo || v>hi)
            throw std::invalid_argument("Invalid integer or range: "+s);
        return v;
    };
    for (int i=3;i<argc;++i) {
        const std::string key=argv[i];
        if (key=="--grid") {
            o.grid.gridW=integer(argument(i),2,2048); o.grid.gridH=integer(argument(i),2,2048);
        } else if (key=="--window") {
            o.window_width=integer(argument(i),64,8192); o.window_height=integer(argument(i),64,8192);
        } else if (key=="--frames") o.frames=integer(argument(i),1,1000000);
        else if (key=="--warmup") o.warmup=integer(argument(i),0,1000000);
        else if (key=="--repeats") o.repeats=integer(argument(i),1,100);
        else if (key=="--inflight") o.inflight=integer(argument(i),1,64);
        else if (key=="--jacobi") o.jacobi=integer(argument(i),1,10000);
        else if (key=="--diffuse") o.diffuse=integer(argument(i),1,10000);
        else if (key=="--seed") {
            const auto s=argument(i); const auto r=std::from_chars(s.data(),s.data()+s.size(),o.seed);
            if (r.ec!=std::errc{} || r.ptr!=s.data()+s.size()) throw std::invalid_argument("Invalid seed");
        } else if (key=="--viscosity") {
            const auto s=argument(i); std::size_t used=0;
            o.viscosity=std::stof(s,&used);
            if (used!=s.size() || !std::isfinite(o.viscosity) || o.viscosity<0 || o.viscosity>1000)
                throw std::invalid_argument("Viscosity must be finite and in [0,1000]");
        } else if (key=="--output") o.output=argument(i);
        else if (key=="--label") o.label=argument(i);
        else if (key=="--gray") o.color=false;
        else if (key=="--visible") o.visible=true;
        else throw std::invalid_argument("Unknown benchmark option: "+key);
    }
    if (o.output.empty() || o.label.empty()) throw std::invalid_argument("Output and label must not be empty");
    return o;
}
inline const char* benchmark_usage() {
    return "Fluid --benchmark solver|pipeline [--grid W H] [--window W H]\n"
           "  [--frames N] [--warmup N] [--repeats N] [--inflight N]\n"
           "  [--jacobi N] [--viscosity V] [--diffuse N] [--seed N]\n"
           "  [--gray] [--visible] [--label NAME] [--output PREFIX]\n"
           "Writes PREFIX_samples.csv, PREFIX_summary.csv, PREFIX_metadata.json.\n"
           "Existing reports are not overwritten. Pipeline uses VSync=0.\n";
}
}
