#pragma once
#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace fluidgl {
struct BenchmarkStats { std::size_t count; double mean,p50,p95,maximum; };
inline BenchmarkStats benchmark_stats(std::vector<double> values) {
    if (values.empty()) throw std::invalid_argument("Cannot summarize empty samples");
    for (double x:values) if (!std::isfinite(x) || x<0) throw std::invalid_argument("Invalid timing sample");
    const double mean=std::accumulate(values.begin(),values.end(),0.0)/values.size();
    std::sort(values.begin(),values.end());
    auto percentile=[&](double q) {
        const double position=q*(values.size()-1);
        const auto lo=static_cast<std::size_t>(position),hi=std::min(lo+1,values.size()-1);
        return values[lo]+(values[hi]-values[lo])*(position-lo);
    };
    return {values.size(),mean,percentile(0.50),percentile(0.95),values.back()};
}
}
