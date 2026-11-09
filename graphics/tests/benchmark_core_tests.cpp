#include "benchmark_options.h"
#include "benchmark_stats.h"
#include <iostream>
#include <limits>
#include <stdexcept>

using namespace fluidgl;
void require(bool value,const char* message) { if (!value) throw std::runtime_error(message); }
void invalid(std::initializer_list<const char*> args) {
    bool failed=false;
    try { parse_benchmark(static_cast<int>(args.size()),args.begin()); }
    catch (const std::exception&) { failed=true; }
    require(failed,"Invalid benchmark options were accepted");
}
int main() {
    try {
        // Known percentiles verify the report's public statistics contract.
        const auto s=benchmark_stats({4,1,3,2});
        require(s.count==4 && s.mean==2.5 && s.p50==2.5 && std::abs(s.p95-3.85)<1e-12 && s.maximum==4,
                "Statistics/percentile calculation is wrong");
        require(benchmark_stats({7}).p95==7,"Single sample percentile failed");
        bool failed=false;
        try { benchmark_stats({1,std::numeric_limits<double>::quiet_NaN()}); }
        catch (const std::invalid_argument&) { failed=true; }
        require(failed,"Non-finite sample was accepted");
        invalid({"Fluid","--benchmark"});
        invalid({"Fluid","--benchmark","unknown"});
        invalid({"Fluid","--benchmark","solver","--frames","0"});
        invalid({"Fluid","--benchmark","solver","--grid","31"});
        invalid({"Fluid","--benchmark","solver","--grid","2049","19"});
        invalid({"Fluid","--benchmark","solver","--frames","12oops"});
        invalid({"Fluid","--benchmark","solver","--viscosity","nan"});
        invalid({"Fluid","--benchmark","solver","--seed","-1"});
        invalid({"Fluid","--benchmark","solver","--unsupported"});
        const char* args[]={"Fluid","--benchmark","pipeline","--grid","31","19","--window","320","240",
            "--warmup","0","--seed","4294967295","--gray","--visible","--output","reports/example","--label","baseline"};
        const auto o=parse_benchmark(static_cast<int>(sizeof(args)/sizeof(args[0])),args);
        require(o.pipeline && !o.color && o.visible && o.grid.gridW==31 && o.grid.gridH==19
            && o.warmup==0 && o.seed==4294967295u && o.output=="reports/example" && o.label=="baseline",
            "Valid options were parsed incorrectly");
        std::cout << "PASS: benchmark argument validation and known percentile cases\n";
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
