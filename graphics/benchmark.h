#pragma once
#include "benchmark_options.h"
struct GLFWwindow;
namespace fluidgl {
void run_benchmark(GLFWwindow*,const BenchmarkOptions&);
}
