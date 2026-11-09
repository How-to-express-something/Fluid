cmake_minimum_required(VERSION 3.24)

# CUDA architecture must be selected before initializing the CUDA compiler.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()
project(Fluid LANGUAGES C CXX CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()
find_package(CUDAToolkit REQUIRED)
include(CTest)
add_subdirectory(graphics)

# The sole application: GLFW/OpenGL frontend and the CUDA solver.
add_executable(Fluid
    graphics/main.cpp
    graphics/application.cpp
    graphics/renderer.cpp
    graphics/benchmark.cpp
    graphics/control_panel.cpp
    graphics/panel_test.cpp
    source/fluid.cu
    source/fluid_kernel.cu)
target_include_directories(Fluid PRIVATE source graphics)
target_link_libraries(Fluid PRIVATE CUDA::cudart glfw fluid_glad fluid_imgui)
# Content identity is independent of build timestamps and includes uncommitted inputs.
file(GLOB_RECURSE FLUID_SNAPSHOT_INPUTS CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/source/*.h" "${CMAKE_CURRENT_SOURCE_DIR}/source/*.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/graphics/*.h" "${CMAKE_CURRENT_SOURCE_DIR}/graphics/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/graphics/tests/*.cu")
list(APPEND FLUID_SNAPSHOT_INPUTS "${CMAKE_CURRENT_SOURCE_DIR}/CMakeLists.txt"
    "${CMAKE_CURRENT_SOURCE_DIR}/graphics/CMakeLists.txt")
list(SORT FLUID_SNAPSHOT_INPUTS)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${FLUID_SNAPSHOT_INPUTS})
set(FLUID_SNAPSHOT_CONTENT "")
foreach(FLUID_INPUT IN LISTS FLUID_SNAPSHOT_INPUTS)
    file(SHA256 "${FLUID_INPUT}" FLUID_INPUT_HASH)
    file(RELATIVE_PATH FLUID_INPUT_NAME "${CMAKE_CURRENT_SOURCE_DIR}" "${FLUID_INPUT}")
    string(APPEND FLUID_SNAPSHOT_CONTENT "${FLUID_INPUT_NAME}:${FLUID_INPUT_HASH}\n")
endforeach()
string(SHA256 FLUID_SOURCE_SNAPSHOT "${FLUID_SNAPSHOT_CONTENT}")
set(FLUID_BENCHMARK_VARIANT "graph" CACHE STRING "Version label for benchmark metadata (does not select code)")
target_compile_definitions(Fluid PRIVATE FLUID_BUILD_CONFIG="$<CONFIG>"
    FLUID_BENCHMARK_VARIANT="${FLUID_BENCHMARK_VARIANT}"
    FLUID_SOURCE_SNAPSHOT="${FLUID_SOURCE_SNAPSHOT}")
if(MSVC)
    target_compile_options(Fluid PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/W4>
        $<$<COMPILE_LANGUAGE:CXX>:/utf-8>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/W4,-wd4819>
        $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
        $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
    target_compile_definitions(Fluid PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
    # GUI executable: use main's CRT startup without opening an extra console.
    set_target_properties(Fluid PROPERTIES WIN32_EXECUTABLE TRUE)
    target_link_options(Fluid PRIVATE /ENTRY:mainCRTStartup)
else()
    target_compile_options(Fluid PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra>)
endif()
if(BUILD_TESTING)
    add_test(NAME Fluid_cuda_render COMMAND Fluid --smoke-test)
    set_tests_properties(Fluid_cuda_render PROPERTIES TIMEOUT 60)
    add_test(NAME Fluid_parameter_panel COMMAND Fluid --panel-test)
    set_tests_properties(Fluid_parameter_panel PROPERTIES TIMEOUT 60)
    add_executable(FluidBenchmarkCoreTests graphics/tests/benchmark_core_tests.cpp)
    target_include_directories(FluidBenchmarkCoreTests PRIVATE source graphics)
    add_test(NAME Fluid_benchmark_options_and_statistics COMMAND FluidBenchmarkCoreTests)
    add_test(NAME Fluid_benchmark_reports
        COMMAND "${Python_EXECUTABLE}" -B "${CMAKE_CURRENT_SOURCE_DIR}/graphics/tests/benchmark_smoke.py"
                --exe "$<TARGET_FILE:Fluid>")
    set_tests_properties(Fluid_benchmark_reports PROPERTIES TIMEOUT 90)
endif()

if(BUILD_TESTING)
    add_executable(FluidSolverTests graphics/tests/solver_tests.cu source/fluid.cu source/fluid_kernel.cu)
    target_include_directories(FluidSolverTests PRIVATE source)
    target_link_libraries(FluidSolverTests PRIVATE CUDA::cudart fluid_glad)
    target_compile_definitions(FluidSolverTests PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
    add_test(NAME Fluid_solver_numerics COMMAND FluidSolverTests)
    set_tests_properties(Fluid_solver_numerics PROPERTIES TIMEOUT 60)
endif()
