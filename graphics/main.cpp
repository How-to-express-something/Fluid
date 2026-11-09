#include "application.h"
#include "benchmark.h"
#include <exception>
#include <iostream>
#include <string>
#include <optional>

int main(int argc,char** argv) {
    if (argc==2 && std::string(argv[1])=="--help") {
        std::cout << "Fluid [--demo | --preview | --smoke-test | --panel-test]\n" << fluidgl::benchmark_usage(); return 0;
    }
    const bool bench=argc>1 && std::string(argv[1])=="--benchmark";
    std::optional<fluidgl::BenchmarkOptions> benchmark;
    if (bench) {
        try { benchmark=fluidgl::parse_benchmark(argc,argv); }
        catch (const std::exception& error) { std::cerr << error.what() << '\n' << fluidgl::benchmark_usage(); return 2; }
    }
    const bool smoke=argc==2 && std::string(argv[1])=="--smoke-test";
    const bool panel_test=argc==2 && std::string(argv[1])=="--panel-test";
    const bool preview=argc==2 && std::string(argv[1])=="--preview";
    const bool demo=argc==2 && std::string(argv[1])=="--demo";
    if (argc>1 && !smoke && !panel_test && !preview && !demo && !bench) {
        std::cout << "Fluid [--demo | --preview | --smoke-test | --panel-test]\n" << fluidgl::benchmark_usage(); return 2;
    }
    glfwSetErrorCallback([](int code,const char* text){std::cerr << "GLFW " << code << ": " << text << '\n';});
    if (!glfwInit()) { std::cerr << "GLFW initialization failed\n"; return 1; }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR,3); glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR,3);
    glfwWindowHint(GLFW_OPENGL_PROFILE,GLFW_OPENGL_CORE_PROFILE);
#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT,GLFW_TRUE);
#endif
    if (smoke || panel_test || (benchmark && !benchmark->visible)) glfwWindowHint(GLFW_VISIBLE,GLFW_FALSE);
    const int width=benchmark ? benchmark->window_width : 1024;
    const int height=benchmark ? benchmark->window_height : 720;
    GLFWwindow* window=glfwCreateWindow(width,height,"Fluid | CUDA density",nullptr,nullptr);
    if (!window) { std::cerr << "Window/context creation failed\n"; glfwTerminate(); return 1; }
    glfwMakeContextCurrent(window);
    if (!gladLoadGL(reinterpret_cast<GLADloadfunc>(glfwGetProcAddress))) {
        std::cerr << "GLAD initialization failed\n"; glfwDestroyWindow(window); glfwTerminate(); return 1;
    }
    std::cout << "OpenGL: " << glGetString(GL_VERSION) << "\nRenderer: " << glGetString(GL_RENDERER) << '\n';
    int result=0;
    try {
        fluidgl::Renderer::select_cuda_device();
        if (benchmark) fluidgl::run_benchmark(window,*benchmark);
        else {
            fluidgl::Application app(window,!smoke);
            if (smoke) app.smoke_test();
            else if (panel_test) app.panel_test();
            else app.run(preview,demo);
        }
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; result=1; }
    // app (including Renderer) has already been destroyed with a current context.
    glfwDestroyWindow(window); glfwTerminate(); return result;
}
