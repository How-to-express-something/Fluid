include(FetchContent)
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GLFW_INSTALL OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
if(NOT Python_EXECUTABLE)
    find_program(FLUID_GL_PYTHON NAMES python python3 REQUIRED)
    set(Python_EXECUTABLE "${FLUID_GL_PYTHON}" CACHE FILEPATH "Python for GLAD")
endif()
execute_process(COMMAND "${Python_EXECUTABLE}" -c "import jinja2"
    RESULT_VARIABLE JINJA_STATUS OUTPUT_QUIET ERROR_QUIET)
if(NOT JINJA_STATUS EQUAL 0)
    message(FATAL_ERROR "GLAD requires Jinja2 in ${Python_EXECUTABLE}")
endif()
FetchContent_Declare(glfw
    URL https://codeload.github.com/glfw/glfw/zip/refs/tags/3.4
    URL_HASH SHA256=a133ddc3d3c66143eba9035621db8e0bcf34dba1ee9514a9e23e96afd39fd57a
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
FetchContent_Declare(glad
    URL https://codeload.github.com/Dav1dde/glad/zip/refs/tags/v2.0.8
    URL_HASH SHA256=471886d363ea2b7e2f92081266c4775de91ee971a10359bd33559f1c5a1cbb92
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE SOURCE_SUBDIR cmake)
FetchContent_MakeAvailable(glfw glad)

# Pinned native UI dependency, built only with the GLFW/OpenGL backends we use.
FetchContent_Declare(imgui
    URL https://codeload.github.com/ocornut/imgui/zip/refs/tags/v1.91.9b
    URL_HASH SHA256=fd37507c8476a6d14cc7c4b352401f31bcbd0f0d995d35390811e968c466f46e
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
FetchContent_MakeAvailable(imgui)
add_library(fluid_imgui STATIC
    "${imgui_SOURCE_DIR}/imgui.cpp" "${imgui_SOURCE_DIR}/imgui_draw.cpp"
    "${imgui_SOURCE_DIR}/imgui_tables.cpp" "${imgui_SOURCE_DIR}/imgui_widgets.cpp"
    "${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp"
    "${imgui_SOURCE_DIR}/backends/imgui_impl_opengl3.cpp")
target_include_directories(fluid_imgui PUBLIC "${imgui_SOURCE_DIR}" "${imgui_SOURCE_DIR}/backends")
target_compile_definitions(fluid_imgui PRIVATE GLFW_INCLUDE_NONE)
target_link_libraries(fluid_imgui PUBLIC glfw)
target_compile_features(fluid_imgui PUBLIC cxx_std_17)
glad_add_library(fluid_glad STATIC REPRODUCIBLE API gl:core=3.3 EXTENSIONS NONE)

# CPU-only coordinate/timing checks; the application is defined in the root.
if(BUILD_TESTING)
    add_executable(FluidCoreTests tests/core_tests.cpp)
    target_compile_features(FluidCoreTests PRIVATE cxx_std_17)
    add_test(NAME Fluid_coordinates_and_timing COMMAND FluidCoreTests)
endif()
