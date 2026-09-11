cmake_minimum_required(VERSION 3.15)
project(SmartGC_Simulator VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# The MinGW Makefiles generator defaults to putting include flags in an
# @response-file to dodge Windows command-line length limits. clangd (used by
# editor tooling via compile_commands.json) does not reliably resolve those on
# Windows, which shows up as false "file not found" / "unknown type" errors
# for project headers and even <cstdint>/std:: in the editor despite the
# actual build succeeding. Forcing inline flags fixes editor IntelliSense with
# no effect on the real build (this project's command lines are nowhere near
# the length limit response files exist to avoid).
set(CMAKE_CXX_USE_RESPONSE_FILE_FOR_INCLUDES OFF)

# Compiler warning flags
if(MSVC)
    add_compile_options(/W4 /permissive-)
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Core simulator library
add_library(simulator_core STATIC
    src/segment.cpp
    src/lfs_simulator.cpp
    src/synthetic_workload.cpp
)

target_include_directories(simulator_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Simulator CLI runner
add_executable(smartgc_sim
    src/main.cpp
)

target_link_libraries(smartgc_sim PRIVATE
    simulator_core
)

# Simulator Unit Tests
add_executable(simulator_tests
    tests/test_lfs.cpp
)

target_include_directories(simulator_tests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/tests
)

target_link_libraries(simulator_tests PRIVATE
    simulator_core
)
