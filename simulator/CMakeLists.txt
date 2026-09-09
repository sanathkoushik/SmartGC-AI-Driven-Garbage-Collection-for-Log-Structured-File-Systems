cmake_minimum_required(VERSION 3.15)
project(SmartGC_Simulator VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Compiler warning flags
if(MSVC)
    add_compile_options(/W4 /permissive-)
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Statically link the GCC runtime on MinGW/MSYS2. Besides removing the need to
# ship libstdc++/libgcc DLLs alongside the binaries, some MSYS2 installations
# cannot link against the shared runtime at all (ld exits 116 with no
# diagnostic), which otherwise blocks the build entirely.
if(MINGW)
    add_link_options(-static-libgcc -static-libstdc++)
endif()

# Core simulator library
add_library(simulator_core STATIC
    src/config.cpp
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
