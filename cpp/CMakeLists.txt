cmake_minimum_required(VERSION 3.14)
project(auv_obstacle_avoidance CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)

# ---------------------------------------------------------------------------
# auv_oavoid — the LCM-free, Python-free core.  This is the only target a
# consumer needs; it is what the acfr-lcm oa-mapper node links against.
# ---------------------------------------------------------------------------
add_library(auv_oavoid STATIC src/occupancy_map.cpp)
target_include_directories(auv_oavoid PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
target_link_libraries(auv_oavoid PUBLIC Eigen3::Eigen)
set_target_properties(auv_oavoid PROPERTIES POSITION_INDEPENDENT_CODE ON)

# ---------------------------------------------------------------------------
# Python extension — built only when this project is configured standalone.
# A consumer that add_subdirectory()s this project wants the core and nothing
# else: pybind11 and a matching Python are not dependencies of a vehicle build.
# Override with -DAUV_OAVOID_BUILD_PYTHON=ON/OFF.
# ---------------------------------------------------------------------------
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    set(_auv_oavoid_python_default ON)
else()
    set(_auv_oavoid_python_default OFF)
endif()
option(AUV_OAVOID_BUILD_PYTHON
       "Build the pybind11 Python extension module"
       ${_auv_oavoid_python_default})

if(AUV_OAVOID_BUILD_PYTHON)
    # Resolve the Python interpreter before pybind11 so the extension is built
    # for the interpreter that will import it.  Without this, CMake may pick a
    # different Python than the active venv and silently install a module with
    # a non-importable ABI suffix, leaving a stale .so in place.  build.sh
    # passes Python_EXECUTABLE explicitly; this find_package honours it.
    find_package(Python COMPONENTS Interpreter Development REQUIRED)
    message(STATUS "Building extension for ${Python_EXECUTABLE} (${Python_VERSION})")
    find_package(pybind11 CONFIG REQUIRED)

    pybind11_add_module(occupancy_map_cpp python/bindings.cpp)
    target_link_libraries(occupancy_map_cpp PRIVATE auv_oavoid)

    # Install beside the Python implementation in the repo root.  Must be
    # CURRENT_SOURCE_DIR: CMAKE_SOURCE_DIR points at the consuming project when
    # this directory is pulled in via add_subdirectory().
    install(TARGETS occupancy_map_cpp DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/..)
endif()
