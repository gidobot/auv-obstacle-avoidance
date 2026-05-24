cmake_minimum_required(VERSION 3.14)
project(auv_obstacle_avoidance CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(pybind11 CONFIG REQUIRED)

add_library(auv_oavoid STATIC src/occupancy_map.cpp)
target_include_directories(auv_oavoid PUBLIC include)
target_link_libraries(auv_oavoid PUBLIC Eigen3::Eigen)
set_target_properties(auv_oavoid PROPERTIES POSITION_INDEPENDENT_CODE ON)

pybind11_add_module(occupancy_map_cpp python/bindings.cpp)
target_link_libraries(occupancy_map_cpp PRIVATE auv_oavoid)

install(TARGETS occupancy_map_cpp DESTINATION ${CMAKE_SOURCE_DIR}/..)
