cmake_minimum_required(VERSION 3.0)
project(extra_task)
find_package(OpenCV REQUIRED)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

include_directories(${OpenCV_INCLUDE_DIRS})
add_executable(extra_task extra.cpp)
target_link_libraries(extra_task ${OpenCV_LIBS} )