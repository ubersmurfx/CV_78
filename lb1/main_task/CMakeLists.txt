cmake_minimum_required(VERSION 3.0)
project(main_task)
find_package(OpenCV REQUIRED)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

include_directories(${OpenCV_INCLUDE_DIRS})
add_executable(main_task main.cpp)
target_link_libraries(main_task ${OpenCV_LIBS} )