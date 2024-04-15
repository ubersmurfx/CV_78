cmake_minimum_required(VERSION 3.5.0)
project(lb3)

#set(CMAKE_CXX_COMPILER Dmingw64bing++)
#set(BUILD_SHARED_LIBS ON)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(lb3 main.cpp)

target_link_libraries( lb3 ${OpenCV_LIBS} )