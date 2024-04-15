cmake_minimum_required(VERSION 3.5.0)
project(lb6)

#set(CMAKE_CXX_COMPILER Dmingw64bing++)
#set(BUILD_SHARED_LIBS ON)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(lb6 main.cpp)

target_link_libraries( lb6 ${OpenCV_LIBS} )