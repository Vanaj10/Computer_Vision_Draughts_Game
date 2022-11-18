cmake_minimum_required(VERSION 3.12)
project(Start)
find_package(OpenCV REQUIRED)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

include_directories(${OpenCV_INCLUDE_DIRS})
add_executable(Start myApplication.cpp)
target_link_libraries(Start ${OpenCV_LIBS})