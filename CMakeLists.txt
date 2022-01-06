cmake_minimum_required(VERSION 3.16)

set(CMAKE_CXX_STANDARD 14)

project(CV_lab2)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(CV_lab2 main.cpp)

# linking
target_link_libraries(CV_lab2 ${OpenCV_LIBS})