cmake_minimum_required(VERSION 3.17)
project(Hist_of_height)

set(CMAKE_CXX_STANDARD 14)

find_package(OpenCV REQUIRED)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)


include_directories(${OpenCV_INCLUDE_DIRS})
add_compile_options(-O3)

add_link_options(-O3)
add_executable(Hist_of_height main.cpp)
target_link_libraries(Hist_of_height ${OpenCV_LIBS})
target_link_libraries(Hist_of_height Eigen3::Eigen)

