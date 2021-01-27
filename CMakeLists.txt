cmake_minimum_required(VERSION 3.17)
project(HT)

set(CMAKE_CXX_STANDARD 14)

find_package(OpenCV REQUIRED)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)


include_directories(${OpenCV_INCLUDE_DIRS})
add_compile_options(-O3)

add_executable(HT main.cpp)
add_link_options(-O3)
target_link_libraries(HT ${OpenCV_LIBS})
target_link_libraries(HT Eigen3::Eigen)
