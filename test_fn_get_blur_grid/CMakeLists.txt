cmake_minimum_required(VERSION 3.17)
project(test_fn_get_blur_grid)
find_package (Eigen3 3.3 REQUIRED NO_MODULE)


set(CMAKE_CXX_STANDARD 14)

add_executable(test_fn_get_blur_grid main.cpp)
target_link_libraries (test_fn_get_blur_grid Eigen3::Eigen)