cmake_minimum_required(VERSION 3.5)
# set C++ standard to C17
set(CMAKE_CXX_STANDARD 17)
find_package(beethoven REQUIRED)

beethoven_build(echo SOURCES echo.cc)
#target_link_libraries(prose_pytorch_mm_unit PUBLIC pytorch_linear_golden)


