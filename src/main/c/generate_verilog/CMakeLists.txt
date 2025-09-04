cmake_minimum_required(VERSION 3.26)
project(generate_verilog)
set(CMAKE_CXX_STANDARD 17)

add_executable(generate_gelu_exact generate_gelu_exact.cc)
add_executable(generate_exp_exact generate_exp_exact.cc)
add_executable(generate_invsqrt generate_invsqrt.cc)
add_executable(generate_normal generate_normal.cc)
