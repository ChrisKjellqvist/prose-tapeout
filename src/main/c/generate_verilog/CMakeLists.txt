cmake_minimum_required(VERSION 3.5)
project(generate_verilog)
set(CMAKE_CXX_STANDARD 20)

add_executable(generate_gelu_rom generate_gelu_feed.cc)
add_executable(generate_exp_rom generate_exp_feed.cc)
add_executable(generate_invsqrt generate_invsqrt.cc)
add_executable(generate_normal generate_normal.cc)
add_executable(generate_reciprocal generate_reciprocal.cc)
