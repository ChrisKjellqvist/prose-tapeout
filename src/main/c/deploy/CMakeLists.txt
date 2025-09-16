cmake_minimum_required(VERSION 3.15.0)
project(beethoven_deploy)
set(CMAKE_CXX_STANDARD 20)

find_package(beethoven_baremetal)

include_directories(include)
add_compile_definitions(BAREMETAL=1)

beethoven_build_baremetal(check SOURCES basic/core_check.cc lib/prose_lib.cc)
beethoven_build_baremetal(full SOURCES basic/full_check.cc lib/prose_lib.cc)
beethoven_build_baremetal(fib SOURCES basic/fib.cc)