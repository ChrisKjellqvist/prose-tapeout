cmake_minimum_required(VERSION 3.15.0)
project(beethoven_deploy)
set(CMAKE_CXX_STANDARD 20)

find_package(beethoven_baremetal)

include_directories(include)
add_subdirectory(basic)
