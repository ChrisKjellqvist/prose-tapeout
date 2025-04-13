cmake_minimum_required(VERSION 3.15)
project(c_exec)
find_package(beethoven REQUIRED)
include_directories(SYSTEM ${TORCH_INCLUDE_DIRS})

set(CMAKE_CXX_STANDARD 17)

add_subdirectory(src/prose_unit)
add_subdirectory(src/model)
add_subdirectory(src/layernorm)

if("${BAREMETAL}" STREQUAL "1")
    message("AHH")
    add_compile_definitions(BAREMETAL)
endif()

beethoven_build(kill SOURCES src/kill.cc)
