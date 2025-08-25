cmake_minimum_required(VERSION 3.15)
project(prose)
find_package(beethoven REQUIRED)
include_directories(SYSTEM ${TORCH_INCLUDE_DIRS})

set(CMAKE_CXX_STANDARD 17)

add_subdirectory(prose_unit)
add_subdirectory(model)
add_subdirectory(layernorm)

if("${BAREMETAL}" STREQUAL "1")
    message("AHH")
    add_compile_definitions(BAREMETAL)
endif()

beethoven_build(kill SOURCES kill.cc)
