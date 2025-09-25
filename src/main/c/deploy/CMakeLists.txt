cmake_minimum_required(VERSION 3.15.0)
project(beethoven_deploy)
set(CMAKE_CXX_STANDARD 20)

option(LOCAL OFF)
include_directories(include)
message(STATUS "LOCAL: ${LOCAL}")
set(EXTRA_SOURCES "")

add_library(prose lib/prose_lib.cc lib/prose_rptr.cc)

if (${LOCAL})
  add_compile_definitions(LOCAL=1)
  find_package(beethoven REQUIRED)
  set(EXTRA_SOURCES "${EXTRA_SOURCES} lib/kria_lib.cc")

  beethoven_build(norm_check SOURCES basic/norm_check.cc)
  target_link_libraries(norm_check PUBLIC prose)
else()
  add_compile_definitions(BAREMETAL=1)
  find_package(beethoven_baremetal REQUIRED)
endif()
link_beethoven_to_target(prose)

beethoven_build(core SOURCES basic/core_check.cc)
beethoven_build(full SOURCES basic/full_check.cc)
beethoven_build(fib SOURCES basic/fib.cc)
target_link_libraries(core PUBLIC prose)
target_link_libraries(full PUBLIC prose)