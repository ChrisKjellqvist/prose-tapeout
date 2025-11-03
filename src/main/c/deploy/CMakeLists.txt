cmake_minimum_required(VERSION 3.15.0)
project(beethoven_deploy)
set(CMAKE_CXX_STANDARD 20)

option(LOCAL OFF)
option(VERBOSE OFF)
include_directories(include)
message(STATUS "LOCAL: ${LOCAL}")
set(EXTRA_SOURCES "")

add_library(prose lib/prose_lib.cc lib/prose_rptr.cc)


if (${VERBOSE})
  add_compile_definitions(VERBOSE=1)
endif()

if (${LOCAL})
  add_compile_definitions(LOCAL=1)
  find_package(beethoven REQUIRED)
  set(EXTRA_SOURCES "${EXTRA_SOURCES} lib/kria_lib.cc")

  beethoven_build(norm_check SOURCES basic/norm_check.cc)
  beethoven_build(output_proj SOURCES basic/local_output_proj_test.cc)
  target_link_libraries(output_proj PUBLIC prose)
  beethoven_build(mlp SOURCES basic/local_mlp_test.cc)
  target_link_libraries(mlp PUBLIC prose)

else()
  add_compile_definitions(BAREMETAL=1)
  find_package(beethoven_baremetal REQUIRED)
endif()
link_beethoven_to_target(prose)

beethoven_build(decoder SOURCES basic/decoder_check.cc)
beethoven_build(full SOURCES basic/full_check.cc)
beethoven_build(fib SOURCES basic/fib.cc)
beethoven_build(sanity_m SOURCES basic/sanity_check_m.cc)

target_link_libraries(decoder PUBLIC prose)
target_link_libraries(full PUBLIC prose)
target_link_libraries(sanity_m PUBLIC prose)
