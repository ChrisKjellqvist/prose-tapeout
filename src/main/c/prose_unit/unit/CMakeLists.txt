cmake_minimum_required(VERSION 3.5)
# set C++ standard to C17
set(CMAKE_CXX_STANDARD 17)
find_package(beethoven REQUIRED)
include_directories(../../utility)
beethoven_build(prose_m_unit SOURCES prose_m_unit.cc)
beethoven_build(prose_e_unit SOURCES prose_e_unit.cc)
beethoven_build(prose_g_unit SOURCES prose_g_unit.cc)
beethoven_build(unit_layernorm SOURCES unit_norm.cc)
beethoven_build(random_layernorm SOURCES unit_random_norm.cc)
target_link_libraries(prose_m_unit PUBLIC prose_utils prose_hw_impl)
target_link_libraries(prose_e_unit PUBLIC prose_utils prose_hw_impl)
target_link_libraries(prose_g_unit PUBLIC prose_utils prose_hw_impl)

add_executable(gen_matrix ../generate_arrays.cc)
target_link_libraries(gen_matrix PUBLIC prose_utils)

if ("${BAREMETAL}" STREQUAL "1")
  add_compile_definitions(prose_e_unit.cc BAREMETAL)
endif()
#target_link_libraries(prose_pytorch_mm_unit PUBLIC pytorch_linear_golden)

