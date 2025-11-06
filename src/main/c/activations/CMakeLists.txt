beethoven_build(check_gelu_lut SOURCES check_gelu_lut.cc)
target_link_libraries(check_gelu_lut PUBLIC ${TORCH_LIBRARIES})

add_executable(gelu_gen_lut gelu_lut.cc)
target_link_libraries(gelu_gen_lut ${TORCH_LIBRARIES})

add_executable(exp_gen_lut exp_lut.cc)
target_link_libraries(exp_gen_lut ${TORCH_LIBRARIES})
