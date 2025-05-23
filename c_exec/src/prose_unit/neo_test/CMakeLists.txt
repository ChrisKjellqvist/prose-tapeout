add_executable(mlp src/mlp.cc)
add_executable(mlp_small src/mlp_small.cc)
add_executable(decoder src/decoder.cc)
add_executable(attention_small src/attention_small.cc)
add_executable(attention src/attention.cc)
add_executable(full_model src/full_model.cc)
add_executable(chris_full_model src/chris_full_model.cc)

add_executable(decoder_small src/decoder_small.cc)
add_executable(format_equality_test src/format_equality_test.cc)
add_executable(float2tensor src/floatblob2tensor.cc)

#add_executable(m_matmul_test src/m_matmul_test.cc)

set(targets_to_link mlp mlp_small decoder attention_small attention full_model decoder_small format_equality_test chris_full_model float2tensor)

foreach (target ${targets_to_link})
    target_link_libraries(${target} PUBLIC prose_utils prose_hw_impl golden)
    target_compile_options(${target} PRIVATE -g -O0)
endforeach ()

option(TEST_PROSE "Build tests" OFF)
if (TEST_PROSE)
    add_compile_definitions(TEST_PROSE)
endif ()