add_library(
        prose_utils SHARED
        util.cc)
target_include_directories(prose_utils PUBLIC ./ ${TORCH_INCLUDE_DIRS})
target_link_libraries(prose_utils PUBLIC ${TORCH_LIBRARIES})
beethoven_build(kill_sim SOURCES kill_sim.cc)