add_library(golden SHARED
        src/prose_golden.cc)
target_include_directories(golden PUBLIC ./include ${TORCH_INCLUDE_DIRS})
message(STATUS "TORCH_INCLUDE_DIRS: ${TORCH_INCLUDE_DIRS}")
target_link_libraries(golden PUBLIC ${TORCH_LIBRARIES} prose_utils)
