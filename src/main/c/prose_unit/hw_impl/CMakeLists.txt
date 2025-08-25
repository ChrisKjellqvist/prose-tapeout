beethoven_library(
        prose_hw_impl
        SOURCES
            src/prose_impl.cc
            src/float_wrapper.cc)

target_include_directories(prose_hw_impl PUBLIC include)
target_link_libraries(prose_hw_impl PUBLIC prose_utils)

