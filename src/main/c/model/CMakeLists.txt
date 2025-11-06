# add cmake search direcotyr in libtorch/share/cmake
set(CMAKE_PREFIX_PATH "libtorch/share/cmake")

find_package(beethoven REQUIRED)
set(CMAKE_CXX_STANDARD 20)
add_library(util STATIC ../utility/util.cc)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

beethoven_library(nn_ops
        SOURCES nn_ops.cc)