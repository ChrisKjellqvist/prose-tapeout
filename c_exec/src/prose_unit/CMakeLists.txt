cmake_minimum_required(VERSION 3.5)
project(prose_unit_tests)
cmake_policy(SET CMP0074 NEW)

set(CMAKE_CXX_STANDARD 20)

#set(BEETHOVEN_ROOT "/Users/entropy/Development/Composer-Software/beethoven_dir" CACHE PATH "Path to Beethoven library")
option(USE_TORCH "Enable PyTorch testing" ON)
if (${USE_TORCH})
    message("Enabling PyTorch")
    find_package(Torch REQUIRED)
    add_compile_definitions(USE_TORCH=1)
    include_directories(${TORCH_INCLUDE_DIRS})
endif ()
find_package(beethoven REQUIRED)

add_subdirectory(unit)
add_subdirectory(golden)
add_subdirectory(utility)
add_subdirectory(hw_impl)
add_subdirectory(neo_test)
