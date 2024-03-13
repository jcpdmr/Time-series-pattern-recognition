cmake_minimum_required(VERSION 3.22)
project(Time-series-pattern-recognition LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# OpenMP
add_executable(OpenMP-Time-series-pattern-recognition main.cpp utility.h)

find_package(OpenMP REQUIRED)

target_link_libraries(OpenMP-Time-series-pattern-recognition PUBLIC OpenMP::OpenMP_CXX)

set(CMAKE_CXX_FLAGS "-O3 -Wall -Wextra")

# CUDA
add_executable(CUDA-Time-series-pattern-recognition main.cu utility.h utility.h)

# Set include directories for CUDA
target_include_directories(CUDA-Time-series-pattern-recognition PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

# Set compiler options for CUDA
target_compile_options(CUDA-Time-series-pattern-recognition PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

# Set NVCC flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --use_fast_math")

