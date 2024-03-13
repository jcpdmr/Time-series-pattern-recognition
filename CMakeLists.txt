cmake_minimum_required(VERSION 3.22)

project(Time-series-pattern-recognition LANGUAGES CUDA CXX)

# Find OpenMP package
find_package(OpenMP REQUIRED)

# Set executable name
add_executable(Time-series-pattern-recognition main.cu)

# Set include directories
target_include_directories(Time-series-pattern-recognition PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

# Set compiler options for CUDA
target_compile_options(Time-series-pattern-recognition PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

# Set compiler features
target_compile_features(Time-series-pattern-recognition PUBLIC cxx_std_17)

# Set NVCC flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --use_fast_math")

# Enable OpenMP
target_link_libraries(Time-series-pattern-recognition PUBLIC OpenMP::OpenMP_CXX)

set(CMAKE_CXX_FLAGS "-O3 -Wall -Wextra")
