cmake_minimum_required(VERSION 3.26)
project(Time-series-pattern-recognition)
find_package(OpenMP REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# set(CMAKE_CXX_FLAGS "-O3 -Wall -Wextra")

add_executable(Time-series-pattern-recognition main.cpp utility.h utility.cpp)

target_link_libraries(Time-series-pattern-recognition PUBLIC OpenMP::OpenMP_CXX)
