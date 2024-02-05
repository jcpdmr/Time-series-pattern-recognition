cmake_minimum_required(VERSION 3.26)
project(Time-series-pattern-recognition)
find_package(OpenMP REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_executable(Time-series-pattern-recognition main.cpp utility.h utility.cpp)

target_link_libraries(Time-series-pattern-recognition PUBLIC OpenMP::OpenMP_CXX)
