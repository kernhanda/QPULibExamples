cmake_minimum_required(VERSION 3.4)
project(QPULibExamples CXX)
set(CMAKE_CXX_STANDARD 14)

set(ENABLE_QPU ON CACHE INTERNAL "")
add_subdirectory(QPULib)

add_executable(Dot Dot.cpp)
target_link_libraries(Dot qpu.a openblas)

add_executable(Accumulator Accumulator.cpp)
target_link_libraries(Accumulator qpu.a)
