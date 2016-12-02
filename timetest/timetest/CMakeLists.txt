project(timetest)
cmake_minimum_required(VERSION 2.8)
aux_source_directory(. SRC_LIST)
set(CMAKE_CXX_FLAGS "-O3 -msse2 -msse3 -Wall -std=c++11")
add_executable(${PROJECT_NAME} ${SRC_LIST})

