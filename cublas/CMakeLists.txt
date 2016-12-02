project(cublas)
cmake_minimum_required(VERSION 2.8)

find_package(CUDA QUIET)

OPTION(WITH_CUDA "CUDA support?" ${CUDA_FOUND})
include_directories(${CUDA_INCLUDE_DIRS})
IF(WITH_CUDA)

 cuda_add_executable(cublas main.cpp)
  target_link_libraries(cublas ${CUDA_LIBRARIES})
  target_link_libraries(cublas cublas.so)
ELSE()
 MESSAGE( FATAL_ERROR "message to display")
ENDIF()


