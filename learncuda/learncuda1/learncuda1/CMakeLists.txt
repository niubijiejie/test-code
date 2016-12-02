project(learncuda1)
cmake_minimum_required(VERSION 2.8)

find_package(CUDA)
find_package(OpenGL REQUIRED)

OPTION(WITH_CUDA "CUDA support?" ${CUDA_FOUND})

set(EXECUTABLE_OUTPUT_PATH, ${PROJECT_SOURCE_DIR}/bin)

add_subdirectory(src)

include_directories(${PROJECT_SOURCE_DIR}/src)

IF(WITH_CUDA)
include_directories(${CUDA_INCLUDE_DIRS})
 cuda_add_executable(learncuda1 main.cpp)
  target_link_libraries(learncuda1 ${CUDA_LIBRARY})
ELSE()
 MESSAGE( FATAL_ERROR "message to display")
ENDIF()
target_link_libraries(learncuda1 src)


