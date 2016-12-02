project(toontest)
cmake_minimum_required(VERSION 2.8)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")

find_package(TooN REQUIRED)

include_directories(
        ${CMAKE_BINARY_DIR}/include
        ${TOON_INCLUDE_PATHS}
)

aux_source_directory(. SRC_LIST)
add_executable(${PROJECT_NAME} ${SRC_LIST})

