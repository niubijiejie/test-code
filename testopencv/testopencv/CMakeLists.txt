project(testopencv)
SET( CMAKE_BUILD_TYPE Debug )
INCLUDE_DIRECTORIES(/usr/local/inlude)
cmake_minimum_required(VERSION 2.8)
aux_source_directory(. SRC_LIST)
find_package( OpenCV REQUIRED )
add_executable(${PROJECT_NAME} ${SRC_LIST})
target_link_libraries( testopencv ${OpenCV_LIBS} )

