project(testopencv)
INCLUDE_DIRECTORIES(/usr/local/inlude)
cmake_minimum_required(VERSION 2.8)
SET(SRC_LIST main.cpp cholesky.cpp)
find_package( OpenCV REQUIRED )
add_executable(${PROJECT_NAME} ${SRC_LIST})
target_link_libraries( testopencv ${OpenCV_LIBS} )

