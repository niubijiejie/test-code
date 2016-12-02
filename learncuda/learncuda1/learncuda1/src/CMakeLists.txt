set(LIBRARY_OUTPUT_PATH, ${PROJECT_SOURCE_DIR}/srclib)
find_package(GLUT REQUIRED)
find_package(OpenGL REQUIRED)

if(OPENGL_FOUND)
  include_directories(${OPENGL_INCLUDE_DIR})
endif()

if(GLUT_FOUND)
  include_directories(${GLUT_INCLUDE_DIR})
endif()

IF(WITH_CUDA)
  include_directories(${CUDA_INCLUDE_DIRS})
  #include_directories(/usr/include)
  #include_directories(${OPENGL_INCLUDE_DIRS})
  cuda_add_library(src
     book.h
     gl_helper.h
     texture2D.h
    texture2D.cu
    cpu_anim.h
    )
  target_link_libraries(src ${CUDA_LIBRARY}   ${OPENGL_LIBRARIES}
  ${GLUT_glut_LIBRARY} -pthread)
ENDIF()

