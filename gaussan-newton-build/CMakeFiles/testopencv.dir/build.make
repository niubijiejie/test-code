# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 2.8

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list

# Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The program to use to edit the cache.
CMAKE_EDIT_COMMAND = /usr/bin/ccmake

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/hwj/desktop/qt/gaussan-newton

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/hwj/desktop/qt/gaussan-newton-build

# Include any dependencies generated for this target.
include CMakeFiles/testopencv.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/testopencv.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/testopencv.dir/flags.make

CMakeFiles/testopencv.dir/main.cpp.o: CMakeFiles/testopencv.dir/flags.make
CMakeFiles/testopencv.dir/main.cpp.o: /home/hwj/desktop/qt/gaussan-newton/main.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/hwj/desktop/qt/gaussan-newton-build/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/testopencv.dir/main.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/testopencv.dir/main.cpp.o -c /home/hwj/desktop/qt/gaussan-newton/main.cpp

CMakeFiles/testopencv.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/testopencv.dir/main.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/hwj/desktop/qt/gaussan-newton/main.cpp > CMakeFiles/testopencv.dir/main.cpp.i

CMakeFiles/testopencv.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/testopencv.dir/main.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/hwj/desktop/qt/gaussan-newton/main.cpp -o CMakeFiles/testopencv.dir/main.cpp.s

CMakeFiles/testopencv.dir/main.cpp.o.requires:
.PHONY : CMakeFiles/testopencv.dir/main.cpp.o.requires

CMakeFiles/testopencv.dir/main.cpp.o.provides: CMakeFiles/testopencv.dir/main.cpp.o.requires
	$(MAKE) -f CMakeFiles/testopencv.dir/build.make CMakeFiles/testopencv.dir/main.cpp.o.provides.build
.PHONY : CMakeFiles/testopencv.dir/main.cpp.o.provides

CMakeFiles/testopencv.dir/main.cpp.o.provides.build: CMakeFiles/testopencv.dir/main.cpp.o

CMakeFiles/testopencv.dir/cholesky.cpp.o: CMakeFiles/testopencv.dir/flags.make
CMakeFiles/testopencv.dir/cholesky.cpp.o: /home/hwj/desktop/qt/gaussan-newton/cholesky.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/hwj/desktop/qt/gaussan-newton-build/CMakeFiles $(CMAKE_PROGRESS_2)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/testopencv.dir/cholesky.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/testopencv.dir/cholesky.cpp.o -c /home/hwj/desktop/qt/gaussan-newton/cholesky.cpp

CMakeFiles/testopencv.dir/cholesky.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/testopencv.dir/cholesky.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/hwj/desktop/qt/gaussan-newton/cholesky.cpp > CMakeFiles/testopencv.dir/cholesky.cpp.i

CMakeFiles/testopencv.dir/cholesky.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/testopencv.dir/cholesky.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/hwj/desktop/qt/gaussan-newton/cholesky.cpp -o CMakeFiles/testopencv.dir/cholesky.cpp.s

CMakeFiles/testopencv.dir/cholesky.cpp.o.requires:
.PHONY : CMakeFiles/testopencv.dir/cholesky.cpp.o.requires

CMakeFiles/testopencv.dir/cholesky.cpp.o.provides: CMakeFiles/testopencv.dir/cholesky.cpp.o.requires
	$(MAKE) -f CMakeFiles/testopencv.dir/build.make CMakeFiles/testopencv.dir/cholesky.cpp.o.provides.build
.PHONY : CMakeFiles/testopencv.dir/cholesky.cpp.o.provides

CMakeFiles/testopencv.dir/cholesky.cpp.o.provides.build: CMakeFiles/testopencv.dir/cholesky.cpp.o

# Object files for target testopencv
testopencv_OBJECTS = \
"CMakeFiles/testopencv.dir/main.cpp.o" \
"CMakeFiles/testopencv.dir/cholesky.cpp.o"

# External object files for target testopencv
testopencv_EXTERNAL_OBJECTS =

testopencv: CMakeFiles/testopencv.dir/main.cpp.o
testopencv: CMakeFiles/testopencv.dir/cholesky.cpp.o
testopencv: CMakeFiles/testopencv.dir/build.make
testopencv: /usr/lib/x86_64-linux-gnu/libopencv_videostab.so.2.4.8
testopencv: /usr/lib/x86_64-linux-gnu/libopencv_video.so.2.4.8
testopencv: /usr/lib/x86_64-linux-gnu/libopencv_ts.so.2.4.8
testopencv: /usr/lib/x86_64-linux-gnu/libopencv_superres.so.2.4.8
testopencv: /usr/lib/x86_64-linux-gnu/libopencv_stitching.so.2.4.8
testopencv: /usr/lib/x86_64-linux-gnu/libopencv_photo.so.2.4.8
testopencv: /usr/lib/x86_64-linux-gnu/libopencv_ocl.so.2.4.8
testopencv: /usr/lib/x86_64-linux-gnu/libopencv_objdetect.so.2.4.8
testopencv: /usr/lib/x86_64-linux-gnu/libopencv_ml.so.2.4.8
testopencv: /usr/lib/x86_64-linux-gnu/libopencv_legacy.so.2.4.8
testopencv: /usr/lib/x86_64-linux-gnu/libopencv_imgproc.so.2.4.8
testopencv: /usr/lib/x86_64-linux-gnu/libopencv_highgui.so.2.4.8
testopencv: /usr/lib/x86_64-linux-gnu/libopencv_gpu.so.2.4.8
testopencv: /usr/lib/x86_64-linux-gnu/libopencv_flann.so.2.4.8
testopencv: /usr/lib/x86_64-linux-gnu/libopencv_features2d.so.2.4.8
testopencv: /usr/lib/x86_64-linux-gnu/libopencv_core.so.2.4.8
testopencv: /usr/lib/x86_64-linux-gnu/libopencv_contrib.so.2.4.8
testopencv: /usr/lib/x86_64-linux-gnu/libopencv_calib3d.so.2.4.8
testopencv: /usr/lib/x86_64-linux-gnu/libopencv_photo.so.2.4.8
testopencv: /usr/lib/x86_64-linux-gnu/libopencv_legacy.so.2.4.8
testopencv: /usr/lib/x86_64-linux-gnu/libopencv_video.so.2.4.8
testopencv: /usr/lib/x86_64-linux-gnu/libopencv_objdetect.so.2.4.8
testopencv: /usr/lib/x86_64-linux-gnu/libopencv_ml.so.2.4.8
testopencv: /usr/lib/x86_64-linux-gnu/libopencv_calib3d.so.2.4.8
testopencv: /usr/lib/x86_64-linux-gnu/libopencv_features2d.so.2.4.8
testopencv: /usr/lib/x86_64-linux-gnu/libopencv_highgui.so.2.4.8
testopencv: /usr/lib/x86_64-linux-gnu/libopencv_imgproc.so.2.4.8
testopencv: /usr/lib/x86_64-linux-gnu/libopencv_flann.so.2.4.8
testopencv: /usr/lib/x86_64-linux-gnu/libopencv_core.so.2.4.8
testopencv: CMakeFiles/testopencv.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX executable testopencv"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/testopencv.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/testopencv.dir/build: testopencv
.PHONY : CMakeFiles/testopencv.dir/build

CMakeFiles/testopencv.dir/requires: CMakeFiles/testopencv.dir/main.cpp.o.requires
CMakeFiles/testopencv.dir/requires: CMakeFiles/testopencv.dir/cholesky.cpp.o.requires
.PHONY : CMakeFiles/testopencv.dir/requires

CMakeFiles/testopencv.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/testopencv.dir/cmake_clean.cmake
.PHONY : CMakeFiles/testopencv.dir/clean

CMakeFiles/testopencv.dir/depend:
	cd /home/hwj/desktop/qt/gaussan-newton-build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/hwj/desktop/qt/gaussan-newton /home/hwj/desktop/qt/gaussan-newton /home/hwj/desktop/qt/gaussan-newton-build /home/hwj/desktop/qt/gaussan-newton-build /home/hwj/desktop/qt/gaussan-newton-build/CMakeFiles/testopencv.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/testopencv.dir/depend

