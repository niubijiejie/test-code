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
CMAKE_EDIT_COMMAND = /usr/bin/cmake-gui

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/hwj/desktop/qt/testcuda/testcuda

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/hwj/desktop/qt/testcuda/testcuda-build

# Include any dependencies generated for this target.
include CMakeFiles/testcuda.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/testcuda.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/testcuda.dir/flags.make

CMakeFiles/testcuda.dir/main.cpp.o: CMakeFiles/testcuda.dir/flags.make
CMakeFiles/testcuda.dir/main.cpp.o: /home/hwj/desktop/qt/testcuda/testcuda/main.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/hwj/desktop/qt/testcuda/testcuda-build/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/testcuda.dir/main.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/testcuda.dir/main.cpp.o -c /home/hwj/desktop/qt/testcuda/testcuda/main.cpp

CMakeFiles/testcuda.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/testcuda.dir/main.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/hwj/desktop/qt/testcuda/testcuda/main.cpp > CMakeFiles/testcuda.dir/main.cpp.i

CMakeFiles/testcuda.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/testcuda.dir/main.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/hwj/desktop/qt/testcuda/testcuda/main.cpp -o CMakeFiles/testcuda.dir/main.cpp.s

CMakeFiles/testcuda.dir/main.cpp.o.requires:
.PHONY : CMakeFiles/testcuda.dir/main.cpp.o.requires

CMakeFiles/testcuda.dir/main.cpp.o.provides: CMakeFiles/testcuda.dir/main.cpp.o.requires
	$(MAKE) -f CMakeFiles/testcuda.dir/build.make CMakeFiles/testcuda.dir/main.cpp.o.provides.build
.PHONY : CMakeFiles/testcuda.dir/main.cpp.o.provides

CMakeFiles/testcuda.dir/main.cpp.o.provides.build: CMakeFiles/testcuda.dir/main.cpp.o

# Object files for target testcuda
testcuda_OBJECTS = \
"CMakeFiles/testcuda.dir/main.cpp.o"

# External object files for target testcuda
testcuda_EXTERNAL_OBJECTS =

testcuda: CMakeFiles/testcuda.dir/main.cpp.o
testcuda: CMakeFiles/testcuda.dir/build.make
testcuda: CMakeFiles/testcuda.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX executable testcuda"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/testcuda.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/testcuda.dir/build: testcuda
.PHONY : CMakeFiles/testcuda.dir/build

CMakeFiles/testcuda.dir/requires: CMakeFiles/testcuda.dir/main.cpp.o.requires
.PHONY : CMakeFiles/testcuda.dir/requires

CMakeFiles/testcuda.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/testcuda.dir/cmake_clean.cmake
.PHONY : CMakeFiles/testcuda.dir/clean

CMakeFiles/testcuda.dir/depend:
	cd /home/hwj/desktop/qt/testcuda/testcuda-build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/hwj/desktop/qt/testcuda/testcuda /home/hwj/desktop/qt/testcuda/testcuda /home/hwj/desktop/qt/testcuda/testcuda-build /home/hwj/desktop/qt/testcuda/testcuda-build /home/hwj/desktop/qt/testcuda/testcuda-build/CMakeFiles/testcuda.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/testcuda.dir/depend

