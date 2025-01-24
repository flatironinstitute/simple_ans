cmake_minimum_required(VERSION 3.14)
project(simple_ans VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_RELEASE_FLAGS "-O3 -march=native")

# Find pybind11
find_package(pybind11 REQUIRED)

include(FetchContent)
FetchContent_Declare(
  unordered_dense
  GIT_REPOSITORY https://github.com/martinus/unordered_dense.git
  GIT_TAG v4.5.0
  GIT_SHALLOW TRUE)
FetchContent_MakeAvailable(unordered_dense)

# Set C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Add source files
set(SOURCES
    simple_ans/cpp/simple_ans.cpp
    simple_ans_bind.cpp
)

# Add header files
set(HEADERS
    simple_ans/cpp/simple_ans.hpp
)

# Create Python module
pybind11_add_module(_simple_ans ${SOURCES} ${HEADERS})

# Set include directories
target_include_directories(_simple_ans
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/simple_ans/cpp
)
target_link_libraries(_simple_ans PRIVATE unordered_dense)

# Set compiler flags
if(MSVC)
    target_compile_options(_simple_ans PRIVATE /W4)
else()
    target_compile_options(_simple_ans PRIVATE -Wall -Wextra -Wpedantic)
endif()

# Install rules
include(GNUInstallDirs)
install(TARGETS _simple_ans
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
install(FILES ${HEADERS}
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/simple_ans
)

# For editable installs, copy the extension module to the source directory
add_custom_command(TARGET _simple_ans POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:_simple_ans> ${CMAKE_SOURCE_DIR}/simple_ans/
    COMMENT "Copying extension module to source directory for editable install"
)
