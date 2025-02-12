cmake_minimum_required(VERSION 3.15)
project(simple_ans LANGUAGES CXX)

set(ARCH_FLAGS "-march=native" CACHE STRING "Architecture flags for the compiler")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 ${ARCH_FLAGS}")

include(CheckIPOSupported)
check_ipo_supported(RESULT ipo_supported OUTPUT ipo_error)
if (ipo_supported)
  message(STATUS "IPO/LTO is supported")
else()
  message(WARNING "IPO/LTO is not supported: ${ipo_error}")
endif()

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

if(ipo_supported AND (CMAKE_BUILD_TYPE STREQUAL "Release"))
  set_target_properties(_simple_ans PROPERTIES INTERPROCEDURAL_OPTIMIZATION TRUE)
endif()


# Install rules
install(TARGETS _simple_ans LIBRARY DESTINATION simple_ans)
