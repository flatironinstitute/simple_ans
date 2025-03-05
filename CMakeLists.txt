cmake_minimum_required(VERSION 3.15)
project(simple_ans LANGUAGES CXX)

# Detect architecture and set flags
if(APPLE)
  execute_process(COMMAND uname -m OUTPUT_VARIABLE ARCH OUTPUT_STRIP_TRAILING_WHITESPACE)
  if(ARCH STREQUAL "arm64")
    # Apple Silicon (M1/M2) configuration
    set(ARCH_FLAGS "-arch arm64")
    set(CMAKE_OSX_ARCHITECTURES "arm64" CACHE STRING "Build architecture for macOS" FORCE)
    set(IS_ARM_MAC TRUE)
  else()
    # Intel Mac configuration
    set(ARCH_FLAGS "-march=native")
    set(IS_ARM_MAC FALSE)
  endif()
else()
  # Non-Mac configuration (Linux, Windows)
  set(ARCH_FLAGS "-march=native")
  set(IS_ARM_MAC FALSE)
endif()

# Set compiler flags
set(CMAKE_CXX_FLAGS_RELEASE "-O3 ${ARCH_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${ARCH_FLAGS}")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${ARCH_FLAGS}")

# Link-time optimization setup
include(CheckIPOSupported)
check_ipo_supported(RESULT ipo_supported OUTPUT ipo_error)
if(ipo_supported)
  message(STATUS "IPO/LTO is supported")
else()
  message(WARNING "IPO/LTO is not supported: ${ipo_error}")
endif()

# Find pybind11
find_package(pybind11 REQUIRED)

# Get emhash (header-only library)
include(FetchContent)
FetchContent_Declare(
  emhash
  GIT_REPOSITORY https://github.com/DiamonDinoia/emhash.git
  GIT_TAG master
  GIT_SHALLOW TRUE
)
FetchContent_GetProperties(emhash)
if(NOT emhash_POPULATED)
  FetchContent_Populate(emhash)
endif()

# Setup emhash as interface library
add_library(emhash INTERFACE)
target_include_directories(emhash INTERFACE ${emhash_SOURCE_DIR})
target_compile_features(emhash INTERFACE cxx_std_17)

# Configure C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Define source files
set(SOURCES
  simple_ans/cpp/simple_ans.cpp
  simple_ans_bind.cpp
)

set(HEADERS
  simple_ans/cpp/simple_ans.hpp
)

# Create Python module
pybind11_add_module(_simple_ans ${SOURCES} ${HEADERS})

# Configure include directories
target_include_directories(_simple_ans PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/simple_ans/cpp)
target_link_libraries(_simple_ans PRIVATE emhash)

# Set compiler-specific options
if(MSVC)
  target_compile_options(_simple_ans PRIVATE /W4)
else()
  # Standard warning flags
  target_compile_options(_simple_ans PRIVATE -Wall -Wextra -Wpedantic -Wno-unknown-pragmas)
  
  # Architecture-specific flags
  if(IS_ARM_MAC)
    target_compile_options(_simple_ans PRIVATE -arch arm64)
    set_target_properties(_simple_ans PROPERTIES
      COMPILE_FLAGS "-arch arm64"
      LINK_FLAGS "-arch arm64"
    )
  else()
    target_compile_options(_simple_ans PRIVATE -march=native)
  endif()
endif()

# Enable LTO in release mode
if(ipo_supported AND (CMAKE_BUILD_TYPE STREQUAL "Release"))
  set_target_properties(_simple_ans PROPERTIES INTERPROCEDURAL_OPTIMIZATION TRUE)
endif()

# Installation
install(TARGETS _simple_ans LIBRARY DESTINATION simple_ans)
