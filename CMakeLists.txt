cmake_minimum_required(VERSION 3.15)
project(simple_ans LANGUAGES CXX)

# Check if we're on Mac with Apple Silicon
if(APPLE)
  execute_process(COMMAND uname -m OUTPUT_VARIABLE ARCH OUTPUT_STRIP_TRAILING_WHITESPACE)
  if(ARCH STREQUAL "arm64")
    set(ARCH_FLAGS "-arch arm64" CACHE STRING "Architecture flags for the compiler")
    # Explicitly set platform flags for Apple Silicon
    set(CMAKE_OSX_ARCHITECTURES "arm64" CACHE STRING "Build architectures for macOS" FORCE)
    
    # Completely remove any -march=native flags
    string(REGEX REPLACE "-march=native" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
    string(REGEX REPLACE "-march=native" "" CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")
    string(REGEX REPLACE "-march=native" "" CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")
    
    # Set optimization flags without -march=native
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 ${ARCH_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${ARCH_FLAGS}")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${ARCH_FLAGS}")
  else()
    set(ARCH_FLAGS "-march=native" CACHE STRING "Architecture flags for the compiler")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 ${ARCH_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${ARCH_FLAGS}")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${ARCH_FLAGS}")
  endif()
else()
  set(ARCH_FLAGS "-march=native" CACHE STRING "Architecture flags for the compiler")
  set(CMAKE_CXX_FLAGS_RELEASE "-O3 ${ARCH_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${ARCH_FLAGS}")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${ARCH_FLAGS}")
endif()

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
  emhash
  GIT_REPOSITORY https://github.com/DiamonDinoia/emhash.git
  GIT_TAG master
  GIT_SHALLOW TRUE)

# Only populate emhash, don't build it
FetchContent_GetProperties(emhash)
if(NOT emhash_POPULATED)
  FetchContent_Populate(emhash)
endif()

add_library(emhash INTERFACE)
target_include_directories(emhash INTERFACE ${emhash_SOURCE_DIR})
target_compile_features(emhash INTERFACE cxx_std_17)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
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

# Set target-specific properties
if(APPLE AND ARCH STREQUAL "arm64")
  set_target_properties(_simple_ans PROPERTIES
    COMPILE_FLAGS "-arch arm64"
    LINK_FLAGS "-arch arm64")
endif()

# Set include directories
target_include_directories(_simple_ans
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/simple_ans/cpp
)
target_link_libraries(_simple_ans PRIVATE emhash)

# Set compiler flags
if(MSVC)
    target_compile_options(_simple_ans PRIVATE /W4)
else()
    # Apply standard warning flags
    target_compile_options(_simple_ans PRIVATE -Wall -Wextra -Wpedantic -Wno-unknown-pragmas)
    
    # Apply architecture flags (but only if on Apple Silicon, handle the flag differently)
    if(APPLE AND ARCH STREQUAL "arm64")
        target_compile_options(_simple_ans PRIVATE -arch arm64)
    elseif(NOT APPLE OR NOT ARCH STREQUAL "arm64")
        target_compile_options(_simple_ans PRIVATE -march=native)
    endif()
endif()

# Remove -march=native from target properties if on Apple Silicon
if(APPLE AND ARCH STREQUAL "arm64")
    get_target_property(target_compile_options _simple_ans COMPILE_OPTIONS)
    if(target_compile_options)
        string(REGEX REPLACE "-march=native" "" target_compile_options "${target_compile_options}")
        set_target_properties(_simple_ans PROPERTIES COMPILE_OPTIONS "${target_compile_options}")
    endif()
endif()

if(ipo_supported AND (CMAKE_BUILD_TYPE STREQUAL "Release"))
  set_target_properties(_simple_ans PROPERTIES INTERPROCEDURAL_OPTIMIZATION TRUE)
endif()


# Install rules
install(TARGETS _simple_ans LIBRARY DESTINATION simple_ans)
