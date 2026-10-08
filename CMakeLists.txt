cmake_minimum_required(VERSION 3.25)
project(occamworm VERSION 0.1.0 LANGUAGES CXX)

# C++26 reference toolchain (docs/engineering/REPRODUCIBILITY.md). Only features implemented by both
# Clang and GCC are used; the allowlist lives in that document.
set(CMAKE_CXX_STANDARD 26)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(OW_ENABLE_FUZZ "Build the libFuzzer parser target (Clang only)" OFF)
option(OW_SANITIZE "Build with AddressSanitizer and UndefinedBehaviorSanitizer" OFF)
option(OW_WERROR "Treat warnings as errors" ON)

# Floating-point reproducibility (§14.8): float64 throughout, no fast-math, no FMA contraction.
add_compile_options(-ffp-contract=off -fno-fast-math)
add_compile_options(-Wall -Wextra -Wpedantic -Wshadow -Wconversion -Wsign-conversion)
if(OW_WERROR)
  add_compile_options(-Werror)
endif()

if(OW_SANITIZE)
  add_compile_options(-fsanitize=address,undefined -fno-sanitize-recover=undefined -fno-omit-frame-pointer)
  add_link_options(-fsanitize=address,undefined)
endif()

# Build identity recorded with every result (docs/language/WRL_SYNTAX.md, "Provenance").
set(OW_BUILD_STRING
    "ow ${PROJECT_VERSION}; ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}; C++${CMAKE_CXX_STANDARD}; ${CMAKE_BUILD_TYPE}; fp-contract=off"
)
configure_file(libs/ow-core/build_info.hpp.in ${CMAKE_BINARY_DIR}/generated/occamworm/core/build_info.hpp @ONLY)

include(CTest)

add_subdirectory(libs/ow-core)

if(BUILD_TESTING)
  add_subdirectory(tests/cpp)
endif()

if(OW_ENABLE_FUZZ)
  if(NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    message(FATAL_ERROR "OW_ENABLE_FUZZ requires Clang (libFuzzer)")
  endif()
  add_subdirectory(tests/fuzz)
endif()
