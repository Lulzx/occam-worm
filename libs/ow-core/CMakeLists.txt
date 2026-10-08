add_library(ow_core STATIC
  src/error.cpp
  src/sha256.cpp
  src/json.cpp
  src/units.cpp
  src/graph.cpp
  src/numfmt.cpp)
target_include_directories(ow_core PUBLIC include ${CMAKE_BINARY_DIR}/generated)
