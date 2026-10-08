add_library(ow_ir STATIC
  src/op.cpp
  src/parser.cpp
  src/check.cpp
  src/canonical.cpp
  src/bits.cpp
  src/source.cpp
  src/compile.cpp)
target_include_directories(ow_ir PUBLIC include)
target_link_libraries(ow_ir PUBLIC ow_core)
