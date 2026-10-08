add_library(ow_sim STATIC
  src/sim.cpp
  src/conformance.cpp)
target_include_directories(ow_sim PUBLIC include)
target_link_libraries(ow_sim PUBLIC ow_ir ow_core)
