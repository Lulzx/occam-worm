add_library(ow_search STATIC src/enumerate.cpp)
target_include_directories(ow_search PUBLIC include)
target_link_libraries(ow_search PUBLIC ow_ir ow_core)
