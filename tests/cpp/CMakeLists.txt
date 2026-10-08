add_library(ow_test_main STATIC ow_test_main.cpp)
target_include_directories(ow_test_main PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(ow_test_main PUBLIC ow_core)

# add_ow_test(<name> <libs...>) builds tests/cpp/<name>.cpp and registers it with CTest.
function(add_ow_test name)
  add_executable(${name} ${name}.cpp)
  target_link_libraries(${name} PRIVATE ow_test_main ${ARGN})
  target_compile_definitions(${name} PRIVATE OW_SOURCE_DIR="${PROJECT_SOURCE_DIR}")
  add_test(NAME ${name} COMMAND ${name})
endfunction()

add_ow_test(test_sha256 ow_core)
add_ow_test(test_core ow_core)
