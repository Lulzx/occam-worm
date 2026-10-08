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
add_ow_test(test_ir ow_ir)
add_ow_test(test_sim ow_sim)
add_ow_test(test_search ow_search)

# Command-line checks: conformance suite, example rules (the rejected one must fail) and enumeration.
add_test(NAME conformance_suite COMMAND ow sim conformance --suite ${PROJECT_SOURCE_DIR}/tests/conformance)
foreach(rule leak-adapt leak-adapt-euler gap-leak rule90)
  add_test(NAME rule_inspect_${rule} COMMAND ow rule inspect ${PROJECT_SOURCE_DIR}/configs/rules/${rule}.wrl)
endforeach()
add_test(NAME rule_rejected_unstable_euler COMMAND ow rule inspect ${PROJECT_SOURCE_DIR}/configs/rules/rejected/leak-euler-unstable.wrl)
set_tests_properties(rule_rejected_unstable_euler PROPERTIES WILL_FAIL TRUE)
foreach(search g0-tiny g1-tiny)
  add_test(NAME search_enumerate_${search} COMMAND ow search enumerate --config ${PROJECT_SOURCE_DIR}/configs/searches/${search}.json)
endforeach()
add_ow_test(test_robustness ow_ir)
