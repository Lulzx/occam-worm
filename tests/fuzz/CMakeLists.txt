# libFuzzer target; only built when OW_ENABLE_FUZZ=ON under Clang (see the `fuzz` preset).
add_executable(fuzz_parser fuzz_parser.cpp)
target_link_libraries(fuzz_parser PRIVATE ow_ir ow_core -fsanitize=fuzzer)
