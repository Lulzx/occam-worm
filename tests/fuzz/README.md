# tests/fuzz/

libFuzzer targets for the WRL parser and canonicalizer.

Tickets: [OW-008](../../docs/planning/tickets/OW-008.md) · Spec: [§16.5](../../docs/evaluation/TESTING.md)

Status: libFuzzer target `fuzz_parser.cpp` for the WRL front end (parser, checker, canonicaliser, IR JSON), built only with the `fuzz` preset; it checks that accepted programs canonicalise idempotently. Commands: [REPRODUCIBILITY.md](../../docs/engineering/REPRODUCIBILITY.md#local-build-and-test). A deterministic mutation test (`tests/cpp/test_robustness.cpp`) runs in every CTest build.
