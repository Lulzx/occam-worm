# Reproducibility, artifact identity and C++ toolchain

> Part of the **Occam's Worm specification v0.4.1** · [Spec index](../README.md) · Source: §14.7–§14.8

Determinism rules for the simulator itself (RNG, seeding, distributions) are in [§6.6](../runtime/SIM_SEMANTICS.md).

## 14.7 Artifact identity and immutability

Use content-addressed run IDs derived from hashes of:

```text
(raw manifest + normalized schema + split IDs + graph version
 + program hash + fitting configuration + source code revision + RNG seeds)
```

Avoid mutable `latest` pointers in publication artifacts. Analysis notebooks may link immutable runs, but the report generator must never silently recompute against a changed dataset.

## 14.8 C++ toolchain and conventions

These defaults are proposals to freeze in `docs/REPRODUCIBILITY.md` before confirmatory runs.

- **Standard and build:** C++26 (`-std=c++26`); CMake with presets; dependencies pinned through a vcpkg manifest with a fixed baseline. The Python extension is built with scikit-build-core, so one `pip install -e .` produces both the Python package and the compiled module.
- **C++26 feature policy:** compiler support for C++26 is uneven, so only features implemented by *both* pinned compilers (Clang and GCC) may be used. The permitted list lives in this file and a CI build on both compilers enforces it. Candidates, each admitted only once both compilers support it: contracts (`pre`, `post`, `contract_assert`) for the hard constraints of [§5.8](../language/GRAMMAR.md) and interpreter invariants; `std::simd` for the vectorized CPU backend ([§18.2](PERFORMANCE.md)); static reflection for schema and manifest serialization; `#embed` for test fixtures. Apple's Xcode Clang lags upstream, so reference builds use a pinned upstream LLVM rather than the system toolchain.
- **Bindings:** nanobind (pybind11 is an acceptable alternative). Arrays cross the boundary as typed, contiguous buffers with explicit shape and dtype checks and no implicit conversions.
- **Floating-point reproducibility:** the reference build uses float64, disables `-ffast-math` and floating-point contraction (`-ffp-contract=off`), and fixes reduction order. Optimized builds may relax these only behind a declared tolerance checked against the reference build.
- **Determinism hazards:** output and hashes must never depend on unordered-container iteration order, pointer values or thread scheduling. Use a counter-based RNG (for example Philox) with explicitly implemented distributions ([§6.6](../runtime/SIM_SEMANTICS.md)).
- **Safety:** RAII and value types; no owning raw pointers; bounds-checked access in the reference interpreter. CI runs AddressSanitizer and UndefinedBehaviorSanitizer, clang-tidy and warnings-as-errors, and fuzzes the WRL parser ([§16.5](../evaluation/TESTING.md)).
- **Libraries (proposed):** GoogleTest and RapidCheck for unit and property tests; yaml-cpp and nlohmann/json for WRL sources and manifests; a vetted SHA-256 implementation for program and artifact hashes. Arrow/Parquet IO stays in Python unless profiling says otherwise.
- **Compilers:** pin one Clang version for the reference build; also build with GCC in CI to catch non-portable code.

### C++26 feature policy (as implemented, OW-008/009/010)

The code builds with `-std=c++26` (CMake `CXX_STANDARD 26`, `CXX_EXTENSIONS OFF`) on Clang 23 (Homebrew LLVM, libc++) and GCC 16 (libstdc++), and was written so that it needs nothing newer than GCC 14 and Clang 18. **Allowlist actually used:**

| Feature | Where | Minimum compilers |
|---|---|---|
| Placeholder `_` in a structured binding (P2169, C++26) | `libs/ow-search/src/enumerate.cpp` | GCC 14, Clang 18 |
| `static_assert` with a constant-expression message (P2741, C++26) | `libs/ow-core/src/sha256.cpp` | GCC 13, Clang 17 |
| `std::expected`, `std::bit_cast`, `<=>`, `std::ranges` algorithms, `std::erase_if` (C++20/23 library) | `ow-ir`, `ow-search`, `ow-core` | GCC 12 / Clang 16 |

**Evaluated and not used** (not implemented by both compilers, or not needed): contracts (`pre`/`post`/`contract_assert`: GCC 16 only; interpreter invariants use explicit checks that throw `Error`), `std::simd` (no vectorized backend yet), static reflection (not in Clang 23), `#embed` (fixtures are read from disk), `std::inplace_vector`, pack indexing, `= delete("reason")` (Clang 19+), `std::print` (not needed). Language-version-independent rules: no `std::normal_distribution` or other implementation-defined distributions (there is no RNG in G0/G1), no unordered-container iteration in anything that feeds output or hashes, `std::to_chars` for number formatting (identical shortest round-trip digits in both standard libraries), `strtod` for parsing after the grammar has been validated by hand.

### Floating-point policy (as implemented)

`-ffp-contract=off -fno-fast-math`, float64 only. All operators of the reference interpreter use IEEE-754 basic operations in a fixed order except `tanh` and `exp` (libm, agreement to about 1 ulp across platforms). Constant folding in the canonicaliser never evaluates libm functions, so program hashes are platform independent. The exact arithmetic is specified in [WRL_SYNTAX.md](../language/WRL_SYNTAX.md) §6.

### Local build and test

GitHub Actions is not used for the C++ code at present; verify locally. Presets (`CMakePresets.json`) are `clang` (reference, Release), `gcc` (Release), `asan` (Clang Debug, AddressSanitizer + UndefinedBehaviorSanitizer) and `fuzz` (libFuzzer parser target). They name `clang++`/`g++` from `PATH`; override the compiler on the command line. On macOS with Homebrew:

```bash
cmake --preset clang -DCMAKE_CXX_COMPILER=/opt/homebrew/opt/llvm/bin/clang++
cmake --build --preset clang && ctest --preset clang

cmake --preset gcc   -DCMAKE_CXX_COMPILER=/opt/homebrew/bin/g++-16
cmake --build --preset gcc && ctest --preset gcc

cmake --preset asan  -DCMAKE_CXX_COMPILER=/opt/homebrew/opt/llvm/bin/clang++
cmake --build --preset asan && ctest --preset asan
```

Each build lands in `build/<preset>/`; the binary is `build/<preset>/libs/ow-cli/ow`. CTest runs the unit tests (SHA-256 FIPS vectors, core, IR, simulator, enumerator, robustness), the conformance suite (`ow sim conformance --suite tests/conformance`), the example rules (the rule in `configs/rules/rejected/` must be rejected) and both enumeration configs. The Python simulators and fitting tests (`pytest`) consume the IR that `build/clang/libs/ow-cli/ow` prints (override the path with `OW_CLI`). Tests that need the binary are skipped with a reason when it is missing; the conformance traces still run from the committed IR cache `tests/conformance/ir_cache/` (refresh it with `OW_UPDATE_IR_CACHE=1 pytest tests/conformance` after a compiler change). Fuzzing the parser (Clang only; the target is built only when `OW_ENABLE_FUZZ=ON`):

```bash
cmake --preset fuzz -DCMAKE_CXX_COMPILER=/opt/homebrew/opt/llvm/bin/clang++
cmake --build --preset fuzz --target fuzz_parser
mkdir -p build/fuzz-corpus && cp configs/rules/*.wrl build/fuzz-corpus/
build/fuzz/tests/fuzz/fuzz_parser build/fuzz-corpus -max_total_time=60 -max_len=4096
```
