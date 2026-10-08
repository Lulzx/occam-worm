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
