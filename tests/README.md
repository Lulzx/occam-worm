# tests/

Test suites. See the test matrix in the spec.

- `unit/` — fast unit tests for the Python package (`pytest`); no network access.
- [`synthetic_truth/`](synthetic_truth/README.md) — Known generators for recovery tests: leaky neuron, E/I pairs, delayed chains, adaptation, oscillation, misspecification, non-identifiability, carryover and false-sharing cases.
- `cpp/` — C++ unit and property tests for `libs/` (in-repo harness, run by CTest).
- [`conformance/`](conformance/README.md) — 3–8-neuron graphs that define correct behavior. The C++ reference, the independent scalar Python implementation and the differentiable simulator must all agree.
- [`adversarial/`](adversarial/README.md) — Deliberate attempts to falsify the preferred model family (lookup-table memorization, filter absorption, selection inflation).
- [`leakage/`](leakage/README.md) — Automated leakage assertions. CI injects a leaked animal and must fail.
- [`fuzz/`](fuzz/README.md) — libFuzzer targets for the WRL parser and canonicalizer.
- [`integration/`](integration/README.md) — End-to-end runs from raw manifests to reports on small fixtures.

Spec: [§16](../docs/evaluation/TESTING.md)

Status: C++ tests, conformance suite (C++ reference only) and the parser fuzz target are implemented; the other directories are not.
