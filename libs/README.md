# libs/ — C++26 libraries

C++26 libraries in namespace `occamworm::`. C++ owns the rule toolchain, enumeration, the reference interpreter and the CLI; Python owns data, baselines, differentiable simulation and fitting.

| Library | Purpose |
| --- | --- |
| [`ow-core`](ow-core/README.md) | Typed graphs, units and neural state definitions shared by every other library. |
| [`ow-ir`](ow-ir/README.md) | WRL front end: parser, typed AST, unit checks, canonical IR, `program_hash = SHA256(canonical_IR)` and bit costs (`L_struct`, `L_params`). |
| [`ow-sim`](ow-sim/README.md) | Scalar, bounds-checked, deterministic reference interpreter. This is the correctness oracle for every other backend. |
| [`ow-sim-fast`](ow-sim-fast/README.md) | Vectorized CPU runtime with per-shape specialized kernels. Built **only** if profiling shows Python rollouts limit search. |
| [`ow-observe`](ow-observe/README.md) | Calcium and other readout operators, including the separately parameterized indicator stage. |
| [`ow-likelihood`](ow-likelihood/README.md) | Masked, correlated-noise predictive scores (AR residuals, state-space or block scoring). |
| [`ow-search`](ow-search/README.md) | Program enumeration under bit budgets, structured mutation, canonical cache and Pareto ranking. |
| [`ow-infer`](ow-infer/README.md) | Program posterior bookkeeping. Parameter fitting itself lives in Python (`python/occamworm/fit`). |
| [`ow-experiments`](ow-experiments/README.md) | Perturbation menus and expected-information-gain ranking of discriminating experiments. |
| [`ow-equivalence`](ow-equivalence/README.md) | Empirical, observer-relative equivalence between candidate models. |
| [`ow-data`](ow-data/README.md) | Schemas, validators and manifests for the C++ side. |
| [`ow-bindings`](ow-bindings/README.md) | nanobind Python bindings, built through scikit-build-core. |
| [`ow-cli`](ow-cli/README.md) | The `occamworm` command-line front end. Command names and paths in the spec are acceptance contracts. |

Spec: [§14.2](../docs/engineering/ARCHITECTURE.md), [§14.8](../docs/engineering/REPRODUCIBILITY.md)

Status: not implemented.
