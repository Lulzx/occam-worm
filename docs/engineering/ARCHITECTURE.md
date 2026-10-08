# Software architecture

> Part of the **Occam's Worm specification v0.4.1** · [Spec index](../README.md) · Source: §14.1–§14.3, §14.5

## 14. Software architecture and repository design


### 14.1 Design principles

1. **Local-first, reproducible, offline execution:** no hosted dependency is required for core simulation and analysis.
2. **Strong separations of concern:** ingestion, program compilation, simulation, parameter inference, selection, evaluation and visualization have separate APIs.
3. **Immutable evidence:** raw datasets and frozen validation targets cannot be edited by fitting/search code.
4. **Explicit provenance:** every output traces to source version, split, model hash, code version and config.
5. **Scientific transparency:** no silent filtering, calibration, fallback or edge completion.
6. **Composable biological fidelity:** each extra modality must improve a declared scientific endpoint.
7. **No premature GPU requirement:** prioritize correctness and profiling before accelerated search.
8. **Language follows the bottleneck:** prototype numerics in Python; move a component to C++ when profiling or determinism requirements justify it.

### 14.2 Proposed monorepo

```text
occamworm/
├── README.md
├── LICENSE
├── CITATION.cff
├── CMakeLists.txt
├── CMakePresets.json                # pinned compiler, flags and build types
├── vcpkg.json                       # pinned C++ dependencies (fixed baseline)
├── .clang-format
├── .clang-tidy
├── pyproject.toml                   # scikit-build-core builds the C++ extension
├── docs/
│   ├── SPEC.md                      # this document
│   ├── DATA_CONTRACT.md
│   ├── GRAMMAR.md
│   ├── SIM_SEMANTICS.md
│   ├── BENCHMARK.md
│   ├── VALIDATION_PLAN.md
│   ├── REPRODUCIBILITY.md
│   └── EXPERIMENT_REGISTRY.md
├── libs/                            # C++26 libraries, namespace occamworm::
│   ├── ow-core/                    # typed graphs, units, state definitions
│   ├── ow-ir/                      # WRL AST, parser, canonical IR, hash, bit-cost
│   ├── ow-sim/                     # scalar deterministic reference interpreter (oracle)
│   ├── ow-sim-fast/                # vectorized CPU runtime, only after profiling (§18.1)
│   ├── ow-observe/                 # calcium and other readout operators
│   ├── ow-likelihood/              # masked correlated-noise scores
│   ├── ow-search/                  # enumeration, mutation, Pareto ranking
│   ├── ow-infer/                   # program posterior bookkeeping (fitting lives in Python)
│   ├── ow-experiments/             # perturbations and expected information gain
│   ├── ow-equivalence/             # empirical observational equivalence
│   ├── ow-data/                    # schemas, validators, manifests
│   ├── ow-bindings/                # nanobind Python bindings
│   └── ow-cli/                     # command-line frontend
├── python/
│   └── occamworm/
│       ├── importers/
│       │   ├── pumpprobe.py
│       │   ├── wormneuroatlas.py
│       │   ├── openworm.py
│       │   └── wormwideweb.py
│       ├── baselines/
│       │   ├── null.py
│       │   ├── shared_kernel.py
│       │   ├── independent_kernel.py
│       │   ├── lowrank_kernel.py
│       │   ├── linear_network.py
│       │   ├── history.py           # stimulation-history terms (§4.1)
│       │   └── indicator.py         # indicator kinetics and control (§4.5)
│       ├── sim/                     # differentiable simulator (JAX or PyTorch)
│       ├── fit/                     # parameter fitting, multi-start, gradient checks
│       ├── analysis/
│       │   ├── splits.py
│       │   ├── uncertainty.py
│       │   └── reporting.py
│       └── plotting/
├── configs/
│   ├── datasets/
│   ├── splits/
│   ├── rules/
│   ├── experiments/
│   └── searches/
├── tests/
│   ├── synthetic_truth/
│   ├── conformance/
│   ├── adversarial/
│   ├── leakage/
│   ├── fuzz/                        # libFuzzer targets for the WRL parser
│   └── integration/
├── benches/
│   ├── forward_runtime/
│   ├── candidate_throughput/
│   └── fitting/
├── notebooks/                      # exploratory only; not canonical pipeline
├── scripts/
│   ├── acquire_sources.py
│   ├── build_manifest.py
│   └── reproduce_figures.py
├── artifacts/                       # generated, immutable run outputs
└── data/
    ├── raw/                         # gitignored, immutable
    ├── normalized/                  # versioned manifests
    └── splits/                      # frozen IDs and checksums
```

C++ handles the rule-language toolchain (parsing, type and unit checks, canonicalization, hashing, bit accounting), program enumeration, the deterministic reference interpreter, the CLI and immutable IO contracts. Python handles data import, baselines, the differentiable simulator, parameter fitting, analysis and visualization. The forward pass moves to C++ (`ow-sim-fast`) only if profiling shows that Python rollouts, not fitting or data loading, limit search throughput. The C++/Python boundary (nanobind) passes typed, contiguous arrays without unnecessary copying where practical.

### 14.3 Components and contracts

| Component | Inputs | Outputs | Critical invariants |
|---|---|---|---|
| Dataset importer | Original scientific files | Normalized tables, metadata | Animal IDs and raw provenance preserved |
| Graph builder | Connectome source + mapping | Typed CSR graph | No invented edges; stable ID map |
| DSL compiler | `WRL` YAML/AST | Canonical IR + hash | Type, stability and units checks |
| Simulator | IR + graph + inputs + RNG | Latent traces | Old-state synchronous semantics |
| Observer | Latent trajectories + acquisition metadata | Predicted measurements | No hidden true output access |
| Parameter fitter | Train trials and program | Parameter posterior/point fit | Train-only calibration |
| Searcher | Grammar + train/inner-val | Candidate population | Search budget recorded |
| Evaluator | Frozen model + outer test inputs/labels | Proper scores and diagnostics | Only scored once after freeze |
| Hypothesis-graph analyzer | Candidate posteriors and intervention menu | Ranked experiments | Prediction uncertainty included |
| Report generator | Frozen run artifacts | Markdown/JSON/figures | Traceable models, splits and evidence |


### 14.5 Input/output interfaces

#### Core C++ interfaces

```cpp
// Illustrative C++26 interfaces; not compiled code.
namespace occamworm {

class NeuralProgram {
public:
    virtual ~NeuralProgram() = default;
    virtual ProgramHash hash() const = 0;
    virtual StateLayout state_layout() const = 0;
    // Reads only the old state; writes only the new state (§6.1).
    virtual void step(const StepContext& context,
                      const State& old_state,
                      State& new_state) const = 0;
};

class ObservationModel {
public:
    virtual ~ObservationModel() = default;
    virtual Prediction predict(const LatentTrace& latent,
                               const Acquisition& acquisition) const = 0;
    virtual double log_prob(const Prediction& prediction,
                            const ObservedTrace& observed) const = 0;
};

class CandidateSearch {
public:
    virtual ~CandidateSearch() = default;
    virtual std::vector<ProgramAst> propose(const SearchHistory& history) = 0;
    virtual void update(std::span<const CandidateEvaluation> evaluated) = 0;
};

class InterventionPlanner {
public:
    virtual ~InterventionPlanner() = default;
    virtual std::vector<RankedExperiment> rank(const HypothesisSet& posterior,
                                               const ExperimentMenu& menu) const = 0;
};

}  // namespace occamworm
```

These virtual interfaces serve the reference path and tooling. Optimized kernels in `ow-sim-fast` use compile-time specialization or generated code instead of virtual dispatch inside the timestep loop ([§18.2](PERFORMANCE.md)).

The runtime **must not expose test labels** in `StepContext`, candidate generation, observation prediction or parameter inference.

#### CLI

```bash
# Acquire/reference upstream data via documented importers.
occamworm data audit --manifest configs/datasets/primary.toml \
  --out artifacts/audit-v1/

# Create immutable, group-stratified animal splits.
occamworm splits build --dataset data/normalized/atlas-v1/ \
  --strategy group-kfold-animal --folds 5 --seed 20261008 \
  --out data/splits/atlas-v1/

# Evaluate strong and simple baselines first.
occamworm baseline benchmark \
  --config configs/experiments/shared-timescales.toml \
  --splits data/splits/atlas-v1/ \
  --out artifacts/baselines-v1/

# Inspect/typecheck/canonicalize a rule.
occamworm rule inspect --input configs/rules/leak-adapt.yaml

# Test small programs against a synthetic oracle.
occamworm sim conformance --suite tests/synthetic_truth/

# Train/search on one frozen split protocol.
occamworm search run --config configs/searches/g1-small.toml \
  --out artifacts/search-g1-v1/

# Evaluation requires a frozen model-selection artifact.
occamworm evaluate locked --selection artifacts/search-g1-v1/frozen.json \
  --splits data/splits/atlas-v1/ --out artifacts/eval-g1-v1/

# Select candidate discriminating interventions.
occamworm experiments rank --models artifacts/eval-g1-v1/models/ \
  --menu configs/experiments/feasible-interventions.toml \
  --out artifacts/interventions-v1/

# Reconstruct reports with provenance links.
occamworm report build --run artifacts/eval-g1-v1/ \
  --out artifacts/reports/eval-g1-v1/
```

CLI names and paths are proposed and should be treated as acceptance contracts for implementation, not as currently runnable commands.
