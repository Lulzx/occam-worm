# `occamworm.search`

Frozen nested evaluation (§7.4) and the search over WRL programs (§7.3, §7.8).

**Invariant:** no search function has a parameter through which held-out data can arrive. The locked evaluator refuses three things: a tampered selection, a selection trained on held-out animals, and a second scoring of a fold.

Tickets: [OW-012](../../../docs/planning/tickets/OW-012.md) · Spec: [§7.4](../../../docs/inference/SEARCH_AND_INFERENCE.md), [§7.8](../../../docs/inference/SEARCH_AND_INFERENCE.md), [§11.8](../../../docs/evaluation/VALIDATION_PLAN.md)

| Module | Role |
|---|---|
| `views.py` | Training-only copies of the task data per outer fold. Each copy has its own animals, targets and pairs, and shares no memory with the parent. |
| `candidates.py` | The candidate protocol, `KernelCandidate` (B0–B3 families with a declared `L_struct`), and `score_kernels`, which scores frozen pair kernels. |
| `wrl.py` | `WrlCandidate`: one WRL program on the declared connectome (details below). Also `enumerate_programs` (`ow search enumerate`), `search_volume` and `budget_curve`. |
| `nested.py` | Ranks candidates by inner-validation NLL (ties go to the shorter `L_total`). Keeps the (`L_total`, NLL) Pareto front, refits the winner, and freezes the selection. Candidates rejected by a gate are logged. |
| `locked.py` | Frozen selections (sha256 plus the hash of the training animals). `LockedEvaluator` scores each fold once. |
| `__main__.py` | `python -m occamworm.search run --config configs/searches/atlas-demo.json` writes `run.json`, `search_volume.json`, `folds.jsonl` and per-fold selections under `artifacts/search-<name>/`. |

`WrlCandidate` details:

- **Connectome.** `cook2019-herm` weights. Chemical signs come from `fenyves2020-sign-prediction`; edges with an unknown sign are treated as excitatory and counted.
- **Kernels.** Pair kernels are the program's linear response to a small pulse into each target, sampled at the shared tent knots.
- **Fitting.** Parameters are fitted with bounded L-BFGS-B on the same profiled pair-statistics loss as B4.
- **Gates.** Before any fit, a program must read the stimulus, couple neurons, and stay bounded at its declared parameters.
- **Unseen pairs.** Kernels come from the graph, so held-out pairs never seen in training are still predicted.

The budget curve gives the best inner NLL after k candidates, in enumeration order and over random orders. The random orders are the §7.8 control.
