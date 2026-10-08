# `occamworm.search`

Frozen nested evaluation (§7.4). `views.py` builds training-only copies of the task data per outer fold; `candidates.py` defines the candidate protocol (kernel families now, WRL programs via the simulator); `nested.py` ranks candidates by inner-validation NLL, keeps the (L_total, NLL) Pareto front and freezes the selection; `locked.py` holds the held-out data and scores each fold once.

**Invariant:** no search function has a parameter through which held-out data can arrive; the locked evaluator refuses tampered selections, selections trained on held-out animals, and second scorings.

Tickets: [OW-012](../../../docs/planning/tickets/OW-012.md) · Spec: [§7.4](../../../docs/inference/SEARCH_AND_INFERENCE.md), [§11.8](../../../docs/evaluation/VALIDATION_PLAN.md)
