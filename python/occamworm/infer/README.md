# `occamworm.infer`

Approximate posterior over evaluated programs (§7.7): weights proportional to the cross-validated likelihood times `2^-L_struct`, effective sample size, and top-diverse candidate selection. Hypothesis and hypothesis-edge records follow §8.2.

Implemented in Python rather than `libs/ow-infer/` (§14.1: language follows the bottleneck); nothing here is compute-bound.

Tickets: [OW-014](../../../docs/planning/tickets/OW-014.md) · Spec: [§7.7](../../../docs/inference/SEARCH_AND_INFERENCE.md), [§8.2](../../../docs/science/EXPERIMENT_DESIGN.md)
