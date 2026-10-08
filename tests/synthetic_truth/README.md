# tests/synthetic_truth/

Known generators for recovery tests: leaky neuron, E/I pairs, delayed chains, adaptation, oscillation, misspecification, non-identifiability, carryover and false-sharing cases.

Tickets: [OW-006](../../docs/planning/tickets/OW-006.md), [OW-009](../../docs/planning/tickets/OW-009.md), [OW-011](../../docs/planning/tickets/OW-011.md) · Spec: [§16.3](../../docs/evaluation/TESTING.md)

Status: parameter recovery is implemented (`test_parameter_recovery.py`, generators in `recovery_scenarios.py`): small G1 circuits with E/I chemical edges, delays, gap junctions, adaptation and a calcium observation are simulated at known parameters, noised (Gaussian or AR(1)) and masked, then fitted by `occamworm.fit`; every estimate must be within 3 inverse-Hessian standard errors, the SE must agree with a parametric bootstrap and be calibrated over replicate datasets, and the optimisation budget must be logged ([OW-011](../../docs/planning/tickets/OW-011.md)). The other generators listed above are not implemented; `test_kernel_families.py` covers the OW-006 baselines.
