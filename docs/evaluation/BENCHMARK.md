# Benchmark definition

> Part of the **Occam's Worm specification v0.4.1** · [Spec index](../README.md)

**Status: not frozen.** This file becomes the frozen animal-wise benchmark at the end of milestone M1 ([roadmap](../planning/ROADMAP.md)). Until then it lists what the frozen version must pin down. Every item comes from elsewhere in the specification; nothing here adds a requirement.

## Must be fixed before any outer fold is scored

| Item | Decided by | Spec |
| --- | --- | --- |
| Dataset manifest and hash | OW-001, OW-002 | [§2.2](../data/SOURCES.md) |
| Eligible stimulus-target cohort and exclusion flowchart | M0 audit (OW-003) | [§2.3](../data/AUDIT_GATE.md), [§4.2](../science/BASELINES.md) |
| Split scheme, chosen from coverage tables alone, with rationale | OW-004 | [§2.3](../data/AUDIT_GATE.md), [§11.1](VALIDATION_PLAN.md) |
| Prediction tasks: T2a primary, T2b alongside; T3 conditioning | — | [§3.6](../science/PROBLEM_STATEMENT.md) |
| Observation model and correlated-noise likelihood | OW-005 | [§3.4](../science/PROBLEM_STATEMENT.md), [§11.2](VALIDATION_PLAN.md) |
| Indicator model, resolvable bandwidth, false-sharing threshold | OW-005, OW-006 | [§4.5](../science/BASELINES.md) |
| Stimulation-history term and initial-state policy | Training-fold audit | [§4.1](../science/BASELINES.md), [§6.7](../runtime/SIM_SEMANTICS.md) |
| Primary metric: paired per-animal ΔNLL with animal-level CI | — | [§4.3](../science/BASELINES.md), [§11.5](VALIDATION_PLAN.md) |
| Calibration and null-response margins | Open decision 7 | [§11.6](VALIDATION_PLAN.md) |
| Baseline tuning budgets | — | [§11.4](VALIDATION_PLAN.md) |

Open items are tracked in [OPEN_DECISIONS.md](../planning/OPEN_DECISIONS.md). Record each frozen value, with date and rationale, in the [experiment registry](EXPERIMENT_REGISTRY.md).
