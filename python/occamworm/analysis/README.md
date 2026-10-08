# `occamworm.analysis`

Splits, uncertainty and reporting.

**Invariant:** The outer test is scored once, after the model is frozen.

- `dataset.py` — trial-aligned dF/F windows (pre-stimulus baseline only), cached under `data/derived/`
- `audit.py`, `audit_report.py` — M0 audit: `python -m occamworm.analysis audit [--publish]`
- `splits.py` — animal-wise outer and inner splits, frozen split files, leakage checks
- `freeze.py` — `python -m occamworm.analysis splits` freezes the scheme the audit chose
- `uncertainty.py`, `reporting.py` — planned (OW-005, OW-015)

Tickets: [OW-003](../../../docs/planning/tickets/OW-003.md), [OW-004](../../../docs/planning/tickets/OW-004.md), [OW-015](../../../docs/planning/tickets/OW-015.md) · Spec: [§11](../../../docs/evaluation/VALIDATION_PLAN.md)

Status: audit and splits implemented (OW-003, OW-004).
