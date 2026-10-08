# `occamworm.baselines`

Baselines B0–B4, each in history and no-history variants, scored on T2a and T2b by the same nested animal-wise evaluation.

**Invariant:** Every baseline gets the same tuning budget (one ridge grid), the same nuisance terms and the same fitted observation model.

- `data.py` — task data (trials, responder traces, designs) and per-(animal, pair) sufficient statistics; the history terms (pre-stimulus slope and stimulation index) live in its nuisance design
- `kernels.py` — B0 null, B1 shared kernel per target (B1d with per-pair delays), B2 global kernel banks (K = 1, 2, 4, 8), B3 independent kernels
- `linear_network.py` — B4 linear network on the Cook 2019 topology, fixed and learned variants, stable by construction
- `indicator.py` — indicator stage and resolvable bandwidth (§4.5)
- `evaluate.py` — inner ridge selection, outer refit, single scoring of held-out animals
- `benchmark.py`, `summary.py`, `control.py` — `python -m occamworm.baselines {benchmark,summarize,control}` on the frozen split
- `synthetic.py` — synthetic data with known kernels (done-when tests, false-sharing control)

B5 (leaky-rate network) and B6 (shared latent state) are not implemented; B5 is expressible as a G1 WRL program for the search (OW-012).

Tickets: [OW-005](../../../docs/planning/tickets/OW-005.md), [OW-006](../../../docs/planning/tickets/OW-006.md), [OW-007](../../../docs/planning/tickets/OW-007.md) · Spec: [§4](../../../docs/science/BASELINES.md)
