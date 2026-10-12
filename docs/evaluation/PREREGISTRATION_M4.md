# M4 pre-registration: rule search on the atlas (`atlas-v1`)

> Part of the **Occam's Worm specification v0.4.1** · [Spec index](../README.md) · Ticket: [OW-017](../planning/tickets/OW-017.md) · Config: [`configs/searches/atlas-v1.json`](../../configs/searches/atlas-v1.json)

**Status: draft for review (2026-10-12). Not frozen.** It freezes when this line says "frozen", with the date and the sha256 of `atlas-v1.json`, `g1-m4.json` and this file recorded in the [experiment registry](EXPERIMENT_REGISTRY.md#registered-entries). That must happen before the first outer fold of `atlas-v1` is scored. Any later change needs a new name (`atlas-v2`) and a note on what changed.

It settles open decisions 3, 4, 6, 7, 8 and 10 ([Appendix B](../planning/OPEN_DECISIONS.md)) for this experiment.

## 1. Question and hypotheses

Primary (H2, [§2](../SUMMARY.md)): does a short WRL rule run on the connectome predict held-out animals' responses to stimulation better than the strongest eligible baseline?

- **H2.** The rule track beats the baseline track on T2a (§7, criteria 1–4).
- **H2-anatomy.** The rule track's advantage depends on the real wiring. It beats the same program structures refitted on degree-preserving randomized graphs (§9).
- **H2-np (secondary).** Neuropeptide routes add predictive information. `gap-leak-np` beats `gap-leak` and beats the same rule on label-shuffled peptide graphs (§9).
- **H1** is tested in M1 (`baselines-v1`, `baselines-v2-t2b`) under the §4.5 rule and is not re-tested here.

## 2. What has already been seen

Choices in this document were made after the following were seen. The analysis plan reports them so readers can judge the risk of adaptive choices.

| Seen | Scope | Used for |
| --- | --- | --- |
| T2a baseline benchmark, outer scores | all 113 folds, no history; 36 folds with history (`baselines-v1`) | comparator definition (§6). The comparator is a rule, not a family chosen from these results. |
| `atlas-demo`, `atlas-demo-np` | inner and outer scores of folds 0–2 | grammar additions (§4), the H2-np control design |
| Noise-model evidence | inner folds of outer folds 0–4 only | decision 3 (§3) |
| Pilot-versus-full ranking and timing | inner folds of fold 0 only | search procedure (§5) |

Sensitivity row: every primary result is also reported without folds 0–2 (§10).

## 3. Data, tasks and observation model

- **Data:** `randi2023-wt-v1`, eligible targets from `audit-v2`, the frozen LOAO split (decision 15). There are 113 outer × 5 inner folds; 110 test animals have eligible traces.
- **Primary task (decision 4): T2a, no history term.** T2b is reported alongside with the published GCaMP6s indicator (experiment v2, decision 14). T2a and T2b are never averaged. The history variant (decision 13) is a sensitivity row. T3 (held-out targets) is exploratory here and makes no confirmatory claim.
- **Observation model (decision 3): Student-t AR(1)**, scale a²s² + b², with ν fitted on training residuals per fit (no test-time calibration). On inner folds of outer folds 0–4 it beat Gaussian by 0.134 nats per sample for B0, B1 and B3 alike, and calibrated the 50/80/95% intervals (50/82/96% against 62/88/96%). Gaussian AR(1) is a sensitivity row.
- **All tracks are fitted and scored in the same run, with the same code and observation model.** Scores from `baselines-v1` (Gaussian, an older commit) are not reused.

## 4. Search space (decision 6)

- **Grammar:** G1 with the operator allowlist of `g1-tiny` (`neg relu tanh add mul leaky_integrate euler_leak stimulus sum_in const param`).
  - `sum_in` selectors: `exc`, `inh`, `all`.
  - Two `gap` variants per program: none, or `gap r0 scale p` with a trainable dimensionless scale in [0, 10] (10 bits).
  - `gap` and `all` are added because `gap-leak`, which reads `sum_in(v, all)` and couples over gap junctions, closed 26–49% of the B0→B3 inner gap in the demos. Nothing enumerated without them closed more than 4%. This addition was informed by inner scores of demo folds 0–2.
- **Budget:** `max_nodes` 6, `max_depth` 4, `max_registers` 1, `max_parameters` 3. Parameters: the `g1-tiny` pool (gain in [0, 2]; time constants in [0.1, 5] s and [0.01, 5] s; 8 bits each) plus the gap scale. Config: [`configs/searches/g1-m4.json`](../../configs/searches/g1-m4.json).
- **Pre-declared hand-written rules:** `gap-leak` and `leak-adapt` on the rule track; `gap-leak-np` variants for H2-np only (§9).
- **Volume rule.** Every gated program is scored by the pilot; nothing is truncated by enumeration order. If the measured cost (§5) projects past 400 worker-hours, `max_nodes` drops to 5 before launch. That decision uses only the enumeration count and fold-0 timing.

## 5. Search procedure (per outer fold, training animals only)

1. **Enumerate and gate.** Run `ow search enumerate`, then the structure and dynamics gates. Every gate rejection is logged.
2. **Pilot.** Each gated program is fitted on inner split 0 with 1 start (seed 0) and scored on that split's validation animals.
3. **Full evaluation.** The pilot's top K = **32** programs plus the pre-declared rules get all 5 inner splits with 2 starts. The rule-track winner has the lowest summed inner-validation NLL; ties go to the shorter `L_total`.
4. **Freeze.** Refit the winner on the outer-training animals, freeze it (sha256 plus the training-animal hash) and score the held-out animal once (`LockedEvaluator`).
5. **Log.** Record the Pareto front (`L_total`, inner NLL), the search log, and budget curves in enumeration order, pilot order and over random orders (§7.8).

Evidence for K comes from fold 0's training view only:

- **Sample.** Every 4th program of `g1-tiny` was gated (983 of 5,465 kept, 18%), and 60 of the kept programs were scored both ways.
- **Agreement.** The pilot and full inner NLL agree with Spearman ρ = 0.996. The full top 5 are exactly the pilot top 5; the full winner was 4th in the pilot.
- **Ties.** 29 of the 60 programs tie at one score.

So K = 32 leaves a wide margin.

Compute (fold 0, one core):

| Step | Cost |
| --- | --- |
| Gate | 7.8 ms per program |
| Pilot | 3.5 s per program |
| Full evaluation | 11.5 s per program; 21 s for `gap-leak` |

- **Volume.** Without gap variants, `g1-m4` has 32,140 unique programs at 6 nodes and 5,402 at 5. The gap variants are assumed to double this, which gives about 11,600 gated programs at 6 nodes.
- **Projected cost at today's pilot cost.** About 1,260 worker-hours at 6 nodes and 220 at 5.
- **Overhead.** Most of the pilot's cost is per-program overhead: the T2a pair statistics, which are the same for every candidate, are recomputed each time, and each program is traced again. OW-017 removes that overhead before the measured projection.
- **Applying the rule.** The volume rule of §4 is applied to the measured projection.

## 6. Comparator: the strongest eligible baseline

- **Baseline track.** Per outer fold, the family with the lowest summed inner-validation NLL among B0, B1, B1d, B2-K1, B2-K2, B2-K4, B2-K8, B3, B4-fixed and B4-learned. Each family uses its inner-selected ridge, the same observation model, and is refitted and scored in the same run.
  - The comparator is fixed by this rule, not by outcome. On the earlier Gaussian T2a run this rule picked B2-K4 in 85 of 110 folds.
- **Named secondary contrasts:** rule track against B4-learned (the equally budgeted linear anatomical model) and against B3 (independent pairwise kernels). Both come from H2.

## 7. Primary analysis and success criteria (§11.6)

- **Statistic.** Δa = NLL_baseline(a) − NLL_rule(a), the summed held-out NLL of animal a. Positive favours the rule.
- **Estimate.** Mean Δ over the 110 scored animals, with a 95% percentile interval from 2,000 animal-level bootstrap replicates (seed 20261008).

H2 is supported only if all of the following hold:

1. Mean Δ > 0 and the interval excludes zero.
2. **Not driven by one animal, day or responder subset:**
   - The median Δa is > 0.
   - Mean Δ stays > 0 when any single animal is dropped, and when any single recording day is dropped.
   - Mean Δ stays > 0 when traces of the 10% of pairs with the largest training-set |mean amplitude| are excluded.
3. **Calibration and null responses do not get worse (decision 7):**
   - Calibration: at the 50%, 80% and 95% levels, the rule track's |coverage − nominal| on pooled held-out samples is at most the baseline track's + 0.05.
   - Null responses: on held-out traces of pairs that are non-responsive in the outer-training animals (|t| < 1, computed on those animals only), the 95% upper bound of the mean per-sample NLL difference (rule − baseline) is at most 0.02 nats.
   - Scale of that margin: the t-versus-Gaussian difference was 0.134 nats/sample.
4. The complexity/accuracy trade-off is reported: the Pareto front, `L_total` of each fold's winner, and the gap fraction (B0 − rule) / (B0 − baseline) with its interval.
5. Any shared-timescale wording follows the M1 §4.5 outcome.

If H2 fails, the result is reported as a calibrated negative result: the gap fraction with its interval, the Pareto front, and where the rule loses (§10). No victory language is used if power is inadequate.

## 8. Statistical unit (decision 8)

- **Unit: the animal** (one recording per animal).
- 113 animals were recorded on 51 days, at most 6 per day; the date is parsed from the recording name. `batch_id` is empty in the export, so the recording day is the batch proxy.
- **Sensitivity:** the primary interval is recomputed with a day-clustered bootstrap (51 clusters).

## 9. Negative controls

Controls refit the rule-track structure selected in each fold on corrupted outer-training data and score the held-out animal once. Each yields a paired Δ against the real-data fit.

| Control | Construction | Supports |
| --- | --- | --- |
| Degree-preserving randomized graphs | chemical edges rewired preserving in- and out-degree, each edge keeping its sign; gap junctions rewired preserving degree; 5 seeds | H2-anatomy: real − mean(shuffled) > 0, interval excluding zero |
| Stimulus-target label shuffle | target labels permuted across trials within each recording (timing kept); 5 seeds | the rule uses target identity |
| No modulation | `gap-leak-np` vs `gap-leak` | H2-np |
| Label-shuffled peptide graph | `gap-leak-np` on 5 label-shuffled peptide graphs (seeds 0–4) | H2-np: real peptide labels beat the shuffled ones, interval excluding zero |

- **Full re-search check.** A complete nested re-search on randomized graphs runs for a pre-declared subset of 10 folds (every 11th fold: 0, 11, …, 99). It checks that refitting the selected structure is not a weaker control than re-searching.
- **Why the H2-np controls are required.** In the demo, peptide edges raised the gap fraction of `gap-leak` from 0.36 to 0.45 (fold 0). Label-shuffled peptide graphs reached 0.42, so most of the gain was not label-specific.

## 10. Sensitivity analyses (reported, not criteria)

- Gaussian AR(1) observation model.
- Excluding folds 0–2.
- Day-clustered bootstrap.
- History variant (decision 13).
- Winner refit with seeds 0–2, 2 or 4 starts, and `dt_max` halved: the largest per-animal change in held-out NLL.
- T2b scores of the same structures.
- Residuals by neuron class, response sign and peak latency (exploratory, labelled as such).

## 11. Claim levels (decision 10)

| Claim | Required evidence |
| --- | --- |
| "A short rule predicts held-out responses" | §7 criteria 1–3 |
| "…and depends on the wiring" | the above plus H2-anatomy |
| "Shared timescales are neural" | the M1 §4.5 interpretation rule |
| "Extrasynaptic (neuropeptide) routes contribute" | H2-np: beats the no-modulation rule and the label-shuffled peptide graphs |
| "Cell-specific mechanism" | not testable in this search space (type-shared global parameters only); no such claim |

## 12. Reported regardless of outcome

- Every fold's winner and Pareto front.
- Search volume, gate rejections and budget curves.
- All targets, including near-null pairs.
- Every control and sensitivity row.
- T2a and T2b separately.
- Any deviation from this document, with its reason.
