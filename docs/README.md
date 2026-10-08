# Occam's Worm specification

**Occam's Worm: A Computational Rule-Discovery Engine for *C. elegans* Neural Emulation**

| | |
| --- | --- |
| Version | 0.4.0 (research design; not an implemented system; supersedes 0.3.0) |
| Date | 2026-10-08 |
| Status | Proposed / falsifiable research program |
| Primary organism | adult hermaphrodite *Caenorhabditis elegans* |
| Implementation | Python (JAX or PyTorch) differentiable simulator and parameter fitting; C++20 for the rule-language toolchain, program enumeration, deterministic reference interpreter and CLI; optional Apple Metal accelerator (via metal-cpp) only after profiling |
| Codename | `occamworm` |
| License | [Apache 2.0](../LICENSE) for original code; upstream data and simulators keep their own licenses (see [open decision 12](planning/OPEN_DECISIONS.md)) |

> **One-sentence goal:** Discover compact, interpretable, biologically constrained computational programs that reproduce and **predict** causal neural responses in *C. elegans*, and quantify exactly which microscopic details must be preserved for progressively stronger forms of neural emulation.

The specification is split by topic. Section numbers (§) from the single-file spec are kept in every heading, so cross-references still resolve; each `§` reference links to the file that holds it.

## Reading order

1. [SUMMARY.md](SUMMARY.md): the concise one-page specification.
2. [overview/EXECUTIVE_SUMMARY.md](overview/EXECUTIVE_SUMMARY.md) and [overview/SCOPE_AND_HYPOTHESES.md](overview/SCOPE_AND_HYPOTHESES.md).
3. The first milestone path: [data/AUDIT_GATE.md](data/AUDIT_GATE.md) → [science/BASELINES.md](science/BASELINES.md) → [evaluation/VALIDATION_PLAN.md](evaluation/VALIDATION_PLAN.md).
4. [planning/ROADMAP.md](planning/ROADMAP.md) and [planning/tickets/](planning/tickets/README.md) to start implementation.

## Layout

| Folder | Contents |
| --- | --- |
| [`overview/`](overview/) | Executive summary, scope and hypotheses, changelog, glossary, references |
| [`data/`](data/) | Sources and acquisition policy, the source registry and acquisition guide, the M0 audit gate, the data contract and table schemas |
| [`science/`](science/) | Problem statement, baselines and indicator control, experiment design, equivalence, shortcuts, perturbations, embodiment |
| [`language/`](language/) | WRL, the Worm Rule Language |
| [`runtime/`](runtime/) | Simulation semantics |
| [`inference/`](inference/) | Program search and parameter fitting |
| [`evaluation/`](evaluation/) | Validation plan, benchmark, testing, experiment registry |
| [`engineering/`](engineering/) | Architecture, reproducibility and toolchain, performance, diagnostics |
| [`planning/`](planning/) | Roadmap, tickets, risks, open decisions |
| [`publication/`](publication/) | Papers, claims discipline, first report outline |
| [`archive/`](archive/) | Frozen single-file copy of v0.4.0 (not canonical) |

## Section map

| § | Section | File |
| --- | --- | --- |
| — | Changes in 0.2.0–0.4.0 | [overview/CHANGELOG.md](overview/CHANGELOG.md) |
| — | Executive summary | [overview/EXECUTIVE_SUMMARY.md](overview/EXECUTIVE_SUMMARY.md) |
| 1 | Scientific motivation and scope | [overview/SCOPE_AND_HYPOTHESES.md](overview/SCOPE_AND_HYPOTHESES.md) |
| 2.1–2.2 | Data assets and acquisition policy | [data/SOURCES.md](data/SOURCES.md) |
| 2.3 | Data-audit gate | [data/AUDIT_GATE.md](data/AUDIT_GATE.md) |
| 2.4 | Trial inclusion and response definition | [data/DATA_CONTRACT.md](data/DATA_CONTRACT.md) |
| 3 | Formal problem statement | [science/PROBLEM_STATEMENT.md](science/PROBLEM_STATEMENT.md) |
| 4 | Baselines and indicator-kinetics control | [science/BASELINES.md](science/BASELINES.md) |
| 5 | Rule language (WRL) | [language/GRAMMAR.md](language/GRAMMAR.md) |
| 6 | Simulation runtime | [runtime/SIM_SEMANTICS.md](runtime/SIM_SEMANTICS.md) |
| 7 | Program discovery and parameter inference | [inference/SEARCH_AND_INFERENCE.md](inference/SEARCH_AND_INFERENCE.md) |
| 8 | Hypothesis graph and experiment design | [science/EXPERIMENT_DESIGN.md](science/EXPERIMENT_DESIGN.md) |
| 9 | Observer-relative equivalence | [science/EQUIVALENCE.md](science/EQUIVALENCE.md) |
| 10 | Predictive shortcuts | [science/PREDICTIVE_SHORTCUTS.md](science/PREDICTIVE_SHORTCUTS.md) |
| 11 | Statistical evaluation protocol | [evaluation/VALIDATION_PLAN.md](evaluation/VALIDATION_PLAN.md) |
| 12 | Perturbations, genetics and neuromodulation | [science/PERTURBATIONS.md](science/PERTURBATIONS.md) |
| 13 | Embodied worm extension | [science/EMBODIMENT.md](science/EMBODIMENT.md) |
| 14.1–14.3, 14.5 | Architecture, components, interfaces, CLI | [engineering/ARCHITECTURE.md](engineering/ARCHITECTURE.md) |
| 14.4 | Table schemas | [data/DATA_CONTRACT.md](data/DATA_CONTRACT.md) |
| 14.6 | Example experiment configuration | [evaluation/EXPERIMENT_REGISTRY.md](evaluation/EXPERIMENT_REGISTRY.md) |
| 14.7–14.8 | Artifact identity, C++ toolchain | [engineering/REPRODUCIBILITY.md](engineering/REPRODUCIBILITY.md) |
| 15 | Roadmap and acceptance gates | [planning/ROADMAP.md](planning/ROADMAP.md) |
| 16 | Testing and quality assurance | [evaluation/TESTING.md](evaluation/TESTING.md) |
| 17.1–17.3 | Diagnostics and visualization | [engineering/DIAGNOSTICS.md](engineering/DIAGNOSTICS.md) |
| 17.4 | Experiment registry | [evaluation/EXPERIMENT_REGISTRY.md](evaluation/EXPERIMENT_REGISTRY.md) |
| 18 | Apple Silicon and scalable execution | [engineering/PERFORMANCE.md](engineering/PERFORMANCE.md) |
| 19 | Risks and anti-patterns | [planning/RISKS.md](planning/RISKS.md) |
| 20, 23 | Publication strategy, final commitment | [publication/PUBLICATION_PLAN.md](publication/PUBLICATION_PLAN.md) |
| 21 | Implementation tickets | [planning/tickets/](planning/tickets/README.md) |
| 22 | References | [overview/REFERENCES.md](overview/REFERENCES.md) |
| App. A | Terminology | [overview/GLOSSARY.md](overview/GLOSSARY.md) |
| App. B | Open decisions | [planning/OPEN_DECISIONS.md](planning/OPEN_DECISIONS.md) |
| App. C | First research report outline | [publication/PUBLICATION_PLAN.md](publication/PUBLICATION_PLAN.md) |

## Editing rules

- The split files are canonical. Do not edit `archive/`.
- Keep `§` numbers stable. If a section moves, update the section map above.
- Record specification changes in [overview/CHANGELOG.md](overview/CHANGELOG.md) and bump the version here and in [SUMMARY.md](SUMMARY.md).
