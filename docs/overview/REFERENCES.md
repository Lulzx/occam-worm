# References and background

> Part of the **Occam's Worm specification v0.4.1** · [Spec index](../README.md) · Source: §22

## 22. References and background

The references below ground dataset availability and prior simulators. The specific software architecture, algorithms, objectives, milestones, metrics, and proposed experiments in this specification are **original design proposals** rather than claims made by these references.

**Verification status (0.2.0):** the citations, DOIs and URLs below were not re-verified in this revision. Check each against the publisher or repository before any external release, and record the check in `sources.json` (OW-001).

#### Scientific sources

- **[R1]** Randi, F., Sharma, A. K., Dvali, S., et al. (2023). *Neural signal propagation atlas of Caenorhabditis elegans.* **Nature**, 623, 406–414. https://doi.org/10.1038/s41586-023-06683-4  
  Public dataset: https://doi.org/10.17605/OSF.IO/E2SYT  
  Interactive atlas: https://funconn.princeton.edu
- **[R2]** Leifer Lab, `pumpprobe`: official analysis/functional-atlas integration and per-recording utilities. https://github.com/leiferlab/pumpprobe  
  Relevant helpers include `Funatlas`, `Fconn`, `get_occurrence_matrix()` and reference fitted temporal kernels. Distinguish measured traces from fitted kernels.
- **[R3]** OpenWorm, *C. elegans Connectome Toolbox*, including selectable anatomical datasets: https://openworm.org/ConnectomeToolbox/  
  Dataset detail: https://openworm.org/ConnectomeToolbox/OpenWormUnified_data/
- **[R4]** Atanas, A. A., Kim, J., et al. (2023). *Brain-wide representations of behavior spanning multiple timescales and states in C. elegans.* **Cell**. Public recordings: https://wormwideweb.org/about/datasets/  
  Related explorer: https://wormwideweb.org/activity/
- **[R5]** Zhao, M., Wang, N., Jiang, X., et al. (2024). *An integrative data-driven model simulating C. elegans brain, body and environment interactions.* **Nature Computational Science**, 4, 978–990. https://doi.org/10.1038/s43588-024-00738-w  
  Representative code: https://github.com/Jessie940611/BAAIWorm
- **[R6]** Worm Neuro Atlas (Francesco Randi), molecular and functional atlas integration: https://github.com/francescorandi/wormneuroatlas  
  Documentation: https://francescorandi.github.io/wormneuroatlas/
- **[R7]** Leifer Lab, `worm-functional-connectivity`: functional atlas tools and examples. https://github.com/leiferlab/worm-functional-connectivity

#### Related methodological concepts

The specification uses standard ideas from minimum description length, nested cross-validation, hierarchical inference, proper scoring rules, graph dynamical systems, program synthesis, Bayesian experimental design, and causal intervention testing. These are standard mathematical and methodological ingredients; the specification's contribution is how they are combined and gated.
