# Program discovery and parameter inference

> Part of the **Occam's Worm specification v0.4.1** · [Spec index](../README.md) · Source: §7

## 7. Program discovery and parameter inference

### 7.1 Two coupled search problems

**Structural search:** choose the AST and which registers/operators exist.  
**Numerical inference:** choose shared continuous parameters, edge-scale parameters, observation parameters, nuisance distributions and initial states.

These should not be conflated: optimizing continuous parameters with gradients is not the same task as searching over program structures.

### 7.2 Search phases

**S0 — Enumeration:** exhaustive enumeration of small G0/G1 programs under a fixed bit budget; canonical deduplication; stable-simulation filter.

**S1 — Local structured mutation:** insert/remove one operation, change threshold to saturation, add/remove adaptation, replace sum by typed sum, alter allowed sharing hierarchy.

**S2 — Guided synthesis:** use surrogate predictions of fit/complexity, grammar-based Monte Carlo search or evolutionary proposals while retaining uncertainty and diversity.

**S3 — Hybrid parameter optimization:** estimate continuous `theta` using differentiable simulators when possible, otherwise derivative-free local search, likelihood-free inference, or Bayesian optimization for small dimensions.

**S4 — Conditional model averaging:** retain multiple candidates if the data cannot distinguish them, rather than collapsing to a single winner.

**S5 — Deliberate experimental discrimination:** select held-out or new stimulus protocols predicted to separate candidate models.

### 7.3 Search strategy: avoid evaluating nonsense

Use cheap gates in the following order:

1. Type checking and static unit constraints.
2. Canonical IR deduplication.
3. Stability/finite-output tests on a synthetic graph.
4. Computational budget/complexity limit.
5. Small pilot subset fit, selected **within the training fold**.
6. Full inner-training fit and inner-validation scoring.
7. Diversity-aware Pareto filtering (fit, complexity, stability, biological penalty).
8. Outer-test evaluation **only** for frozen final candidates.

Pure exhaustive search becomes combinatorial quickly. `max_nodes`, `max_registers`, `max_depth`, `max_parameters`, and per-tier operator allowlists are first-class configuration, not suggestions.

### 7.4 Algorithm: frozen nested experimental evaluation

```text
function evaluate_research_round(dataset, grammar, search_config, split_spec):
    raw = validate_dataset(dataset)
    outer_folds = split_by_animal(raw, split_spec.outer_seed)
    all_outer_reports = []

    for outer_train, outer_test in outer_folds:
        # Only outer_train is visible to all development activities.
        inner_folds = split_by_animal(outer_train, split_spec.inner_seed)
        candidates = enumerate_and_mutate(grammar, search_config)
        candidates = canonicalize_and_static_filter(candidates)

        ranked = []
        for program in candidates:
            scores = []
            for inner_train, inner_val in inner_folds:
                params = fit(program, inner_train, search_config.fit_budget)
                prediction = predict(program, params, inner_val.inputs)
                scores.append(score_masked_predictive_density(prediction, inner_val))
            ranked.append((program, aggregate_by_animal(scores)))

        winner_set = freeze_selection(pareto_select(ranked))
        for selected in winner_set:
            params = fit(selected, outer_train, final_fit_budget)
            result = predict(selected, params, outer_test.inputs)
            all_outer_reports.append(score_once(result, outer_test))

    return preregistered_grouped_analysis(all_outer_reports)
```

**Critical implementation detail:** the outer test animal's ground-truth outputs must be inaccessible to the search process, including through plots, aggregate matrices, precomputed denoising transforms, and automatic feature selection. Hyperparameters selected after examining an outer fold require a new nested or external validation set.

### 7.5 Continuous parameter fitting

Supported parameter-sharing strategies, from strongest to weakest:

- Global constant shared by all neurons.
- Per rule-family or cell-type parameter.
- Per presynaptic transmitter class.
- Hierarchical random effect with shrinkage.
- Per-neuron effect with explicit cost.
- Per-edge effect with explicit cost (allowed primarily in comparison baselines).

Optimize in transformed bounded domains: `tau=softplus(raw_tau)+tau_min`, `gain=softplus(raw_gain)` etc. Use multi-start fitting and gradient checks. Require posterior predictive checking when using stochastic parameter inference.

### 7.6 Gradient approaches

**Implementation.** Parameter fitting uses a differentiable simulator written in JAX or PyTorch, run in float64 on CPU for confirmatory fits. C++ automatic-differentiation tools exist, but they add build complexity and correctness risks of their own, and hand-written adjoints are a risk the project does not need early. The Python simulator must pass the same conformance suite as the C++ reference interpreter ([§6.8](../runtime/SIM_SEMANTICS.md)). The canonical program hash, not the implementation language, identifies a model.

For smooth G1/G2 rules:

- Implement analytical/automatic differentiation through time for short windows.
- Use checkpointing or adjoints only if memory becomes consequential.
- Apply gradient clipping only under explicitly recorded settings.
- Compare gradient against finite differences on tiny tests.
- Use truncated backprop only after quantifying truncation bias.

For discrete G0/G3 operations:

- Use exact enumeration of local truth tables when feasible.
- Compare evolutionary and simulated-annealing structure mutations.
- Surrogate gradients may speed optimization, but evaluation always uses the **true forward semantics**.

### 7.7 Posterior over programs

To capture epistemic uncertainty, use an approximate posterior

$$
p(R,\theta\mid D)\propto p(D\mid R,\theta)\,p(\theta\mid R)\,2^{-L_{struct}(R)}.
$$

Here `2^{-L_struct}` is the structural prior and `p(theta|R)` plays the role of `L_params`; the point-estimate objective of [§3.5](../science/PROBLEM_STATEMENT.md) expresses the same trade-off with `L_params` charged in bits.

This is a **chosen prior**, not an assertion that the true nervous system is sampled from a universal algorithmic prior. Exact Solomonoff induction is uncomputable; practical grammar priors depend heavily on the chosen language and length encoding.

For a very small discrete model space, calculate approximate Bayesian evidence or cross-validated likelihoods. For large spaces, maintain likelihood-weighted particles/top candidates and clearly label approximations.

### 7.8 Avoiding overfitting through search volume

A search that tests one million programs can overfit a validation set even if each program is small. Controls:

- Nested CV and independent target holdouts.
- Frozen benchmark and a limited number of final leaderboard submissions.
- Record complete experiment history and number of attempted candidates.
- Include search-budget curves and a random-grammar/control-grammar comparison.
- Repeat across multiple seeds.
- Pre-register key comparisons before final holdout evaluation.

### 7.9 Minimum description length versus pure accuracy

Publish a Pareto curve, not only one weighted scalar:

$$
\mathrm{ParetoFront}=\{(L_{total},\; \mathrm{heldout\ NLL})\}.
$$

A slightly worse but substantially shorter rule may be scientifically valuable. Conversely, a microscopic complexity saving that greatly worsens causal response prediction is not acceptable for faithful emulation.
