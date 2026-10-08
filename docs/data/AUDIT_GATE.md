# M0 data-audit gate

> Part of the **Occam's Worm specification v0.4.0** · [Spec index](../README.md) · Source: §2.3

## 2.3 Immediate data-audit gate

Before developing any novel rule model, produce a machine-generated report with:

- Counts of **unique animals**, recording sessions, stimulation trials and stimulus targets by genotype.
- Distribution of observations per stimulated→responder pair, including zero/one/two/many-repeat strata.
- Count of distinct responders per stimulated neuron with usable traces.
- Distribution of stimulus duration, amplitude, timing, sampling rate and measured windows.
- Counts of valid autoresponses, sham events, missing traces, segmentation/motion artifacts and uncertain neuron identities.
- Trial-to-trial versus animal-to-animal variance, conditioned on stimulated neuron.
- Percentage of pairs with enough animal-level repeats for actual held-out evaluation.
- Intersection of neuron IDs across functional atlas, anatomical graph and molecular annotations.
- Coverage of wild-type, `unc-31`, and any other genotype **without assuming** a sufficient sample size.
- Feasibility of the specific shared-timescale experiment under a grouped animal split.
- Stimulation order within each recording: number of prior stimulations, time since the previous stimulation, and identity of recently stimulated targets.
- Within-recording drift: whether autoresponse and responder amplitudes change with stimulation index or elapsed recording time.
- Target-by-fold coverage tables for at least three candidate split schemes (for example 5-fold grouped, 10-fold grouped and leave-one-animal-out), reporting for each eligible target how many outer test folds and inner validation folds contain it.
- Identity of the calcium indicator, any published kinetic parameters for it, and the autoresponse data available to estimate its impulse response ([§4.5](../science/BASELINES.md)).

**Go/no-go criterion:** at least one clearly specified collection of stimulus targets must admit genuine animal-held-out evaluation with nontrivial response variability. If not, the first paper becomes a dataset/identifiability audit rather than a misleading rule-discovery result.

**Split selection:** choose the split scheme from the coverage tables alone, before any baseline or candidate is scored, and record the choice and rationale in the experiment registry. Prefer the scheme that keeps the most eligible targets present in held-out folds, subject to keeping all of an animal's sessions and trials in the same fold. Leave-one-animal-out is admissible for the outer loop when grouped k-fold leaves eligible targets without held-out coverage.
