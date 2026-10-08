# Data contract: trials, responses and table schemas

> Part of the **Occam's Worm specification v0.4.0** · [Spec index](../README.md) · Source: §2.4, §14.4

## 2.4 Trial inclusion and response definition

A trial is an experimental exposure, not a stimulated–responder pair. A single stimulation produces a vector of simultaneously observed responses. This correlation must be represented in modeling and bootstrap sampling.

For every trial:

- Use pre-stimulus interval to estimate fluorescence baseline and uncertainty.
- Preserve the target's measured autoresponse as a possible input covariate **only if that quantity is available for the intended prediction task**.
- Account for target-neuron stimulation artifacts and spectral contamination.
- Distinguish *unobserved*, *observed but uninterpretable*, and *observed near noise floor*.
- Maintain masks for per-frame motion/extraction failures.
- Avoid hard thresholding to only significantly responsive downstream cells for primary likelihood metrics; this creates selection bias.
- Maintain a separate vetted responder subset for exploratory kinetic-shape studies.
- Record the trial's position in its recording: stimulation index, elapsed recording time, time since the previous stimulation, and the previous target. Trials from one recording share an evolving animal state; they are not exchangeable exposures.
- Store the stimulated neuron's own trace (the autoresponse) as a separately typed input channel. It is an allowed input in T2a and is never scored as a responder output; in T2b it may be scored separately as a stimulation-efficacy prediction.

**Important:** Trial-level negative examples are crucial. A model that predicts a large response everywhere must be penalized even if it fits the strongest pairs well.


## 14.4 Canonical JSON/Parquet schemas

Use Arrow/Parquet for dense trial-level analysis and JSON for metadata. The following schema is **proposed**, not a description of the upstream file format.

### `animals.parquet`

```text
animal_id: string              # hashed/canonical unique experiment animal
session_id: string             # may be multiple per animal
batch_id: string|null
strain: string|null
genotype: string
sex: enum|null                 # adult hermaphrodite expected primary scope
life_stage: string|null
preparation: enum|null         # immobilized, freely moving, other
source_dataset: string
source_recording_id: string
acquisition_date: string|null # if released; never required
quality_flags: list<string>
```

### `trials.parquet`

```text
trial_id: string
animal_id: string
session_id: string
stim_target_id: string
stimulus_id: string
stimulus_start_s: float64
stimulus_duration_s: float64
stimulus_amplitude: float64|null
stimulus_amplitude_units: string|null
inter_trial_interval_s: float64|null
stim_index_in_recording: int32         # 0-based order of this stimulation in its recording
time_since_recording_start_s: float64|null
time_since_previous_stim_s: float64|null
previous_stim_target_id: string|null
autoresponse_trace_ref: string|null    # target's own trace; input channel, never a scored output
autoresponse_usable: bool|null
stimulus_verified_on_target: bool|null
trial_qc_flags: list<string>
```

### `observations.parquet`

```text
trial_id: string
neuron_id: string
time_s: float64
fluorescence_delta_f_over_f: float32|null
measurement_units: string
observed: bool
valid_sample: bool
qc_flags: list<string>
processing_version: string
```

### `edges.parquet`

```text
source_neuron_id: string
target_neuron_id: string
edge_kind: enum                # chem, gap, putative_mod, nmj
synapse_count: float32|null
sign: enum|null                # excitatory, inhibitory, unknown
confidence: float32|null
source_reconstruction: string
life_stage: string|null
provenance_reference: string
```

### `run_manifest.json`

```json
{
  "schema_version": "0.1.0",
  "run_id": "example-not-a-real-run",
  "dataset_manifest_sha256": "<sha256>",
  "split_manifest_sha256": "<sha256>",
  "canonical_program_sha256": "<sha256>",
  "git_commit": "<git-sha>",
  "config_sha256": "<sha256>",
  "compiler_version": "<version>",
  "sim_backend": "cpu_reference",
  "seed": 12345,
  "training_animal_count": null,
  "test_animal_count": null,
  "wall_time_seconds": null,
  "peak_memory_bytes": null,
  "status": "not_executed"
}
```

In deployed schemas, use actual hashes and meaningful versioned enums; the example placeholders above are deliberately not measured results.
