# Randi et al. 2023 atlas: trial-level import

> [Spec index](../README.md) · Implements [OW-002](../planning/tickets/OW-002.md) · Contract: [§2.4, §14.4](DATA_CONTRACT.md) · Sources: [ACQUISITION.md](ACQUISITION.md)

```bash
python -m occamworm.importers randi2023 import
```

```bash
python -m occamworm.importers randi2023 roundtrip --samples 20000
```

`import` re-verifies both input sources against the registry, then writes `data/normalized/randi2023-wt-v1/`. It refuses to write into a non-empty directory. `roundtrip` is the OW-002 acceptance check described below.

## Inputs

| Source | Used for |
| --- | --- |
| `randi2023-atlas-raw-extracted` (OSF `raw_extracted_data/`, 2024) | GCaMP traces, neuron labels, volume times, stimulation frames and target codes, recording names |
| `randi2023-atlas-exported-full` (`exported_data_full.tar.gz`) | Red channel, matched and matchless extractions, manual target files, acquisition and detector metadata |

The archive is streamed once and never extracted into `data/raw`. Its pickles are read by an allowlist unpickler that maps the two upstream classes to an inert container and permits only numpy array reconstruction. No GPL code is imported and no pickle can run code.

The wild-type tarball `exported_data.tar.gz` is **not** used: its traces are post-processed upstream (gaps interpolated, smoothed, rescaled by 1.3–2.9×; see below).

## What the upstream files contain

Established by inspection and exact comparison, 2026-10-08:

- **113 wild-type recordings**, one per text-export index, 2 Hz volumes (0.5 s), 83–210 ROIs and 393–5,395 volumes each. Recording names (`pumpprobe_YYYYMMDD_HHMMSS`) are the stable upstream identity.
- **Traces.** For each ROI, the OSF text column equals, exactly, either the *matched* extraction (`green.txt`) or the *matchless* one (`green_matchless.txt`). This is the per-neuron choice pumpprobe makes when it loads a signal. In 45 recordings every ROI is matched (the text file is byte-identical to `green.txt` without its header); in 68, some high-index ROIs use the matchless trace. A matchless file may cover fewer ROIs than the recording (e.g. 161 of 200); its columns align with the first ROIs by index. The importer records the choice per ROI as `neurons.trace_source` and takes the red channel from the same extraction.
- **NaN** marks volumes where tracking failed (roughly 25–40 % of samples). These are kept as NaN with `*_valid = false`, never filled.
- **Labels** are free text: neuron names (`AVAL`, `AMsoL`), blanks, multicolor-image ids (`41`), and notes. The most common notes are `merge` (740 ROIs) and `target` (39); their meaning is not documented upstream and is left for the M0 audit. Some names appear twice in one recording. A label becomes `neuron_id` only if it looks like a name and is unique in its recording; canonical mapping is OW-013.
- **Target codes** (`<n>_stim_neurons.txt`), as documented in the header of `targets_manually_located.txt`:

  | Code | Meaning | Importer |
  | --- | --- | --- |
  | ≥ 0 | ROI column of the stimulated neuron | `stim_target_code = roi` |
  | −1 | target not located among the tracked ROIs | `not_located` |
  | −2 | targeting failed | `failed_targeting` |
  | −3:LABEL | not tracked, but identified in the multicolor image | `identified_untracked`, label kept |

- **Upstream bug.** pumpprobe assigns −3 from the comments file with a stale loop index, which overwrites the code of the last manually processed event. Both the pickle and the OSF text carry the wrong value. Signature: exported −3, no `-3:label` comment for that event, and a different explicit code in the manual file. Only then does the importer use the manual code; it keeps the exported one in `stim_target_code_exported` and flags the trial `target_code_reconciled_from_manual_file` (4 trials).
- **Bubbles.** pumpprobe writes −2 to every stimulation after a recorded "bubble" event, so recordings can end in a run of −2 even where the manual file had located the target. Those exclusions are deliberate and kept; such trials are flagged `manual_target_superseded_by_export` (4 trials). The bubble file itself is not exported, so a −2 cannot be split into "failed targeting" and "after bubble".
- **Stimulus parameters** come from the acquisition pickle: two-photon, pulse count, trains, rate divider, target x/y/z and wall-clock time. The pulse repetition rate is not exported, so **duration and amplitude are null**; they are never inferred.
- **Animals.** The files identify recordings, not animals. The paper reports 113 animals and the export has 113 recordings, so `animal_id` assumes one recording per animal and every animal carries the flag `animal_id_assumed_one_recording_per_animal`. The M0 audit (OW-003) must confirm or replace this before any animal-wise split.
- **Processed wild-type export.** Compared with the raw text, `exported_data/<n>_gcamp.txt` has no NaN, is 2–4× smoother and is rescaled by 1.3–2.9×; its `97_labels.txt` also differs. It is a downstream product, not a measurement.

## Result (2026-10-08)

113 recordings, 5,808 stimulations, 14,421 ROIs and 51.6 M observation rows (234 MB). 1,504 ROIs in 68 recordings use the matchless trace. Stimulation targets: 4,611 tracked ROIs, 1,011 failed targeting, 162 not located, 24 identified but untracked. 63 recordings list stimulations out of time order and 2 trials share a frame with another. In 3 recordings the detector pickle counts fewer neurons than the trace has ROIs (`fconn_n_neurons_equals_rois`); trace columns are unaffected, but upstream detections may index the shorter list. The 18 full-export recordings outside the wild-type text export (likely unc-31) are listed in the manifest as not imported.

Round trip, 20,000 samples: `gcamp`, `red` and `time_s` 20,000 each, 4,804 trial fields and 2,000 autoresponse samples, **0 mismatches**.

## Output tables

`data/normalized/randi2023-wt-v1/`:

| File | Grain | Notes |
| --- | --- | --- |
| `animals.parquet` | recording | `animal_id`, genotype `wild-type`, acquisition date, frame/ROI/stimulation counts, extraction version, quality flags. Strain, sex, stage and preparation are null: not in the files. |
| `neurons.parquet` | recording × ROI | `roi_id`, raw `label`, `label_status`, `label_duplicated`, `neuron_id`, `trace_source`. |
| `trials.parquet` | stimulation | Schema of §14.4 plus `event_index_in_file`, `stim_frame`, `next_stim_frame`, `previous_trial_id`, target code fields, raw optogenetics fields, `autoresponse_valid_fraction`, `stimulus_verified_on_target` (upstream detector flag), `trial_qc_flags`. `autoresponse_usable` is null until OW-003 declares a criterion. |
| `autoresponses.parquet` | trial × volume | The stimulated ROI's own trace from the previous stimulation to the next one, as a separate input channel. Present only for trials with a tracked target. |
| `observations/<recording>.parquet` | recording × ROI × volume | `gcamp`, `red`, `gcamp_valid`, `red_valid`, `time_s`, `frame`. All volumes, all ROIs, including the target's. |
| `upstream_detections.parquet` | trial × ROI | pumpprobe's detected responders and amplitudes. Derived annotations, never response labels (Anti-pattern B). |
| `manifest.json` | — | Inputs (source ids, asset digests, licenses), code revision, summary counts, sha256 and row count of every table, recordings not imported. |
| `report.json` | recording | Every cross-check result, reconciled events, trailing blank labels. |

### Deviation from the proposed §14.4 schema

§14.4 keys `observations` by trial. Consecutive stimulations are about 30 s apart and windows overlap, so per-trial rows would duplicate most samples. Observations are stored once per recording; a trial is a pointer (`recording_id`, `stim_frame`, `next_stim_frame`), and the target's channel is materialized separately in `autoresponses`. Units are `a.u. (upstream 'box' ROI extraction, camera counts)`; ΔF/F is not computed here, because the baseline is a modeling choice (OW-005).

## Cross-checks

Every redundant field is compared, and a disagreement stops the import or is recorded in `report.json`:

- each text trace column equals the matched or matchless extraction exactly;
- `fconn.stim_indices` and `fconn.stim_neurons` equal the text schedule and codes;
- `fconn.n_neurons` equals the ROI count; the acquisition volume count equals the frame count;
- the acquisition stimulation count equals the event count (otherwise per-event acquisition fields are left null and the trial is flagged `acquisition_metadata_unjoined`);
- −3 labels from the comments file agree with the pickle;
- volume times strictly increase; extra labels beyond the ROI count are blank.

## Round trip (OW-002 definition of done)

`roundtrip` samples rows at random and compares them with the upstream files through a separate code path that splits lines by hand instead of using numpy:

- `gcamp` and `time_s` against the OSF text;
- `red` against `red.txt` or `red_matchless.txt`, per `trace_source`;
- trial frame, start time, exported code, target ROI and label against the text schedule;
- autoresponse samples against the target's text column.

The only documented transform is decimal text → float64, so equality must be exact (NaN equals NaN). The result is written to `roundtrip.json` next to the tables.
