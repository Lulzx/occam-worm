"""Trial-aligned response windows from a normalized atlas import.

A window is one (trial, ROI) pair: ``PRE`` volumes before the stimulation frame and ``POST`` volumes from it on.
Fluorescence is converted to dF/F with a baseline taken only from the trial's own pre-stimulus volumes
(§2.4), so no post-stimulus sample informs the normalization. Samples are masked when tracking failed, when the
window runs past the recording, or when it reaches the next stimulation of the same recording.

Windows are cached under ``data/derived/<dataset>/`` keyed by the import manifest digest and the window
parameters; a changed import invalidates the cache.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pyarrow as pa
import pyarrow.parquet as pq

from occamworm.sources.cli import find_root

DATASET = "randi2023-wt-v1"
WINDOW_VERSION = "windows-v2"
PRE = 10  # volumes before the stimulation frame (5 s at 2 Hz)
POST = 60  # volumes from the stimulation frame on (30 s at 2 Hz)
MIN_BASELINE_SAMPLES = 4
MIN_F0 = 1e-6
# Volumes around the stimulation frame carry a stimulation-light artifact in every ROI (audit-v2: mean dF/F
# jumps by ~0.03 in responders and ~0.3 in targets at offset 0, decaying within two volumes). They are masked in
# every window and excluded from the baseline (§2.4).
ARTIFACT_OFFSETS = (-1, 0, 1)
PROFILE_OFFSETS = np.arange(-3, 5)

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]
IntArray = npt.NDArray[np.int64]


@dataclass
class Windows:
    """Columnar (trial, ROI) windows plus the trial and neuron tables they index."""

    trials: dict[str, Any]
    neurons: dict[str, Any]
    animals: dict[str, Any]
    trial_index: IntArray  # row into ``trials``
    roi_index: IntArray
    neuron_id: npt.NDArray[np.object_]  # canonical-looking unique label or None
    is_target: BoolArray
    dff: npt.NDArray[np.float32]  # [n, PRE + POST]
    valid: BoolArray  # [n, PRE + POST]
    f0: npt.NDArray[np.float32]
    pre_sd: npt.NDArray[np.float32]
    volume_interval_s: float
    meta: dict[str, Any] = field(default_factory=dict)
    # mean unmasked dF/F at PROFILE_OFFSETS: row 0 stimulated ROI, row 1 other ROIs (artifact evidence)
    artifact_profile: FloatArray = field(default_factory=lambda: np.zeros((2, PROFILE_OFFSETS.size)))

    @property
    def offsets(self) -> IntArray:
        """Volume offset of each window column relative to the stimulation frame."""
        return np.arange(-PRE, POST, dtype=np.int64)

    def __len__(self) -> int:
        return int(self.trial_index.size)


def dataset_dir(root: Path | None = None, dataset: str = DATASET) -> Path:
    return (root or find_root(Path.cwd())) / "data" / "normalized" / dataset


def _manifest_digest(path: Path) -> str:
    return hashlib.sha256((path / "manifest.json").read_bytes()).hexdigest()


def _columns(table: Any) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name in table.column_names:
        col = table.column(name)
        if pa.types.is_list(col.type):
            values = col.to_pylist()
            arr = np.empty(len(values), dtype=object)
            arr[:] = values
            out[name] = arr
        else:
            out[name] = col.to_numpy(zero_copy_only=False)
    return out


def load_tables(directory: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    trials = _columns(pq.read_table(directory / "trials.parquet"))
    neurons = _columns(pq.read_table(directory / "neurons.parquet"))
    animals = _columns(pq.read_table(directory / "animals.parquet"))
    return trials, neurons, animals


def _dense_recording(path: Path, n_frames: int, n_rois: int) -> FloatArray:
    t = pq.read_table(path, columns=["roi_index", "frame", "gcamp", "gcamp_valid"])
    roi = t.column("roi_index").to_numpy().astype(np.int64)
    frame = t.column("frame").to_numpy().astype(np.int64)
    g = t.column("gcamp").to_numpy(zero_copy_only=False).astype(np.float64)
    ok = t.column("gcamp_valid").to_numpy(zero_copy_only=False).astype(bool)
    dense = np.full((n_frames, n_rois), np.nan)
    dense[frame[ok], roi[ok]] = g[ok]
    return dense


def build_windows(directory: Path) -> Windows:
    trials, neurons, animals = load_tables(directory)
    intervals = np.unique(animals["volume_interval_s"])
    if intervals.size != 1:
        raise ValueError(f"mixed volume intervals {intervals.tolist()}")

    neuron_ids: dict[tuple[str, int], str | None] = {
        (str(r), int(i)): (None if n is None else str(n))
        for r, i, n in zip(neurons["recording_id"], neurons["roi_index"], neurons["neuron_id"], strict=True)
    }
    rec_rows = {str(r): k for k, r in enumerate(animals["recording_id"])}
    length = PRE + POST

    parts: dict[str, list[Any]] = {k: [] for k in ("trial", "roi", "nid", "tgt", "dff", "valid", "f0", "sd")}
    art_sum = np.zeros((2, PROFILE_OFFSETS.size))  # rows: stimulated ROI, other ROIs
    art_cnt = np.zeros((2, PROFILE_OFFSETS.size))
    by_recording: dict[str, list[int]] = {}
    for k, rec in enumerate(trials["recording_id"]):
        by_recording.setdefault(str(rec), []).append(k)

    for rec in sorted(by_recording):
        a = rec_rows[rec]
        n_frames, n_rois = int(animals["n_frames"][a]), int(animals["n_rois"][a])
        name = rec.split(":", 1)[1]
        dense = _dense_recording(directory / "observations" / f"{name}.parquet", n_frames, n_rois)
        names = np.array([neuron_ids.get((rec, roi)) for roi in range(n_rois)], dtype=object)
        for k in by_recording[rec]:
            stim = int(trials["stim_frame"][k])
            nxt = trials["next_stim_frame"][k]
            stop = n_frames if nxt is None or (isinstance(nxt, float) and np.isnan(nxt)) else int(nxt)
            frames = np.arange(stim - PRE, stim + POST)
            inside = (frames >= 0) & (frames < n_frames) & ((frames < stop) | (frames < stim))
            win = np.full((length, n_rois), np.nan)
            win[inside] = dense[frames[inside]]
            raw_profile = win[PRE + PROFILE_OFFSETS].copy()
            win[PRE + np.asarray(ARTIFACT_OFFSETS)] = np.nan
            pre = win[:PRE]
            n_pre = np.sum(~np.isnan(pre), axis=0)
            with np.errstate(invalid="ignore", divide="ignore"):
                f0 = np.where(n_pre > 0, np.nansum(pre, axis=0) / np.maximum(n_pre, 1), np.nan)
                usable = (n_pre >= MIN_BASELINE_SAMPLES) & (f0 > MIN_F0)
                dff = (win - f0) / f0
                sd = np.nanstd(np.where(np.isnan(pre), np.nan, dff[:PRE]), axis=0)
            valid = ~np.isnan(dff) & usable[None, :]
            dff = np.where(valid, dff, 0.0)
            target = trials["stim_target_roi"][k]
            target_roi = -1 if target is None or (isinstance(target, float) and np.isnan(target)) else int(target)
            with np.errstate(invalid="ignore", divide="ignore"):
                prof = (raw_profile - f0) / f0
            prof_ok = ~np.isnan(prof) & usable[None, :]
            is_t = np.arange(n_rois) == target_roi
            for row, sel in ((0, is_t), (1, ~is_t)):
                art_sum[row] += np.where(prof_ok[:, sel], prof[:, sel], 0.0).sum(axis=1)
                art_cnt[row] += prof_ok[:, sel].sum(axis=1)
            rois = np.arange(n_rois)
            parts["trial"].append(np.full(n_rois, k))
            parts["roi"].append(rois)
            parts["nid"].append(names)
            parts["tgt"].append(rois == target_roi)
            parts["dff"].append(dff.T.astype(np.float32))
            parts["valid"].append(valid.T)
            parts["f0"].append(np.where(usable, f0, np.nan).astype(np.float32))
            parts["sd"].append(np.where(usable, sd, np.nan).astype(np.float32))

    return Windows(
        trials=trials,
        neurons=neurons,
        animals=animals,
        trial_index=np.concatenate(parts["trial"]).astype(np.int64),
        roi_index=np.concatenate(parts["roi"]).astype(np.int64),
        neuron_id=np.concatenate(parts["nid"]),
        is_target=np.concatenate(parts["tgt"]),
        dff=np.concatenate(parts["dff"]),
        valid=np.concatenate(parts["valid"]),
        f0=np.concatenate(parts["f0"]),
        pre_sd=np.concatenate(parts["sd"]),
        volume_interval_s=float(intervals[0]),
        artifact_profile=art_sum / np.maximum(art_cnt, 1),
    )


def _cache_key(directory: Path) -> dict[str, Any]:
    return {
        "dataset": directory.name,
        "window_version": WINDOW_VERSION,
        "pre": PRE,
        "post": POST,
        "min_baseline_samples": MIN_BASELINE_SAMPLES,
        "artifact_offsets": list(ARTIFACT_OFFSETS),
        "input_manifest_sha256": _manifest_digest(directory),
    }


def load_windows(directory: Path | None = None, cache_dir: Path | None = None) -> Windows:
    """Load windows from the cache, rebuilding them if the import or the parameters changed."""
    directory = directory or dataset_dir()
    cache_dir = cache_dir or directory.parent.parent / "derived" / directory.name
    key = _cache_key(directory)
    meta_path, data_path = cache_dir / f"{WINDOW_VERSION}.json", cache_dir / f"{WINDOW_VERSION}.npz"
    if meta_path.exists() and data_path.exists() and json.loads(meta_path.read_text()) == key:
        trials, neurons, animals = load_tables(directory)
        with np.load(data_path, allow_pickle=False) as z:
            nid = z["neuron_id"].astype(object)
            nid[nid == ""] = None
            return Windows(
                trials=trials,
                neurons=neurons,
                animals=animals,
                trial_index=z["trial_index"],
                roi_index=z["roi_index"],
                neuron_id=nid,
                is_target=z["is_target"],
                dff=z["dff"],
                valid=z["valid"],
                f0=z["f0"],
                pre_sd=z["pre_sd"],
                volume_interval_s=float(z["volume_interval_s"]),
                meta=key,
                artifact_profile=z["artifact_profile"],
            )
    w = build_windows(directory)
    w.meta = key
    cache_dir.mkdir(parents=True, exist_ok=True)
    tmp = data_path.with_suffix(".tmp.npz")
    np.savez(
        tmp,
        trial_index=w.trial_index,
        roi_index=w.roi_index,
        neuron_id=np.array(["" if n is None else n for n in w.neuron_id], dtype=str),
        is_target=w.is_target,
        dff=w.dff,
        valid=w.valid,
        f0=w.f0,
        pre_sd=w.pre_sd,
        volume_interval_s=np.float64(w.volume_interval_s),
        artifact_profile=w.artifact_profile,
    )
    tmp.replace(data_path)
    meta_path.write_text(json.dumps(key, indent=2) + "\n")
    return w
