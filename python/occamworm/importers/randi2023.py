"""Trial-level importer for the Randi et al. 2023 signal propagation atlas, wild type (ticket OW-002).

Inputs (both verified against ``configs/datasets/sources.json`` before use):

* ``randi2023-atlas-raw-extracted``: per-recording plain text from OSF. Supplies neuron labels, volume
  times, the stimulation schedule and the recording name. Its ``<n>_gcamp.txt`` is byte-identical to the
  full export's ``green.txt`` without the header line; that is checked, not assumed.
* ``randi2023-atlas-exported-full``: the pumpprobe-format export. Supplies the raw green (GCaMP) and red
  traces, the manual target-location files and two pickles with acquisition and detector metadata. The
  archive is streamed; nothing is extracted into ``data/raw``. Pickles are read with an allowlist
  unpickler and no upstream code is imported.

No values are transformed: traces are parsed from decimal text to float64 and stored as-is, with NaN
marking frames where tracking failed. See docs/data/RANDI2023_IMPORT.md.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import tarfile
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime
from io import BytesIO
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from occamworm import __version__
from occamworm.importers._safe_pickle import load_attrs

DATASET = "randi2023"
TEXT_SOURCE = "randi2023-atlas-raw-extracted"
FULL_SOURCE = "randi2023-atlas-exported-full"
TEXT_DIR = "raw_extracted_data"
FULL_ARCHIVE = "exported_data_full.tar.gz"
TEXT_SUFFIXES = ("gcamp", "t", "labels", "stim_neurons", "stim_volume_i", "ds_name")
FULL_REQUIRED = ("green.txt", "red.txt", "recording.pickle", "fconn.pickle", "targets_manually_located.txt")
FULL_OPTIONAL = ("targets_manually_located_comments.txt", "green_matchless.txt", "red_matchless.txt")
SCHEMA_VERSION = "0.1.0"
OUTPUT_NAME = "randi2023-wt-v1"
FLUORESCENCE_UNITS = "a.u. (upstream 'box' ROI extraction, camera counts)"

# Stimulated-ROI codes, as documented in the header of targets_manually_located.txt.
TARGET_CODES = {-1: "not_located", -2: "failed_targeting", -3: "identified_untracked"}

_NAME = re.compile(r"^pumpprobe_(\d{8})_(\d{6})$")
_NEURON_LIKE = re.compile(r"^[A-Z][A-Za-z0-9]*$")


class ImportError_(Exception):
    """The inputs violate an expectation; the import stops rather than guessing."""


# --------------------------------------------------------------------------- identifiers


def recording_id(name: str) -> str:
    return f"{DATASET}:{name}"


def animal_id(name: str) -> str:
    return f"{DATASET}:animal:{name}"


def roi_id(name: str, roi: int) -> str:
    return f"{recording_id(name)}:roi{roi:03d}"


def trial_id(name: str, event: int) -> str:
    return f"{recording_id(name)}:stim{event:03d}"


def label_status(label: str) -> str:
    if not label:
        return "blank"
    if label.isdigit():
        return "numeric"
    if _NEURON_LIKE.match(label):
        return "name"
    return "other"


# --------------------------------------------------------------------------- parsing


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def parse_matrix(text: bytes, *, header: bool) -> tuple[np.ndarray, dict[str, Any]]:
    meta: dict[str, Any] = {}
    body = text
    if header:
        first, _, body = text.partition(b"\n")
        if not first.startswith(b"#"):
            raise ImportError_("expected a '#' JSON header line")
        meta = json.loads(first[1:])
    arr = np.loadtxt(BytesIO(body), dtype=np.float64, ndmin=2)
    if np.isinf(arr).any():
        raise ImportError_("infinite value in trace matrix")
    return arr, meta


def parse_vector(text: bytes, dtype: type) -> np.ndarray:
    return np.atleast_1d(np.loadtxt(BytesIO(text), dtype=dtype, ndmin=1))


def body_after_header(text: bytes) -> bytes:
    return text.partition(b"\n")[2]


@dataclass
class TextRecording:
    index: int
    name: str
    source_path: str
    labels: list[str]
    trailing_blank_labels: int
    times: np.ndarray
    stim_frames: np.ndarray
    stim_codes: np.ndarray
    gcamp_path: Path


def read_text_recording(text_dir: Path, index: int) -> TextRecording:
    def read(suffix: str) -> bytes:
        return (text_dir / f"{index}_{suffix}.txt").read_bytes()

    source_path = read("ds_name").decode().strip()
    name = PurePosixPath(source_path.rstrip("/")).name
    if not _NAME.match(name):
        raise ImportError_(f"recording {index}: unexpected recording name {name!r}")
    labels = [line.strip() for line in read("labels").decode().split("\n")]
    if labels and labels[-1] == "":
        labels.pop()  # final newline
    return TextRecording(
        index=index,
        name=name,
        source_path=source_path,
        labels=labels,
        trailing_blank_labels=0,
        times=parse_vector(read("t"), float),
        stim_frames=parse_vector(read("stim_volume_i"), int),
        stim_codes=parse_vector(read("stim_neurons"), int),
        gcamp_path=text_dir / f"{index}_gcamp.txt",
    )


def text_indices(text_dir: Path) -> list[int]:
    found: dict[int, set[str]] = {}
    for p in text_dir.iterdir():
        m = re.match(r"^(\d+)_(.+)\.txt$", p.name)
        if m:
            found.setdefault(int(m.group(1)), set()).add(m.group(2))
    incomplete = {i: sorted(set(TEXT_SUFFIXES) - s) for i, s in found.items() if set(TEXT_SUFFIXES) - s}
    if incomplete:
        raise ImportError_(f"text export is incomplete: {incomplete}")
    return sorted(found)


def iter_full_recordings(
    archive: Path, wanted: set[str], all_dirs: set[str] | None = None
) -> Iterator[tuple[str, dict[str, bytes]]]:
    """Stream the full export once, yielding the needed members of each wanted recording directory and
    recording every pumpprobe directory name in ``all_dirs``. Members of one directory must be contiguous
    in the archive; otherwise the import fails loudly."""
    needed = set(FULL_REQUIRED) | set(FULL_OPTIONAL)
    current: str | None = None
    members: dict[str, bytes] = {}
    finished: set[str] = set()
    with tarfile.open(archive, "r|gz") as tar:
        for info in tar:
            parts = PurePosixPath(info.name).parts
            if all_dirs is not None and len(parts) >= 2 and parts[1].startswith("pumpprobe_"):
                all_dirs.add(parts[1])
            if len(parts) != 3 or parts[0] != "exported_data_full" or parts[1] not in wanted:
                continue
            directory, member = parts[1], parts[2]
            if directory != current:
                if current is not None:
                    finished.add(current)
                    yield current, members
                if directory in finished:
                    raise ImportError_(f"archive members of {directory} are not contiguous")
                current, members = directory, {}
            if member in needed and info.isfile():
                f = tar.extractfile(info)
                assert f is not None
                members[member] = f.read()
        if current is not None:
            yield current, members


# --------------------------------------------------------------------------- per-recording assembly


@dataclass
class Report:
    recordings: list[dict[str, Any]] = field(default_factory=list)
    not_imported: list[dict[str, str]] = field(default_factory=list)
    label_status_counts: dict[str, int] = field(default_factory=dict)
    trial_flag_counts: dict[str, int] = field(default_factory=dict)
    target_code_counts: dict[str, int] = field(default_factory=dict)

    def bump(self, table: dict[str, int], key: str, n: int = 1) -> None:
        table[key] = table.get(key, 0) + n


def _lines(raw: bytes) -> list[str]:
    lines = raw.decode().split("\n")
    if lines and lines[-1] == "":
        lines.pop()  # final newline only; blank lines inside the file keep their position
    return lines


def _manual_targets(raw: bytes, comments: bytes | None, n_stim: int) -> tuple[list[int] | None, list[str | None]]:
    """Per-event codes from targets_manually_located.txt (None if the line count does not match the events)
    and the labels of -3 events from the comments file, both indexed by line position."""
    lines = [ln for ln in _lines(raw) if not ln.startswith("#")]
    codes = [int(ln.split(":")[0]) for ln in lines] if len(lines) == n_stim else None
    labels: list[str | None] = [None] * n_stim
    if comments is not None:
        for i, ln in enumerate(_lines(comments)[:n_stim]):
            code, _, label = ln.partition(":")
            if code.strip() == "-3" and label.strip():
                labels[i] = label.strip()
    return codes, labels


def _wallclock(name: str, hhmmss: str) -> str | None:
    m = _NAME.match(name)
    t = hhmmss.strip()
    if not m or not re.match(r"^\d{2}:\d{2}:\d{2}$", t):
        return None
    return datetime.strptime(f"{m.group(1)} {t}", "%Y%m%d %H:%M:%S").isoformat()


def build_recording(
    text: TextRecording, full: dict[str, bytes], report: Report
) -> tuple[dict[str, list[Any]], dict[str, list[Any]], dict[str, list[Any]], dict[str, list[Any]], pa.Table, pa.Table]:
    name = text.name
    missing = [m for m in FULL_REQUIRED if m not in full]
    if missing:
        raise ImportError_(f"{name}: full export lacks {missing}")

    # The OSF text trace is, column by column, exactly one of the two upstream extractions: the matched
    # trace (green.txt) or the matchless one (green_matchless.txt), as pumpprobe chooses when loading a
    # signal. Determine the choice per ROI by exact equality and take the red channel from the same source.
    gcamp_bytes = text.gcamp_path.read_bytes()
    gcamp_identical = sha256_bytes(body_after_header(full["green.txt"])) == sha256_bytes(gcamp_bytes)
    gcamp, _ = parse_matrix(gcamp_bytes, header=False)
    green, green_meta = parse_matrix(full["green.txt"], header=True)
    red, red_meta = parse_matrix(full["red.txt"], header=True)
    n_frames, n_rois = gcamp.shape
    if green.shape != gcamp.shape or red.shape != gcamp.shape or len(text.times) != n_frames:
        raise ImportError_(f"{name}: text {gcamp.shape}, green {green.shape}, red {red.shape}, {len(text.times)} times")
    # The matchless extraction can cover fewer ROIs than the recording (e.g. 161 of 200); its columns align
    # with the first ones by index. Exact whole-column equality verifies the alignment wherever it is used.
    green_ml = red_ml = None
    if "green_matchless.txt" in full and "red_matchless.txt" in full:
        green_ml, _ = parse_matrix(full["green_matchless.txt"], header=True)
        red_ml, _ = parse_matrix(full["red_matchless.txt"], header=True)
        if green_ml.shape != red_ml.shape or green_ml.shape[0] != n_frames or green_ml.shape[1] > n_rois:
            raise ImportError_(f"{name}: matchless traces have shape {green_ml.shape}, recording {gcamp.shape}")
    n_ml = green_ml.shape[1] if green_ml is not None else 0

    def same(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.all((a == b) | (np.isnan(a) & np.isnan(b)), axis=0)

    is_matched = same(gcamp, green)
    is_matchless = np.zeros(n_rois, bool)
    if green_ml is not None:
        is_matchless[:n_ml] = same(gcamp[:, :n_ml], green_ml)
    unexplained = np.where(~is_matched & ~is_matchless)[0]
    if unexplained.size:
        raise ImportError_(
            f"{name}: OSF text columns {unexplained.tolist()} match neither green.txt nor green_matchless.txt"
        )
    trace_source = np.where(is_matched, "matched", "matchless")
    red_sel = red.copy()
    use_ml = np.where(~is_matched)[0]
    if use_ml.size:
        assert red_ml is not None
        red_sel[:, use_ml] = red_ml[:, use_ml]
    if not np.all(np.diff(text.times) > 0):
        raise ImportError_(f"{name}: volume times are not strictly increasing")

    labels = text.labels
    extra = labels[n_rois:]
    if any(extra):
        raise ImportError_(f"{name}: {len(labels)} labels for {n_rois} ROIs, extra labels are not blank")
    labels = labels[:n_rois] + [""] * max(0, n_rois - len(labels))
    text.trailing_blank_labels = len(extra)

    rec = load_attrs(full["recording.pickle"])
    fc = load_attrs(full["fconn.pickle"])
    n_stim = len(text.stim_frames)
    if len(text.stim_codes) != n_stim:
        raise ImportError_(f"{name}: {n_stim} stimulation frames but {len(text.stim_codes)} target codes")
    if np.any(text.stim_frames < 0) or np.any(text.stim_frames >= n_frames):
        raise ImportError_(f"{name}: stimulation frame outside the recording")
    if np.any(text.stim_codes >= n_rois) or np.any(text.stim_codes < -3):
        raise ImportError_(f"{name}: stimulation target code out of range")

    checks = {
        "gcamp_text_bytes_equal_green": gcamp_identical,
        "matchless_rois": int((~is_matched).sum()),
        "matchless_columns_available": n_ml,
        "fconn_n_neurons_equals_rois": int(fc["n_neurons"]) == n_rois,
        "fconn_stim_indices_equal_text": np.array_equal(np.asarray(fc["stim_indices"]), text.stim_frames),
        "fconn_stim_neurons_equal_text": np.array_equal(np.asarray(fc["stim_neurons"]), text.stim_codes),
        "recording_n_volumes_equal_frames": int(rec["nVolume"]) == n_frames,
        "volume_interval_s": float(rec["Dt"]),
    }
    acq_joined = int(rec.get("optogeneticsN", -1)) == n_stim
    checks["acquisition_count_matches_events"] = acq_joined

    manual_codes, manual_labels = _manual_targets(
        full["targets_manually_located.txt"], full.get("targets_manually_located_comments.txt"), n_stim
    )
    checks["manual_targets_count_matches_events"] = manual_codes is not None
    # pumpprobe assigns -3 from the comments file using a stale loop index, which overwrites the last
    # manually processed event's code. Signature: exported -3, no "-3:label" comment for that event, and a
    # different explicit code in the manual file. Only that case is reconciled. An exported -2 is never
    # overridden: pumpprobe also writes -2 to every stimulation after a recorded bubble, deliberately.
    exported_codes = text.stim_codes.copy()
    codes = exported_codes.copy()
    reconciled: set[int] = set()
    superseded: set[int] = set()
    if manual_codes is not None:
        for e, m in enumerate(manual_codes):
            exported = int(exported_codes[e])
            if exported == -3 and manual_labels[e] is None and m not in (-1, -3) and m < n_rois:
                codes[e] = m
                reconciled.add(e)
            elif exported == -2 and m >= 0:
                superseded.add(e)
    checks["exported_codes_agree_with_manual_file"] = not reconciled
    compl = list(np.asarray(fc.get("stim_neurons_compl_labels", [None] * n_stim), dtype=object))
    checks["untracked_labels_agree_with_fconn"] = (
        all((a or None) == (b or None) for a, b in zip(manual_labels, compl, strict=False)) and len(compl) == n_stim
    )

    # ---- neurons
    counts = {lbl: labels.count(lbl) for lbl in set(labels)}
    neurons: dict[str, list[Any]] = {
        k: []
        for k in (
            "recording_id",
            "roi_index",
            "roi_id",
            "label",
            "label_status",
            "label_duplicated",
            "neuron_id",
            "trace_source",
        )
    }
    for i, lbl in enumerate(labels):
        status = label_status(lbl)
        dup = bool(lbl) and counts[lbl] > 1
        report.bump(report.label_status_counts, status + ("_duplicated" if dup else ""))
        neurons["recording_id"].append(recording_id(name))
        neurons["roi_index"].append(i)
        neurons["roi_id"].append(roi_id(name, i))
        neurons["label"].append(lbl or None)
        neurons["label_status"].append(status)
        neurons["label_duplicated"].append(dup)
        neurons["neuron_id"].append(lbl if status == "name" and not dup else None)
        neurons["trace_source"].append(str(trace_source[i]))

    # ---- trials, in upstream file order; stimulation order by frame
    order = sorted(range(n_stim), key=lambda e: (int(text.stim_frames[e]), e))
    rank = {e: r for r, e in enumerate(order)}
    frame_counts: dict[int, int] = {}
    for f in text.stim_frames:
        frame_counts[int(f)] = frame_counts.get(int(f), 0) + 1
    hit = np.asarray(fc.get("targeted_neuron_hit", [None] * n_stim), dtype=object)

    def target_label(e: int) -> str | None:
        code = int(codes[e])
        if code >= 0:
            return labels[code] or None
        return manual_labels[e] if code == -3 else None

    trials: dict[str, list[Any]] = {
        k: []
        for k in (
            "trial_id",
            "animal_id",
            "session_id",
            "recording_id",
            "event_index_in_file",
            "stim_index_in_recording",
            "stim_frame",
            "stimulus_start_s",
            "time_since_recording_start_s",
            "next_stim_frame",
            "time_since_previous_stim_s",
            "previous_stim_target_label",
            "previous_trial_id",
            "stim_target_code",
            "stim_target_code_exported",
            "stim_target_roi",
            "stim_target_label",
            "stim_target_neuron_id",
            "stimulus_id",
            "stimulus_duration_s",
            "stimulus_amplitude",
            "stimulus_amplitude_units",
            "optogenetics_type",
            "optogenetics_n_pulses",
            "optogenetics_n_trains",
            "optogenetics_rep_rate_divider",
            "optogenetics_time_between_trains",
            "optogenetics_target_xyz",
            "optogenetics_wallclock",
            "autoresponse_trace_ref",
            "autoresponse_valid_fraction",
            "autoresponse_usable",
            "stimulus_verified_on_target",
            "trial_qc_flags",
        )
    }
    auto: dict[str, list[Any]] = {
        k: [] for k in ("trial_id", "frame", "time_rel_s", "gcamp", "red", "gcamp_valid", "red_valid")
    }
    for e in range(n_stim):
        frame = int(text.stim_frames[e])
        code = int(codes[e])
        r = rank[e]
        prev_e = order[r - 1] if r > 0 else None
        next_frame = int(text.stim_frames[order[r + 1]]) if r + 1 < n_stim else n_frames
        flags = []
        if code < 0:
            flags.append("target_" + TARGET_CODES[code])
        if frame_counts[frame] > 1:
            flags.append("duplicate_stim_frame")
        if r != e:
            flags.append("file_order_differs_from_time_order")
        if e in reconciled:
            flags.append("target_code_reconciled_from_manual_file")
        if e in superseded:
            flags.append("manual_target_superseded_by_export")
        if not acq_joined:
            flags.append("acquisition_metadata_unjoined")
        for fl in flags:
            report.bump(report.trial_flag_counts, fl)
        report.bump(report.target_code_counts, "roi" if code >= 0 else TARGET_CODES[code])

        label = target_label(e)
        neuron_ids = neurons["neuron_id"]
        tid = trial_id(name, e)
        auto_ref = roi_id(name, code) if code >= 0 else None
        if code >= 0:
            window = slice(frame, next_frame)
            seg = gcamp[window, code]
            valid_fraction = float(np.isfinite(seg).mean()) if seg.size else None
            prev_frame = int(text.stim_frames[prev_e]) if prev_e is not None else 0
            for f in range(prev_frame, next_frame):
                auto["trial_id"].append(tid)
                auto["frame"].append(f)
                auto["time_rel_s"].append(float(text.times[f] - text.times[frame]))
                auto["gcamp"].append(float(gcamp[f, code]))
                auto["red"].append(float(red_sel[f, code]))
                auto["gcamp_valid"].append(bool(np.isfinite(gcamp[f, code])))
                auto["red_valid"].append(bool(np.isfinite(red_sel[f, code])))
        else:
            valid_fraction = None

        def acq(key: str, cast: type, e: int = e) -> Any:
            if not acq_joined or key not in rec:
                return None
            return cast(np.asarray(rec[key])[e])

        xyz = None
        if acq_joined and all(k in rec for k in ("optogeneticsTargetX", "optogeneticsTargetY", "optogeneticsTargetZ")):
            xyz = [
                float(rec["optogeneticsTargetX"][e]),
                float(rec["optogeneticsTargetY"][e]),
                float(rec["optogeneticsTargetZ"][e]),
            ]
        wall = None
        if acq_joined and "optogeneticsTime" in rec:
            wall = _wallclock(name, str(rec["optogeneticsTime"][e]))

        trials["trial_id"].append(tid)
        trials["animal_id"].append(animal_id(name))
        trials["session_id"].append(recording_id(name))
        trials["recording_id"].append(recording_id(name))
        trials["event_index_in_file"].append(e)
        trials["stim_index_in_recording"].append(r)
        trials["stim_frame"].append(frame)
        trials["stimulus_start_s"].append(float(text.times[frame]))
        trials["time_since_recording_start_s"].append(float(text.times[frame] - text.times[0]))
        trials["next_stim_frame"].append(next_frame if r + 1 < n_stim else None)
        trials["time_since_previous_stim_s"].append(
            float(text.times[frame] - text.times[int(text.stim_frames[prev_e])]) if prev_e is not None else None
        )
        trials["previous_stim_target_label"].append(target_label(prev_e) if prev_e is not None else None)
        trials["previous_trial_id"].append(trial_id(name, prev_e) if prev_e is not None else None)
        trials["stim_target_code"].append("roi" if code >= 0 else TARGET_CODES[code])
        trials["stim_target_code_exported"].append(int(exported_codes[e]))
        trials["stim_target_roi"].append(code if code >= 0 else None)
        trials["stim_target_label"].append(label)
        trials["stim_target_neuron_id"].append(neuron_ids[code] if code >= 0 else None)
        trials["stimulus_id"].append(f"{tid}:optogenetic")
        trials["stimulus_duration_s"].append(None)  # pulse rate not exported; never inferred
        trials["stimulus_amplitude"].append(None)
        trials["stimulus_amplitude_units"].append(None)
        trials["optogenetics_type"].append(str(rec.get("optogeneticsType")) if acq_joined else None)
        trials["optogenetics_n_pulses"].append(acq("optogeneticsNPulses", int))
        trials["optogenetics_n_trains"].append(acq("optogeneticsNTrains", int))
        trials["optogenetics_rep_rate_divider"].append(acq("optogeneticsRepRateDivider", int))
        trials["optogenetics_time_between_trains"].append(acq("optogeneticsTimeBtwTrains", float))
        trials["optogenetics_target_xyz"].append(xyz)
        trials["optogenetics_wallclock"].append(wall)
        trials["autoresponse_trace_ref"].append(auto_ref)
        trials["autoresponse_valid_fraction"].append(valid_fraction)
        trials["autoresponse_usable"].append(None)  # needs a declared criterion (OW-003)
        trials["stimulus_verified_on_target"].append(None if hit[e] is None else bool(hit[e]))
        trials["trial_qc_flags"].append(flags)

    # ---- upstream detector output: derived annotations, never response labels
    det: dict[str, list[Any]] = {k: [] for k in ("trial_id", "roi_index", "roi_id", "upstream_amplitude")}
    by_stim = fc.get("resp_neurons_by_stim", [])
    ampl = fc.get("resp_ampl_by_stim", [])
    for e in range(min(n_stim, len(by_stim))):
        rois = np.asarray(by_stim[e], dtype=int).ravel()
        amps = np.asarray(ampl[e], dtype=float).ravel() if e < len(ampl) else np.full(len(rois), np.nan)
        for j, roi in enumerate(rois):
            det["trial_id"].append(trial_id(name, e))
            det["roi_index"].append(int(roi))
            det["roi_id"].append(roi_id(name, int(roi)))
            det["upstream_amplitude"].append(float(amps[j]) if j < len(amps) else None)

    # ---- observations: recording-level, ROI-major
    frames = np.arange(n_frames, dtype=np.int32)
    obs = pa.table(
        {
            "recording_id": pa.array([recording_id(name)] * (n_frames * n_rois)).dictionary_encode(),
            "roi_index": pa.array(np.repeat(np.arange(n_rois, dtype=np.int16), n_frames)),
            "frame": pa.array(np.tile(frames, n_rois)),
            "time_s": pa.array(np.tile(text.times, n_rois)),
            "gcamp": pa.array(gcamp.T.ravel()),
            "red": pa.array(red_sel.T.ravel()),
            "gcamp_valid": pa.array(np.isfinite(gcamp.T.ravel())),
            "red_valid": pa.array(np.isfinite(red_sel.T.ravel())),
        }
    )

    animals: dict[str, list[Any]] = {
        "animal_id": [animal_id(name)],
        "session_id": [recording_id(name)],
        "recording_id": [recording_id(name)],
        "batch_id": [None],
        "strain": [None],
        "genotype": ["wild-type"],
        "sex": [None],
        "life_stage": [None],
        "preparation": [None],
        "source_dataset": [DATASET],
        "source_recording_id": [name],
        "source_text_index": [text.index],
        "source_path": [text.source_path],
        "acquisition_date": [f"{name[10:14]}-{name[14:16]}-{name[16:18]}"],
        "n_frames": [n_frames],
        "n_rois": [n_rois],
        "n_stimulations": [n_stim],
        "volume_interval_s": [float(rec["Dt"])],
        "extraction_method": [str(green_meta.get("method"))],
        "extraction_version": [str(green_meta.get("version"))],
        "quality_flags": [["animal_id_assumed_one_recording_per_animal"]],
    }
    report.recordings.append(
        {
            "name": name,
            "text_index": text.index,
            "frames": n_frames,
            "rois": n_rois,
            "stimulations": n_stim,
            "trailing_blank_labels": text.trailing_blank_labels,
            "reconciled_events": sorted(reconciled),
            "manual_targets_superseded_by_export": sorted(superseded),
            "red_header_matches_green": red_meta == green_meta,
            "checks": checks,
        }
    )
    return animals, neurons, trials, det, obs, _auto_table(auto)


def _auto_table(auto: dict[str, list[Any]]) -> pa.Table:
    return pa.table(
        {
            "trial_id": pa.array(auto["trial_id"], pa.string()),
            "frame": pa.array(auto["frame"], pa.int32()),
            "time_rel_s": pa.array(auto["time_rel_s"], pa.float64()),
            "gcamp": pa.array(auto["gcamp"], pa.float64()),
            "red": pa.array(auto["red"], pa.float64()),
            "gcamp_valid": pa.array(auto["gcamp_valid"], pa.bool_()),
            "red_valid": pa.array(auto["red_valid"], pa.bool_()),
        }
    )


# --------------------------------------------------------------------------- driver


# Checks reported for information only: a False value is expected and not a disagreement.
INFO_CHECKS = frozenset({"gcamp_text_bytes_equal_green"})


def _failed(checks: dict[str, Any]) -> list[str]:
    return [k for k, v in checks.items() if v is False and k not in INFO_CHECKS]


def _concat(parts: list[dict[str, list[Any]]]) -> dict[str, list[Any]]:
    out: dict[str, list[Any]] = {k: [] for k in parts[0]} if parts else {}
    for p in parts:
        for k, v in p.items():
            out[k].extend(v)
    return out


def _file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def _git_revision(root: Path) -> str | None:
    try:
        rev = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain", "--", "python"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        return rev + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return None


def _write(table: pa.Table, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path, compression="zstd")


def run_import(
    raw_dir: Path,
    out_dir: Path,
    *,
    inputs: list[dict[str, Any]],
    limit: int | None = None,
    repo_root: Path | None = None,
    progress: Any = None,
) -> dict[str, Any]:
    text_dir = raw_dir / TEXT_SOURCE / TEXT_DIR
    archive = raw_dir / FULL_SOURCE / FULL_ARCHIVE
    if out_dir.exists() and any(out_dir.iterdir()):
        raise ImportError_(f"{out_dir} is not empty; normalized outputs are written once")
    indices = text_indices(text_dir)
    if limit is not None:
        indices = indices[:limit]
    texts = {}
    for i in indices:
        t = read_text_recording(text_dir, i)
        if t.name in texts:
            raise ImportError_(f"duplicate recording name {t.name}")
        texts[t.name] = t

    report = Report()
    parts: dict[str, list[Any]] = {"animals": [], "neurons": [], "trials": [], "det": [], "auto": []}
    seen: set[str] = set()
    all_dirs: set[str] = set()
    for name, members in iter_full_recordings(archive, set(texts), all_dirs):
        animals, neurons, trials, det, obs, auto = build_recording(texts[name], members, report)
        _write(obs, out_dir / "observations" / f"{name}.parquet")
        for key, val in (("animals", animals), ("neurons", neurons), ("trials", trials), ("det", det)):
            parts[key].append(val)
        parts["auto"].append(auto)
        seen.add(name)
        if progress:
            progress(name, len(seen), len(texts))
    missing = sorted(set(texts) - seen)
    if missing:
        raise ImportError_(f"recordings in the text export but not in the full export: {missing}")
    for d in sorted(all_dirs - set(texts)):
        report.not_imported.append(
            {
                "recording": d,
                "reason": "not in the wild-type text export (likely unc-31; requires randi2023-atlas-exported-unc31)"
                if limit is None
                else "outside --limit",
            }
        )

    order = sorted(range(len(parts["animals"])), key=lambda i: parts["animals"][i]["source_text_index"][0])
    tables = {
        "animals": pa.table(_concat([parts["animals"][i] for i in order])),
        "neurons": pa.table(_concat([parts["neurons"][i] for i in order])),
        "trials": pa.table(_concat([parts["trials"][i] for i in order])),
        "upstream_detections": pa.table(
            _concat([parts["det"][i] for i in order])
            or {"trial_id": [], "roi_index": [], "roi_id": [], "upstream_amplitude": []}
        ),
        "autoresponses": pa.concat_tables([parts["auto"][i] for i in order]),
    }
    for tname, table in tables.items():
        _write(table, out_dir / f"{tname}.parquet")

    outputs = {}
    for path in sorted(out_dir.rglob("*.parquet")):
        rel = path.relative_to(out_dir).as_posix()
        outputs[rel] = {"sha256": _file_sha(path), "rows": pq.ParquetFile(path).metadata.num_rows}
    summary = {
        "recordings": len(seen),
        "stimulations": tables["trials"].num_rows,
        "rois": tables["neurons"].num_rows,
        "observation_rows": sum(v["rows"] for k, v in outputs.items() if k.startswith("observations/")),
        "label_status": dict(sorted(report.label_status_counts.items())),
        "stim_target": dict(sorted(report.target_code_counts.items())),
        "trial_flags": dict(sorted(report.trial_flag_counts.items())),
        "failed_checks": {r["name"]: failed for r in report.recordings if (failed := _failed(r["checks"]))},
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "dataset": OUTPUT_NAME,
        "importer": f"occamworm.importers.randi2023 {__version__}",
        "code_revision": _git_revision(repo_root) if repo_root else None,
        "inputs": inputs,
        "fluorescence_units": FLUORESCENCE_UNITS,
        "transforms": (
            "none: decimal text parsed to float64; per-ROI choice of matched or matchless extraction "
            "recorded in neurons.trace_source; frame = 0-based volume index; time_s from <n>_t.txt"
        ),
        "summary": summary,
        "outputs": outputs,
        "not_imported": report.not_imported,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (out_dir / "report.json").write_text(json.dumps({"recordings": report.recordings}, indent=2) + "\n")
    return manifest
