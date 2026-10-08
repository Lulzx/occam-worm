"""Independent round-trip check for the Randi 2023 import (OW-002 definition of done).

Samples normalized rows at random and compares them with the upstream files using a separate, deliberately
simple code path: lines are split by hand rather than parsed with numpy. Every compared value must be
identical (or NaN on both sides); the import applies no transforms.
"""

from __future__ import annotations

import math
import random
import tarfile
from pathlib import Path, PurePosixPath
from typing import Any

import pyarrow.parquet as pq

from occamworm.importers.randi2023 import FULL_ARCHIVE, FULL_SOURCE, TEXT_DIR, TEXT_SOURCE


def _same(a: float | None, b: float) -> bool:
    if a is None:
        return False
    return (math.isnan(a) and math.isnan(b)) or a == b


class _Lines:
    """Lazy per-file line cache for the OSF text export."""

    def __init__(self, text_dir: Path) -> None:
        self.text_dir = text_dir
        self.cache: dict[str, list[str]] = {}

    def get(self, index: int, suffix: str) -> list[str]:
        key = f"{index}_{suffix}"
        if key not in self.cache:
            self.cache[key] = (self.text_dir / f"{key}.txt").read_text().splitlines()
        return self.cache[key]

    def token(self, index: int, suffix: str, line: int, column: int = 0) -> float:
        return float(self.get(index, suffix)[line].split()[column])


def roundtrip(norm_dir: Path, raw_dir: Path, *, samples: int = 10_000, seed: int = 0) -> dict[str, Any]:
    rng = random.Random(seed)
    text = _Lines(raw_dir / TEXT_SOURCE / TEXT_DIR)
    animals = pq.read_table(norm_dir / "animals.parquet").to_pylist()
    source_of = {
        (n["recording_id"], n["roi_index"]): n["trace_source"]
        for n in pq.read_table(norm_dir / "neurons.parquet").to_pylist()
    }
    index_of = {a["recording_id"]: a["source_text_index"] for a in animals}
    name_of = {a["recording_id"]: a["source_recording_id"] for a in animals}
    obs_files = sorted((norm_dir / "observations").glob("*.parquet"))
    mismatches: list[dict[str, Any]] = []
    compared = {"gcamp": 0, "red": 0, "time": 0, "trial_fields": 0, "reconciled_trials": 0, "autoresponse": 0}

    # --- observations: sample files, then rows
    picks: dict[Path, list[int]] = {}
    sizes = {f: pq.ParquetFile(f).metadata.num_rows for f in obs_files}
    for _ in range(samples):
        f = rng.choice(obs_files)
        picks.setdefault(f, []).append(rng.randrange(sizes[f]))
    red_needed: dict[tuple[str, str], list[tuple[int, int, float]]] = {}
    for f, rows in picks.items():
        table = pq.read_table(f).take(rows).to_pylist()
        for row in table:
            rid, roi, frame = row["recording_id"], row["roi_index"], row["frame"]
            idx = index_of[rid]
            expected_g = text.token(idx, "gcamp", frame, roi)
            expected_t = text.token(idx, "t", frame)
            for kind, got, exp in (("gcamp", row["gcamp"], expected_g), ("time", row["time_s"], expected_t)):
                compared[kind] += 1
                if not _same(got, exp):
                    mismatches.append(
                        {"what": kind, "recording": rid, "roi": roi, "frame": frame, "normalized": got, "upstream": exp}
                    )
            if row["gcamp_valid"] != (not math.isnan(expected_g)):
                mismatches.append({"what": "gcamp_valid", "recording": rid, "roi": roi, "frame": frame})
            member = "red.txt" if source_of[(rid, roi)] == "matched" else "red_matchless.txt"
            red_needed.setdefault((name_of[rid], member), []).append((roi, frame, row["red"]))

    # --- red channel: one streaming pass over the full export for the sampled recordings
    archive = raw_dir / FULL_SOURCE / FULL_ARCHIVE
    with tarfile.open(archive, "r|gz") as tar:
        for info in tar:
            parts = PurePosixPath(info.name).parts
            if len(parts) == 3 and (parts[1], parts[2]) in red_needed:
                fh = tar.extractfile(info)
                assert fh is not None
                lines = fh.read().decode().splitlines()[1:]  # skip the JSON header
                for roi, frame, got in red_needed.pop((parts[1], parts[2])):
                    compared["red"] += 1
                    exp = float(lines[frame].split()[roi])
                    if not _same(got, exp):
                        mismatches.append(
                            {
                                "what": "red",
                                "recording": parts[1],
                                "roi": roi,
                                "frame": frame,
                                "normalized": got,
                                "upstream": exp,
                            }
                        )
    for name, member in red_needed:
        mismatches.append({"what": "red", "recording": name, "error": f"{member} not found in archive"})

    # --- trials: schedule, timing and target against the text export
    trials = pq.read_table(norm_dir / "trials.parquet").to_pylist()
    for t in rng.sample(trials, min(len(trials), max(1, samples // 20))):
        idx, e = index_of[t["recording_id"]], t["event_index_in_file"]
        frame = int(text.token(idx, "stim_volume_i", e))
        code = int(text.token(idx, "stim_neurons", e))
        checks = {
            "stim_frame": t["stim_frame"] == frame,
            "stimulus_start_s": t["stimulus_start_s"] == text.token(idx, "t", frame),
            "stim_target_code_exported": t["stim_target_code_exported"] == code,
        }
        if "target_code_reconciled_from_manual_file" in t["trial_qc_flags"]:
            compared["reconciled_trials"] += 1
            code = t["stim_target_roi"] if t["stim_target_roi"] is not None else -1
        checks["stim_target_roi"] = t["stim_target_roi"] == (code if code >= 0 else None)
        if code >= 0:
            label = text.get(idx, "labels")[code].strip() if code < len(text.get(idx, "labels")) else ""
            checks["stim_target_label"] = t["stim_target_label"] == (label or None)
        compared["trial_fields"] += len(checks)
        for k, ok in checks.items():
            if not ok:
                mismatches.append({"what": f"trial.{k}", "trial": t["trial_id"]})

    # --- autoresponse channel: the stimulated ROI's own trace
    auto = pq.read_table(norm_dir / "autoresponses.parquet")
    if auto.num_rows:
        by_trial = {t["trial_id"]: t for t in trials}
        for row in auto.take([rng.randrange(auto.num_rows) for _ in range(max(1, samples // 10))]).to_pylist():
            t = by_trial[row["trial_id"]]
            exp = text.token(index_of[t["recording_id"]], "gcamp", row["frame"], t["stim_target_roi"])
            compared["autoresponse"] += 1
            if not _same(row["gcamp"], exp):
                mismatches.append({"what": "autoresponse", "trial": row["trial_id"], "frame": row["frame"]})

    return {
        "seed": seed,
        "samples_requested": samples,
        "compared": compared,
        "mismatches": len(mismatches),
        "first_mismatches": mismatches[:20],
        "identical": not mismatches,
        "documented_transforms": "none (exact equality required; NaN matches NaN)",
    }
