"""OW-015: a published table cell traces to its exact model, folds, sample ids and code hashes."""

import json
import re
from pathlib import Path

import pytest

from occamworm.analysis.provenance import ProvenanceError, freeze, write_run_manifest
from occamworm.analysis.reporting import build_report, trace
from occamworm.baselines.summary import summarize

ROOT = Path(__file__).resolve().parents[3]


def fake_run(tmp: Path) -> Path:
    run = tmp / "run"
    write_run_manifest(run, {"kind": "test", "code": {"commit": "abc123", "dirty": False}, "split_sha256": "s" * 64})
    rows = []
    for fold, animal in enumerate(["a0", "a1", "a2"]):
        for fam, base in (("B0", 10.0), ("B3", 8.0)):
            rows.append(
                {
                    "config": "T2a-nohist",
                    "task": "T2a",
                    "history": False,
                    "fold": fold,
                    "family": fam,
                    "lam_rel": 1.0,
                    "inner_scores": {"1.0": base},
                    "inner_best": base + fold,
                    "noise": {"a": 1.0, "b": 0.1, "phi": 0.4, "family": "gaussian_ar1"},
                    "animal_nll": {animal: base + fold},
                    "animal_traces": {animal: 5},
                    "animal_samples": {animal: 100},
                    "mse": 0.01,
                    "coverage": {"0.50": 0.5, "0.80": 0.8, "0.95": 0.95},
                    "n_params": 3,
                    "seconds": 1.0,
                    "test_animals": [animal],
                    "test_trials": [f"{animal}:stim{k}" for k in range(2)],
                }
            )
    (run / "folds.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    summarize(ROOT, run)
    return run


def test_report_cells_trace_to_model_folds_samples_and_code(tmp_path: Path) -> None:
    run = fake_run(tmp_path)
    with pytest.raises(ProvenanceError):
        build_report(run, tmp_path / "report")  # not frozen yet
    freeze(run)
    md = build_report(run, tmp_path / "report").read_text()
    ids = re.findall(r'<sup id="(c\d+)">', md)
    assert ids
    for cid in ids:
        p = trace(tmp_path / "report", cid)
        assert p["code"]["commit"] == "abc123"
        assert p["run_id"] and p["folds"] and p["sample_ids"] and p["model"]["family"]
    nll_b3 = next(
        c
        for c in ids
        if trace(tmp_path / "report", c)["quantity"] == "nll_total"
        and trace(tmp_path / "report", c)["model"]["family"] == "B3"
    )
    p = trace(tmp_path / "report", nll_b3)
    assert p["value"] == pytest.approx(8.0 * 3 + 3)
    assert p["folds"] == [0, 1, 2]
    assert p["sample_ids"] == sorted(f"a{i}:stim{k}" for i in range(3) for k in range(2))


def test_modified_artifact_is_refused(tmp_path: Path) -> None:
    run = fake_run(tmp_path)
    freeze(run)
    with (run / "folds.jsonl").open("a") as fh:
        fh.write("\n")
    with pytest.raises(ProvenanceError, match="changed"):
        build_report(run, tmp_path / "report")


def test_run_manifest_is_immutable(tmp_path: Path) -> None:
    write_run_manifest(tmp_path, {"x": 1})
    write_run_manifest(tmp_path, {"x": 1})
    with pytest.raises(ProvenanceError):
        write_run_manifest(tmp_path, {"x": 2})
