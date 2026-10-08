from __future__ import annotations

import io
import json
import pickle
import sys
import tarfile
import types
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.compute
import pyarrow.parquet as pq
import pytest

from occamworm.importers import randi2023 as R
from occamworm.importers._safe_pickle import load_attrs
from occamworm.importers.randi2023_roundtrip import roundtrip

NAMES = ["pumpprobe_20210101_100000", "pumpprobe_20210102_110000"]
EXTRA = "pumpprobe_20230101_120000"  # in the full export only, like the unc-31 recordings
N_FRAMES, N_ROIS = 40, 4


def _fmt(matrix: np.ndarray) -> bytes:
    buf = io.StringIO()
    np.savetxt(buf, matrix, fmt="%.18e")
    return buf.getvalue().encode()


def _upstream_pickle(module: str, cls: str, attrs: dict[str, Any]) -> bytes:
    """Pickle an object whose class claims to live in an upstream module, then forget the module."""
    parts = module.split(".")
    for i in range(1, len(parts) + 1):
        sys.modules.setdefault(".".join(parts[:i]), types.ModuleType(".".join(parts[:i])))
    klass = type(cls, (), {"__module__": module})
    setattr(sys.modules[module], cls, klass)
    obj = klass()
    obj.__dict__.update(attrs)
    try:
        return pickle.dumps(obj, protocol=4)
    finally:
        for i in range(len(parts), 0, -1):
            sys.modules.pop(".".join(parts[:i]), None)


def _recording(seed: int, frames: list[int], codes: list[int]) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    green = rng.uniform(1, 100, (N_FRAMES, N_ROIS))
    green[rng.random(green.shape) < 0.2] = np.nan
    red = rng.uniform(10, 500, (N_FRAMES, N_ROIS))
    red[np.isnan(green)] = np.nan
    # Matchless extraction: what pumpprobe substitutes for poorly matched ROIs.
    green_ml = rng.uniform(1, 100, (N_FRAMES, N_ROIS))
    red_ml = rng.uniform(10, 500, (N_FRAMES, N_ROIS))
    return {
        "green": green,
        "red": red,
        "green_ml": green_ml,
        "red_ml": red_ml,
        "frames": frames,
        "codes": codes,
        "times": np.arange(N_FRAMES) * 0.5,
    }


@pytest.fixture
def atlas(tmp_path: Path) -> Path:
    raw = tmp_path / "raw"
    text_dir = raw / R.TEXT_SOURCE / R.TEXT_DIR
    text_dir.mkdir(parents=True)
    recs = {
        # Event order in the file differs from time order; the last event is -3 in the export but -2
        # in the manual file (the upstream stale-index case).
        NAMES[0]: _recording(0, frames=[5, 25, 15, 33], codes=[0, -3, 2, -3]),
        NAMES[1]: _recording(1, frames=[3, 12, 20], codes=[0, -1, -2]),
    }
    # NAMES[1] event 2: exported -2 (after a bubble upstream) although the manual file names ROI 3.
    manual = {NAMES[0]: ["-1", "-3", "-1", "-2"], NAMES[1]: ["-1", "-1", "3"]}
    comments = {NAMES[0]: ["-1", "-3:AVJL", "", "-2"], NAMES[1]: ["-1", "-1", "-2"]}
    labels = {NAMES[0]: ["AVAL", "", "RIG", "RIG", "", ""], NAMES[1]: ["SMDVL", "41", "smthng else", "AWBL"]}
    for i, name in enumerate(NAMES):
        r = recs[name]
        gcamp = r["green"].copy()
        if name == NAMES[0]:
            gcamp[:, 3] = r["green_ml"][:, 3]  # the OSF text uses the matchless trace for ROI 3
        files = {
            "gcamp": _fmt(gcamp),
            "t": _fmt(r["times"][:, None]),
            "labels": ("\n".join(labels[name]) + "\n").encode(),
            "stim_neurons": ("\n".join(map(str, r["codes"])) + "\n").encode(),
            "stim_volume_i": ("\n".join(map(str, r["frames"])) + "\n").encode(),
            "ds_name": f"/projects/LEIFER/x/{name}/\n".encode(),
        }
        for suffix, data in files.items():
            (text_dir / f"{i}_{suffix}.txt").write_bytes(data)

    archive = raw / R.FULL_SOURCE / R.FULL_ARCHIVE
    archive.parent.mkdir(parents=True)
    header = b'# {"method": "box", "version": "test"}\n'
    with tarfile.open(archive, "w:gz") as tar:

        def add(path: str, data: bytes) -> None:
            info = tarfile.TarInfo(path)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))

        add("exported_data_full/export.sh", b"#!/bin/sh\n")
        for name in [*NAMES, EXTRA]:
            r = recs.get(name, recs[NAMES[1]])
            n = len(r["frames"])
            d = f"exported_data_full/{name}/"
            add(d + "brains.json", b"{}")
            add(d + "green.txt", header + _fmt(r["green"]))
            add(d + "red.txt", header + _fmt(r["red"]))
            add(d + "green_matchless.txt", header + _fmt(r["green_ml"]))
            add(d + "red_matchless.txt", header + _fmt(r["red_ml"]))
            add(
                d + "targets_manually_located.txt",
                ("# -1 means not manually processed\n" + "\n".join(manual.get(name, ["-1"] * n)) + "\n").encode(),
            )
            comment_lines = comments.get(name, ["-1"] * n)
            add(d + "targets_manually_located_comments.txt", ("\n".join(comment_lines) + "\n").encode())
            compl = np.array(
                ["AVJL" if c == -3 and name == NAMES[0] and k == 1 else None for k, c in enumerate(r["codes"])],
                dtype=object,
            )
            add(
                d + "fconn.pickle",
                _upstream_pickle(
                    "pumpprobe.Fconn",
                    "Fconn",
                    {
                        "n_neurons": N_ROIS,
                        "Dt": 0.5,
                        "stim_indices": np.array(r["frames"]),
                        "stim_neurons": np.array(r["codes"]),
                        "stim_neurons_compl_labels": compl,
                        "targeted_neuron_hit": np.array([True] * n),
                        "resp_neurons_by_stim": [np.array([0, 2])] * n,
                        "resp_ampl_by_stim": [np.array([1.5, 2.5])] * n,
                    },
                ),
            )
            add(
                d + "recording.pickle",
                _upstream_pickle(
                    "wormdatamodel.data.recording",
                    "recording",
                    {
                        "nVolume": N_FRAMES,
                        "Dt": 0.5,
                        "optogeneticsN": n,
                        "optogeneticsType": "twoPhoton",
                        "optogeneticsNPulses": np.full(n, 250000),
                        "optogeneticsNTrains": np.ones(n, int),
                        "optogeneticsRepRateDivider": np.ones(n, int),
                        "optogeneticsTimeBtwTrains": np.zeros(n),
                        "optogeneticsTargetX": np.arange(n, dtype=float),
                        "optogeneticsTargetY": np.zeros(n),
                        "optogeneticsTargetZ": np.zeros(n),
                        "optogeneticsTime": [f"10:00:0{k}\n" for k in range(n)],
                    },
                ),
            )
    return raw


def _run(raw: Path, out: Path) -> dict[str, Any]:
    return R.run_import(raw, out, inputs=[{"source": "test"}])


def test_import_tables(atlas: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    manifest = _run(atlas, out)
    assert manifest["summary"]["recordings"] == 2
    assert manifest["not_imported"] == [{"recording": EXTRA, "reason": manifest["not_imported"][0]["reason"]}]
    assert "pumpprobe" not in sys.modules and "wormdatamodel" not in sys.modules

    neurons = pq.read_table(out / "neurons.parquet").to_pylist()
    first = [n for n in neurons if n["recording_id"] == R.recording_id(NAMES[0])]
    assert [n["label_status"] for n in first] == ["name", "blank", "name", "name"]
    assert [n["neuron_id"] for n in first] == ["AVAL", None, None, None]  # duplicated RIG is not an identity
    second = {n["label"]: n["label_status"] for n in neurons if n["recording_id"] == R.recording_id(NAMES[1])}
    assert second == {"SMDVL": "name", "41": "numeric", "smthng else": "other", "AWBL": "name"}

    trials = {t["trial_id"]: t for t in pq.read_table(out / "trials.parquet").to_pylist()}
    t = [trials[R.trial_id(NAMES[0], e)] for e in range(4)]
    assert [x["stim_index_in_recording"] for x in t] == [0, 2, 1, 3]
    assert t[2]["previous_trial_id"] == t[0]["trial_id"] and t[2]["previous_stim_target_label"] == "AVAL"
    assert t[2]["time_since_previous_stim_s"] == pytest.approx(5.0)
    assert t[0]["stim_target_neuron_id"] == "AVAL" and t[0]["autoresponse_trace_ref"].endswith("roi000")
    assert t[1]["stim_target_code"] == "identified_untracked" and t[1]["stim_target_label"] == "AVJL"
    assert t[1]["stim_target_roi"] is None and t[1]["autoresponse_trace_ref"] is None
    assert "file_order_differs_from_time_order" in t[1]["trial_qc_flags"]
    # Stale-index case: exported -3, manual file -2.
    assert t[3]["stim_target_code_exported"] == -3 and t[3]["stim_target_code"] == "failed_targeting"
    assert "target_code_reconciled_from_manual_file" in t[3]["trial_qc_flags"]
    u = trials[R.trial_id(NAMES[1], 2)]
    assert u["stim_target_code"] == "failed_targeting" and u["stim_target_roi"] is None
    assert "manual_target_superseded_by_export" in u["trial_qc_flags"]
    assert "target_code_reconciled_from_manual_file" not in u["trial_qc_flags"]
    assert t[0]["optogenetics_n_pulses"] == 250000 and t[0]["optogenetics_wallclock"] == "2021-01-01T10:00:00"
    assert t[0]["stimulus_duration_s"] is None and t[0]["stimulus_verified_on_target"] is True

    animals = pq.read_table(out / "animals.parquet").to_pylist()
    assert [a["acquisition_date"] for a in animals] == ["2021-01-01", "2021-01-02"]
    assert all("animal_id_assumed_one_recording_per_animal" in a["quality_flags"] for a in animals)

    assert [n["trace_source"] for n in first] == ["matched", "matched", "matched", "matchless"]
    obs = pq.read_table(out / "observations" / f"{NAMES[0]}.parquet")
    assert obs.num_rows == N_FRAMES * N_ROIS
    roi3 = obs.filter(pa.compute.equal(obs["roi_index"], 3))
    rec0 = _recording(0, [], [])
    assert np.array_equal(roi3["red"].to_numpy(), rec0["red_ml"][:, 3])
    assert np.array_equal(roi3["gcamp"].to_numpy(), rec0["green_ml"][:, 3])
    auto = pq.read_table(out / "autoresponses.parquet").to_pylist()
    assert {a["trial_id"] for a in auto} == {R.trial_id(NAMES[0], 0), R.trial_id(NAMES[0], 2), R.trial_id(NAMES[1], 0)}
    report = json.loads((out / "report.json").read_text())["recordings"]
    assert report[0]["reconciled_events"] == [3] and report[0]["trailing_blank_labels"] == 2


def test_roundtrip_identical_then_detects_corruption(atlas: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    _run(atlas, out)
    result = roundtrip(out, atlas, samples=500, seed=1)
    assert result["identical"], result["first_mismatches"]
    assert result["compared"]["red"] == 500

    path = out / "observations" / f"{NAMES[1]}.parquet"
    table = pq.read_table(path)
    gcamp = table.column("gcamp").to_numpy(zero_copy_only=False).copy()
    gcamp[np.isfinite(gcamp)] += 1e-9
    pq.write_table(table.set_column(table.schema.get_field_index("gcamp"), "gcamp", pa.array(gcamp)), path)
    bad = roundtrip(out, atlas, samples=500, seed=1)
    assert not bad["identical"] and any(m["what"] == "gcamp" for m in bad["first_mismatches"])


def test_rejects_text_matching_neither_extraction(atlas: Path, tmp_path: Path) -> None:
    f = atlas / R.TEXT_SOURCE / R.TEXT_DIR / "0_gcamp.txt"
    f.write_bytes(f.read_bytes().replace(b"e+01", b"e+02", 1))
    with pytest.raises(R.ImportError_, match="match neither green.txt nor green_matchless.txt"):
        _run(atlas, tmp_path / "out")


def test_refuses_to_overwrite_outputs(atlas: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    out.mkdir()
    (out / "keep.txt").write_text("x")
    with pytest.raises(R.ImportError_, match="not empty"):
        _run(atlas, out)


def test_unpickler_blocks_everything_else() -> None:
    class Evil:
        def __reduce__(self) -> tuple[Any, ...]:
            return (print, ("pwned",))

    with pytest.raises(pickle.UnpicklingError, match="blocked global"):
        load_attrs(pickle.dumps(Evil()))
    with pytest.raises(pickle.UnpicklingError, match="expected an upstream object"):
        load_attrs(pickle.dumps(np.arange(3)))


@pytest.mark.parametrize(
    "label, status",
    [
        ("", "blank"),
        ("41", "numeric"),
        ("AMsoL", "name"),
        ("IL2V", "name"),
        ("smthng else", "other"),
        ("mc77", "other"),
    ],
)
def test_label_status(label: str, status: str) -> None:
    assert R.label_status(label) == status
