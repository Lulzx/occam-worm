"""Reports from frozen run artifacts only (§14.7, §17; OW-015).

``build_report`` refuses runs that are not frozen or whose files changed since freezing. Every number in a
published table is a ``Cell`` with an id; ``provenance.json`` maps each id to the run, the exact model
(task, history, family), the outer folds, the held-out animals and trial ids that produced it, the code revision
and the hashes of every input, so ``trace(report_dir, cell_id)`` answers "where did this number come from".
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from occamworm.analysis.provenance import ProvenanceError, verify_frozen
from occamworm.plotting.svg import forest


@dataclass
class Cell:
    value: Any
    provenance: dict[str, Any]
    cell_id: str = ""


@dataclass
class Report:
    title: str
    cells: list[Cell] = field(default_factory=list)
    lines: list[str] = field(default_factory=list)

    def cell(self, value: Any, provenance: dict[str, Any], fmt: str = "{:.4g}") -> str:
        c = Cell(value, provenance, f"c{len(self.cells) + 1}")
        self.cells.append(c)
        shown = fmt.format(value) if isinstance(value, float) else str(value)
        return f'{shown}<sup id="{c.cell_id}">{c.cell_id}</sup>'


def _load(run: Path) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    frozen = verify_frozen(run)
    manifest = json.loads((run / "run.json").read_text())
    if frozen["run_id"] != manifest["run_id"]:
        raise ProvenanceError("frozen.json and run.json disagree on the run id")
    rows = [json.loads(x) for x in (run / "folds.jsonl").read_text().splitlines() if x.strip()]
    return frozen, manifest, rows


def build_report(run: Path, out: Path) -> Path:
    frozen, manifest, rows = _load(run)
    summary = json.loads((run / "summary.json").read_text())
    control = (
        json.loads((run / "indicator_control.json").read_text()) if (run / "indicator_control.json").exists() else None
    )
    by: dict[tuple[str, bool, str], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        if "family" in r:
            by[(r["task"], r["history"], r["family"])].append(r)
    base = {
        "run_id": manifest["run_id"],
        "code": manifest["inputs"]["code"],
        "inputs": {k: v for k, v in manifest["inputs"].items() if k != "code"},
        "artifact_files": {k: frozen["files"][k] for k in ("folds.jsonl", "summary.json") if k in frozen["files"]},
    }

    def prov(task: str, hist: bool, fam: str, quantity: str) -> dict[str, Any]:
        rs = sorted(by[(task, hist, fam)], key=lambda r: r["fold"])
        return {
            **base,
            "quantity": quantity,
            "model": {
                "task": task,
                "history": hist,
                "family": fam,
                "ridge_by_fold": {r["fold"]: r["lam_rel"] for r in rs},
            },
            "folds": [r["fold"] for r in rs],
            "test_animals": sorted({a for r in rs for a in r.get("test_animals", r["animal_nll"].keys())}),
            "sample_ids": sorted({t for r in rs for t in r.get("test_trials", [])}),
        }

    rep = Report(f"Baseline benchmark report — run {manifest['run_id'][:12]}")
    L = rep.lines
    L += [
        f"# {rep.title}",
        "",
        f"Built from frozen run `{run.name}` (run id `{manifest['run_id']}`), code "
        f"`{(manifest['inputs']['code']['commit'] or 'unknown')[:12]}`"
        f"{' (dirty tree)' if manifest['inputs']['code']['dirty'] else ''}. Every number carries a cell id; "
        "`provenance.json` maps it to the model, folds, held-out animals, trial ids and input hashes.",
        "",
    ]
    figures: list[tuple[str, str]] = []
    for task, table in summary["tables"].items():
        L += [
            f"## {task}{' (primary)' if task == summary['primary_task'] else ''}",
            "",
            "| family | history | animals | NLL | NLL/sample | MSE | cov95 |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
        for row in table:
            f, h = row["family"], row["history"]
            L.append(
                f"| {f} | {'yes' if h else 'no'} | {row['n_animals']} | "
                f"{rep.cell(row['nll_total'], prov(task, h, f, 'nll_total'), '{:,.1f}')} | "
                f"{rep.cell(row['nll_per_sample'], prov(task, h, f, 'nll_per_sample'))} | "
                f"{rep.cell(row['mse'], prov(task, h, f, 'mse_mean_over_folds'))} | "
                f"{rep.cell(row['coverage']['0.95'], prov(task, h, f, 'coverage_0.95_mean_over_folds'))} |"
            )
        L += ["", "| contrast | history | ΔNLL mean | 95% CI | animals |", "| --- | --- | --- | --- | --- |"]
        forest_rows = []
        for c in summary["contrasts"].get(task, []):
            if not c.get("n_animals"):
                continue
            hist_flag = c["history"] is True
            p = prov(task, hist_flag, c["candidate"], f"paired ΔNLL {c['baseline']} -> {c['candidate']}")
            p["baseline_model"] = prov(task, hist_flag, c["baseline"], "baseline")["model"]
            p["bootstrap"] = {"reps": c["reps"], "seed": c["seed"], "unit": "animal"}
            L.append(
                f"| {c['baseline']} → {c['candidate']} | {c['history']} | {rep.cell(c['mean'], p)} | "
                f"{c['mean_ci'][0]:.4g} to {c['mean_ci'][1]:.4g} | {c['n_animals']} |"
            )
            if c["history"] is False:
                forest_rows.append((f"{c['baseline']}→{c['candidate']}", c["mean"], *c["mean_ci"]))
        if forest_rows:
            name = f"forest_{task}.svg"
            figures.append(
                (
                    name,
                    forest(
                        forest_rows, f"{task}: paired ΔNLL per animal (no history)", "ΔNLL (positive favours candidate)"
                    ),
                )
            )
            L += ["", f"![{task} contrasts]({name})", ""]
    if control:
        fs = {f["task"]: f["false_sharing_rate"] for f in control["false_sharing"]}
        cprov = {
            **base,
            "quantity": "indicator control (§4.5)",
            "artifact": "indicator_control.json",
            "artifact_sha256": frozen["files"].get("indicator_control.json"),
        }
        L += [
            "## Indicator-kinetics control",
            "",
            f"- Indicator τr {rep.cell(control['indicator']['tau_r_s'], cprov)} s, "
            f"τd {rep.cell(control['indicator']['tau_d_s'], cprov)} s "
            f"({control['indicator']['interpretation']}).",
            f"- Resolvable bandwidth {rep.cell(control['resolvable_bandwidth']['f_resolvable_hz'], cprov)} Hz.",
            "- False-sharing rate: "
            + ", ".join(f"{t} {rep.cell(v, cprov)}" for t, v in fs.items())
            + f" (threshold {control['threshold']}).",
            f"- H1 may be stated as neural: {control['interpretation']['h1_claim_allowed_as_neural']}.",
            "",
        ]
    out.mkdir(parents=True, exist_ok=True)
    for name, svg in figures:
        (out / name).write_text(svg)
    (out / "report.md").write_text("\n".join(L) + "\n")
    (out / "provenance.json").write_text(
        json.dumps({c.cell_id: {"value": c.value, **c.provenance} for c in rep.cells}, indent=1, sort_keys=True) + "\n"
    )
    return out / "report.md"


def trace(report_dir: Path, cell_id: str) -> dict[str, Any]:
    prov: dict[str, Any] = json.loads((report_dir / "provenance.json").read_text())
    if cell_id not in prov:
        raise KeyError(f"no cell {cell_id} in {report_dir}")
    return dict(prov[cell_id])
