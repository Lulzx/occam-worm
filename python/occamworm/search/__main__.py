"""``python -m occamworm.search run --config configs/searches/atlas-demo.json``: nested search on the frozen split."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from occamworm.sources.cli import find_root

for _var in ("VECLIB_MAXIMUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")


def run(root: Path, cfg_path: Path, out: Path) -> Path:
    from occamworm.analysis.dataset import dataset_dir, load_windows
    from occamworm.analysis.provenance import file_sha256, git_revision, write_run_manifest
    from occamworm.analysis.splits import read_split
    from occamworm.baselines.benchmark import eligible_targets, load_experiment
    from occamworm.baselines.data import build_task
    from occamworm.search.candidates import Candidate, InnerResult, KernelCandidate
    from occamworm.search.locked import read_selection
    from occamworm.search.nested import run_fold
    from occamworm.search.wrl import WrlCandidate, budget_curve, enumerate_programs, search_volume

    cfg = json.loads(cfg_path.read_text())
    exp = load_experiment(root)
    ds = dataset_dir(root, exp["data"]["dataset"])
    split_path = root / exp["splits"]["split"]
    ann = root / "data" / "normalized" / "annotations-v1" / "manifest.json"
    write_run_manifest(
        out,
        {
            "kind": "nested_search",
            "search_config_sha256": file_sha256(cfg_path),
            "rules_sha256": {r: file_sha256(root / r) for r in cfg["rules"]},
            "enumeration_config_sha256": file_sha256(root / cfg["enumerate"]["config"]),
            "dataset_manifest_sha256": file_sha256(ds / "manifest.json"),
            "eligibility_sha256": file_sha256(root / exp["data"]["eligibility"]),
            "split_sha256": json.loads(split_path.read_text())["split_sha256"],
            "annotations_manifest_sha256": file_sha256(ann),
            "code": git_revision(root),
        },
    )
    data = build_task(load_windows(ds), eligible_targets(root, exp), cfg["task"], cfg["history"])
    split = read_split(split_path)
    kw = {"starts": cfg["starts"], "seed": cfg["seed"]}
    cands: list[Candidate] = [KernelCandidate(f, float(b)) for f, b in cfg["kernel_families"].items()]
    cands += [WrlCandidate((root / r).read_text(), root, name=Path(r).stem, **kw) for r in cfg["rules"]]
    en = cfg["enumerate"]
    kept, volume = search_volume(enumerate_programs(root / en["config"], en["limit"]), root, **kw)
    cands += kept[: en["max_kept"]]
    volume["offered"] = len(cands)
    (out / "search_volume.json").write_text(json.dumps(volume, indent=2, sort_keys=True) + "\n")
    order = [c.key for c in cands]
    log = out / "folds.jsonl"
    for fold in cfg["folds"]:
        if (out / f"fold{fold:03d}" / "outer.json").exists():
            continue  # resumed run: an outer fold is scored once
        res = run_fold(data, split, fold, cands, out)
        sel = read_selection(out / f"fold{fold:03d}" / "selection.json")
        inner = {r["key"]: r for r in sel["ranking"]}
        ranked = [
            InnerResult(k, 0, inner[k]["inner_nll"], {}, inner[k]["l_total_bits"], 0, 0, 0) for k in order if k in inner
        ]
        row: dict[str, Any] = {
            "fold": fold,
            "winner": res["winner"],
            "pareto": sel["pareto"],
            "outer_nll": sum(res["animal_nll"].values()),
            "test_animals": sorted(res["animal_nll"]),
            "search_log": res["search_log"],
            "budget_curve": budget_curve(ranked, cfg["budget_orders"], cfg["seed"]),
        }
        with log.open("a") as fh:
            fh.write(json.dumps(row, sort_keys=True) + "\n")
        print(f"fold {fold}: winner {res['winner']}", file=sys.stderr, flush=True)
    return log


def summarize(out: Path) -> str:
    """Markdown summary of a search run: volume, per-fold winners, Pareto fronts and budget curves."""
    from occamworm.search.locked import read_selection

    run = json.loads((out / "run.json").read_text())
    volume = json.loads((out / "search_volume.json").read_text())
    rows = [json.loads(line) for line in (out / "folds.jsonl").read_text().splitlines() if line.strip()]
    lines = [
        f"# Nested search: {out.name}",
        "",
        f"Run `{run['run_id'][:12]}`, code `{run['inputs']['code']['commit'][:12]}`, "
        f"split `{run['inputs']['split_sha256'][:12]}`.",
        "",
        f"Search volume: {volume['enumerated']} programs enumerated, gated {volume['gated']}, "
        f"{volume['kept']} kept; {volume['offered']} candidates offered per fold "
        "(kernel baselines, hand-written rules and the head of the enumeration).",
        "",
        "| fold | winner | outer NLL | Pareto front (L_total ascending) | rejected |",
        "| --- | --- | --- | --- | --- |",
    ]
    for r in rows:
        rej = len(r["search_log"].get("candidates_rejected", {}))
        lines.append(
            f"| {r['fold']} | `{r['winner']}` | {r['outer_nll']:,.1f} | {', '.join(f'`{k}`' for k in r['pareto'])} "
            f"| {rej} |"
        )
    for r in rows:
        sel = read_selection(out / f"fold{r['fold']:03d}" / "selection.json")
        lines += [
            "",
            f"## Fold {r['fold']}",
            "",
            "| candidate | inner NLL | L_total (bits) |",
            "| --- | --- | --- |",
        ]
        lines += [f"| `{x['key']}` | {x['inner_nll']:,.1f} | {x['l_total_bits']:,.0f} |" for x in sel["ranking"]]
        bc = r["budget_curve"]
        lines += [
            "",
            "Budget curve (best inner NLL after k candidates, in offer order vs the mean of random orders):",
            "",
            "| k | enumeration | random mean | random SD |",
            "| --- | --- | --- | --- |",
        ]
        lines += [
            f"| {k} | {e:,.1f} | {m:,.1f} | {sd:,.1f} |"
            for k, e, m, sd in zip(bc["k"], bc["enumeration"], bc["random_mean"], bc["random_sd"], strict=True)
        ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m occamworm.search")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="nested search on the frozen split, outer folds scored once")
    r.add_argument("--config", type=Path, required=True)
    r.add_argument("--out", type=Path, help="default: artifacts/search-<config name>")
    m = sub.add_parser("summarize", help="Markdown summary of a finished search run")
    m.add_argument("--out", type=Path, required=True, help="the run directory")
    m.add_argument("--publish", type=Path, help="also write the summary here (e.g. docs/evaluation/...)")
    args = ap.parse_args(argv)
    root = find_root(Path.cwd())
    if args.cmd == "summarize":
        text = summarize(args.out)
        (args.out / "summary.md").write_text(text)
        if args.publish:
            args.publish.write_text(text)
        print(args.out / "summary.md")
        return 0
    cfg_path = args.config.resolve()
    out = args.out or root / "artifacts" / f"search-{json.loads(cfg_path.read_text())['name']}"
    print(run(root, cfg_path, out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
