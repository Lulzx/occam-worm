"""Freeze the split scheme chosen by the M0 audit (OW-004).

The scheme comes from ``audit.json`` (coverage tables only, §2.3); nothing here looks at model scores.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from occamworm.analysis.dataset import dataset_dir, load_tables
from occamworm.analysis.splits import SplitSpec, build_split, check_trials, write_split


def freeze_split(root: Path, audit_path: Path | None = None, out_dir: Path | None = None) -> Path:
    audit_path = audit_path or root / "artifacts" / "audit-v2" / "audit.json"
    audit = json.loads(audit_path.read_text())
    directory = dataset_dir(root, audit["dataset"])
    trials, _, animals = load_tables(directory)
    chosen = audit["split_selection"]["chosen"]
    sch = next(s for s in audit["split_selection"]["schemes"] if s["name"] == chosen)
    spec = SplitSpec(sch["name"], sch["outer"], sch["outer_folds"], sch["inner_folds"], sch["seed"])
    split = build_split([str(a) for a in animals["animal_id"]], spec)
    ids = [str(x) for x in trials["trial_id"]]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate trial ids in the dataset")
    check_trials(
        split,
        dict(zip(ids, map(str, trials["animal_id"]), strict=True)),
        dict(zip(ids, map(str, trials["session_id"]), strict=True)),
    )
    out_dir = out_dir or root / "data" / "splits" / f"{audit['dataset']}-{chosen}"
    path = write_split(split, audit["dataset_manifest_sha256"], out_dir)
    sizes = np.array([len(f.test) for f in split.folds])
    (out_dir / "provenance.json").write_text(
        json.dumps(
            {
                "audit": str(audit_path.relative_to(root)) if audit_path.is_relative_to(root) else str(audit_path),
                "audit_code_revision": audit["code_revision"],
                "selection_rule": audit["split_selection"]["rule"],
                "chosen": chosen,
                "outer_folds": len(split.folds),
                "test_animals_per_fold": [int(sizes.min()), int(sizes.max())],
            },
            indent=2,
        )
        + "\n"
    )
    return path
