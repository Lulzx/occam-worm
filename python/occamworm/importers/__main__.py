"""``python -m occamworm.importers randi2023 {import,roundtrip}``"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from occamworm.importers import randi2023
from occamworm.importers.randi2023_roundtrip import roundtrip
from occamworm.sources import registry as reg_mod
from occamworm.sources.cli import REGISTRY_REL, find_root
from occamworm.sources.verify import format_report, verify_sources


def _inputs(reg: reg_mod.Registry, ids: list[str]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for sid in ids:
        s = reg.get(sid)
        digest = hashlib.sha256("\n".join(f"{a.path} {a.sha256}" for a in s.assets).encode()).hexdigest()
        out.append(
            {
                "source": sid,
                "version": s.version,
                "assets": len(s.assets),
                "assets_digest": digest,
                "license": s.license.label(),
            }
        )
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m occamworm.importers")
    p.add_argument("dataset", choices=["randi2023"])
    p.add_argument("command", choices=["import", "roundtrip"])
    p.add_argument("--data-dir", type=Path, help="raw data directory (default: <repo>/data/raw)")
    p.add_argument(
        "--out", type=Path, help=f"normalized output (default: <repo>/data/normalized/{randi2023.OUTPUT_NAME})"
    )
    p.add_argument("--limit", type=int, help="import only the first N recordings (for development)")
    p.add_argument("--skip-verify", action="store_true", help="do not re-hash inputs (development only)")
    p.add_argument("--samples", type=int, default=10_000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    root = find_root(Path.cwd())
    raw = args.data_dir or root / "data" / "raw"
    out = args.out or root / "data" / "normalized" / randi2023.OUTPUT_NAME
    ids = [randi2023.TEXT_SOURCE, randi2023.FULL_SOURCE]

    if args.command == "roundtrip":
        result = roundtrip(out, raw, samples=args.samples, seed=args.seed)
        (out / "roundtrip.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({k: v for k, v in result.items() if k != "first_mismatches"}, indent=2))
        return 0 if result["identical"] else 1

    reg = reg_mod.load(root / REGISTRY_REL)
    if not args.skip_verify:
        report = verify_sources([reg.get(i) for i in ids], raw)
        if not report.integrity_ok:
            print(format_report(report))
            print("Inputs failed verification; not importing.", file=sys.stderr)
            return 1
    manifest = randi2023.run_import(
        raw,
        out,
        inputs=_inputs(reg, ids),
        limit=args.limit,
        repo_root=root,
        progress=lambda name, i, n: print(f"  [{i}/{n}] {name}", flush=True),
    )
    print(json.dumps(manifest["summary"], indent=2))
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
