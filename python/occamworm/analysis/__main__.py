"""``python -m occamworm.analysis {audit,splits}``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from occamworm.sources.cli import find_root


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m occamworm.analysis")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("audit", help="M0 data audit (OW-003)")
    a.add_argument("--out", type=Path, help="default: artifacts/audit-v1")
    a.add_argument("--publish", action="store_true", help="also copy report.md to docs/data/AUDIT_REPORT.md")
    s = sub.add_parser("splits", help="freeze the split scheme chosen by the audit (OW-004)")
    s.add_argument("--audit", type=Path, help="default: artifacts/audit-v1/audit.json")
    s.add_argument("--out", type=Path, help="default: data/splits/<dataset>-<scheme>")
    args = ap.parse_args(argv)
    root = find_root(Path.cwd())

    if args.cmd == "audit":
        from occamworm.analysis.audit import AUDIT_VERSION, run_audit

        out = args.out or root / "artifacts" / AUDIT_VERSION
        result = run_audit(root, out)
        print(
            json.dumps(
                {
                    "go_no_go": result["go_no_go"]["decision"],
                    "split": result["split_selection"]["chosen"],
                    "eligible_targets": result["targets"]["eligible_targets"],
                },
                indent=2,
            )
        )
        print(f"Wrote {out}")
        if args.publish:
            (root / "docs" / "data" / "AUDIT_REPORT.md").write_text((out / "report.md").read_text())
            print("Wrote docs/data/AUDIT_REPORT.md")
        return 0

    from occamworm.analysis.freeze import freeze_split

    path = freeze_split(root, args.audit, args.out)
    print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
