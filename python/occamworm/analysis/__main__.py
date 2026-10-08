"""``python -m occamworm.analysis {audit,splits,freeze,report}``."""

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
    a.add_argument("--out", type=Path, help="default: artifacts/audit-v2")
    a.add_argument("--publish", action="store_true", help="also copy report.md to docs/data/AUDIT_REPORT.md")
    s = sub.add_parser("splits", help="freeze the split scheme chosen by the audit (OW-004)")
    s.add_argument("--audit", type=Path, help="default: artifacts/audit-v2/audit.json")
    s.add_argument("--out", type=Path, help="default: data/splits/<dataset>-<scheme>")
    f = sub.add_parser("freeze", help="freeze a finished run directory (sha256 of every file)")
    f.add_argument("--run", type=Path, required=True)
    r = sub.add_parser("report", help="build a traceable report from a frozen run (OW-015)")
    r.add_argument("--run", type=Path, required=True)
    r.add_argument("--out", type=Path, help="default: artifacts/reports/<run name>")
    r.add_argument("--publish", type=Path, help="also copy report.md and figures into this docs directory")
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

    if args.cmd == "freeze":
        from occamworm.analysis.provenance import freeze

        body = freeze(args.run)
        print(f"Froze {len(body['files'])} files of run {body['run_id']}")
        return 0
    if args.cmd == "report":
        import shutil

        from occamworm.analysis.reporting import build_report

        out = args.out or root / "artifacts" / "reports" / args.run.name
        path = build_report(args.run, out)
        print(f"Wrote {path}")
        if args.publish:
            args.publish.mkdir(parents=True, exist_ok=True)
            for p in out.iterdir():
                if p.suffix in (".md", ".svg", ".json"):
                    shutil.copy2(p, args.publish / p.name)
            print(f"Published to {args.publish}")
        return 0

    from occamworm.analysis.freeze import freeze_split

    path = freeze_split(root, args.audit, args.out)
    print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
