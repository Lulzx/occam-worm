"""Check local raw data against the registry and report missing, corrupted, unpinned and unlicensed assets."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from occamworm.sources.fetch import asset_path, check, digest_file
from occamworm.sources.registry import RETRIEVAL_FILE, Source

# Asset statuses. Only "ok" passes; "unpinned" fails because nothing proves the bytes are the right ones.
OK, MISSING, MISMATCH, UNPINNED = "ok", "missing", "mismatch", "unpinned"

LICENSE_ADVICE = {
    "unlicensed": "no license declared upstream: treat as all rights reserved; do not redistribute, "
    "reference by metadata only",
    "unknown": "license not established: check before any use beyond local analysis",
    "copyleft": "copyleft: use as an external reference only; do not vendor or import into Apache-2.0 code "
    "without review (open decision 12)",
    "other": "non-standard license: review terms before redistribution",
}


@dataclass
class AssetCheck:
    path: str
    status: str
    size: int | None
    detail: str = ""
    md5_only: bool = False


@dataclass
class SourceCheck:
    source: Source
    assets: list[AssetCheck]
    untracked: list[str] = field(default_factory=list)

    def count(self, status: str) -> int:
        return sum(1 for a in self.assets if a.status == status)

    @property
    def ok(self) -> bool:
        return all(a.status == OK for a in self.assets)


@dataclass
class Report:
    checks: list[SourceCheck]
    quick: bool = False

    @property
    def integrity_ok(self) -> bool:
        return all(c.ok for c in self.checks)

    def license_issues(self) -> list[SourceCheck]:
        return [c for c in self.checks if c.source.license.category in LICENSE_ADVICE]

    def to_dict(self) -> dict[str, Any]:
        return {
            "quick": self.quick,
            "integrity_ok": self.integrity_ok,
            "sources": [
                {
                    "id": c.source.id,
                    "version": c.source.version,
                    "license": {
                        "spdx": c.source.license.spdx,
                        "status": c.source.license.status,
                        "category": c.source.license.category,
                    },
                    "counts": {s: c.count(s) for s in (OK, MISSING, MISMATCH, UNPINNED)},
                    "assets": [
                        {"path": a.path, "status": a.status, "size": a.size, "detail": a.detail, "md5_only": a.md5_only}
                        for a in c.assets
                    ],
                    "untracked": c.untracked,
                }
                for c in self.checks
            ],
        }


def _untracked(source: Source, data_dir: Path) -> list[str]:
    root = data_dir / source.id
    if not root.is_dir():
        return []
    known = {a.path for a in source.assets} | {RETRIEVAL_FILE}
    found = [p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()]
    return sorted(f for f in found if f not in known)


def verify_sources(sources: list[Source], data_dir: Path, *, quick: bool = False) -> Report:
    """``quick`` compares sizes only; the default hashes every byte."""
    checks = []
    for s in sources:
        results = []
        for a in s.assets:
            p = asset_path(data_dir, s, a)
            md5_only = a.sha256 is None and a.md5 is not None
            if not p.is_file():
                results.append(AssetCheck(a.path, MISSING, a.size))
                continue
            if quick:
                size = p.stat().st_size
                if a.size is not None and size != a.size:
                    results.append(AssetCheck(a.path, MISMATCH, a.size, f"size {size} != expected {a.size}"))
                else:
                    results.append(AssetCheck(a.path, OK if a.pinned else UNPINNED, a.size, md5_only=md5_only))
                continue
            problem = check(a, digest_file(p))
            if problem is not None:
                results.append(AssetCheck(a.path, MISMATCH, a.size, problem))
            elif not a.pinned:
                results.append(AssetCheck(a.path, UNPINNED, a.size, "no expected checksum; run `pin`"))
            else:
                results.append(AssetCheck(a.path, OK, a.size, md5_only=md5_only))
        checks.append(SourceCheck(s, results, _untracked(s, data_dir)))
    return Report(checks, quick)


def human_size(n: int | None) -> str:
    if n is None:
        return "size unknown"
    size = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1000 or unit == "GB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1000
    return f"{n} B"


def format_report(report: Report, *, verbose: bool = False, limit: int = 5) -> str:
    lines = []
    width = max((len(c.source.id) for c in report.checks), default=10)
    for c in report.checks:
        n = len(c.assets)
        flag = "ok" if c.ok else ("MISSING" if c.count(MISSING) == n else "FAIL")
        lines.append(
            f"  {flag:<8} {c.source.id:<{width}}  {c.count(OK):>4}/{n:<4} assets  license: {c.source.license.label()}"
        )

    for status, title in ((MISMATCH, "Checksum mismatch"), (MISSING, "Missing"), (UNPINNED, "Unpinned")):
        bad = [(c, a) for c in report.checks for a in c.assets if a.status == status]
        if not bad:
            continue
        total = sum(a.size or 0 for _, a in bad)
        lines += ["", f"{title} ({len(bad)} assets, {human_size(total)}):"]
        for c in report.checks:
            rows = [a for a in c.assets if a.status == status]
            shown = rows if verbose else rows[:limit]
            for a in shown:
                extra = f"  {a.detail}" if a.detail else ""
                lines.append(f"  {c.source.id}  {a.path}  ({human_size(a.size)}){extra}")
            if len(rows) > len(shown):
                lines.append(f"  {c.source.id}  ... and {len(rows) - len(shown)} more (use --verbose)")

    weak = [(c, a) for c in report.checks for a in c.assets if a.status == OK and a.md5_only]
    if weak:
        lines += ["", f"Verified by md5 only ({len(weak)} assets); run `pin` to record sha256."]

    untracked = [(c, u) for c in report.checks for u in c.untracked]
    if untracked:
        lines += ["", f"Untracked files in data/raw ({len(untracked)}), not in the registry:"]
        lines += [f"  {c.source.id}  {u}" for c, u in untracked[: None if verbose else limit]]

    issues = report.license_issues()
    if issues:
        lines += ["", "License issues:"]
        for c in issues:
            lic = c.source.license
            lines.append(f"  {c.source.id}  [{lic.label()}]  {LICENSE_ADVICE[lic.category]}")

    lines.append("")
    if report.integrity_ok:
        mode = "sizes only (--quick)" if report.quick else "sha256/md5"
        lines.append(f"Result: OK, every selected asset is present and verified ({mode}).")
    else:
        counts = {s: sum(c.count(s) for c in report.checks) for s in (MISMATCH, MISSING, UNPINNED)}
        parts = ", ".join(f"{v} {k}" for k, v in counts.items() if v)
        lines.append(f"Result: FAIL ({parts}).")
    return "\n".join(lines)


def write_json(report: Report, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.to_dict(), indent=2) + "\n")
