"""Command line: ``python -m occamworm.sources {list,fetch,verify,pin,refresh}``."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from occamworm.sources import registry as reg_mod
from occamworm.sources.fetch import FetchResult, fetch_sources
from occamworm.sources.pin import pin_sources
from occamworm.sources.registry import Registry, RegistryError, Source
from occamworm.sources.remote import refresh_source
from occamworm.sources.verify import format_report, human_size, verify_sources, write_json

REGISTRY_REL = Path("configs/datasets/sources.json")


def find_root(start: Path) -> Path:
    env = os.environ.get("OCCAMWORM_ROOT")
    if env:
        return Path(env)
    # The working directory first, then the checkout this package lives in (python/occamworm/sources/cli.py).
    for p in (start, *start.parents, Path(__file__).resolve().parents[3]):
        if (p / REGISTRY_REL).is_file():
            return p
    raise SystemExit(f"error: cannot find {REGISTRY_REL} above {start}; set OCCAMWORM_ROOT or pass --registry")


def _add_selection(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("selection (default: sources marked default)")
    g.add_argument("--source", "-s", action="append", metavar="ID", help="a source id (repeatable)")
    g.add_argument("--milestone", "-m", metavar="M", help="current sources needed for a milestone, e.g. M0")
    g.add_argument("--all", action="store_true", dest="everything", help="every source, including superseded")


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m occamworm.sources", description=__doc__)
    p.add_argument("--registry", type=Path, help=f"registry file (default: <repo>/{REGISTRY_REL})")
    p.add_argument("--data-dir", type=Path, help="raw data directory (default: <repo>/data/raw)")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("list", help="list registered sources")
    _add_selection(sp)

    sp = sub.add_parser("fetch", help="download and verify assets into the raw data directory")
    _add_selection(sp)
    sp.add_argument("--jobs", "-j", type=int, default=4)
    sp.add_argument("--retries", type=int, default=3)
    sp.add_argument("--dry-run", action="store_true", help="show what would be downloaded and its size")

    sp = sub.add_parser("verify", help="check local raw data against the registry")
    _add_selection(sp)
    sp.add_argument("--quick", action="store_true", help="compare sizes only; skip hashing")
    sp.add_argument("--json", type=Path, metavar="PATH", help="also write a machine-readable report")
    sp.add_argument(
        "--require-license", action="store_true", help="fail if any selected source lacks a declared license"
    )
    sp.add_argument("--verbose", "-v", action="store_true")

    sp = sub.add_parser("pin", help="record sha256 for downloaded assets that have none in the registry")
    _add_selection(sp)

    sp = sub.add_parser("refresh", help="rebuild asset lists and license evidence from upstream APIs")
    _add_selection(sp)
    sp.add_argument(
        "--accept-upstream-changes",
        action="store_true",
        help="apply changed checksums, removed assets, moved git refs and license changes",
    )
    return p


def _select(reg: Registry, args: argparse.Namespace) -> list[Source]:
    try:
        return reg.select(args.source, args.milestone, args.everything)
    except KeyError as e:
        raise SystemExit(f"error: unknown source id {e}") from None


def _cmd_list(reg: Registry, sources: list[Source]) -> int:
    width = max((len(s.id) for s in sources), default=10)
    for s in sources:
        flags = ",".join(f for f in ("default" if s.default else "", s.status if s.status != "current" else "") if f)
        print(
            f"{s.id:<{width}}  {len(s.assets):>4} assets  {human_size(s.total_size):>12}  "
            f"{s.license.label():<14} {'/'.join(s.milestones) or '-':<8} {flags}"
        )
    total = sum(s.total_size or 0 for s in sources)
    print(f"\n{len(sources)} sources, {human_size(total)} with known sizes")
    return 0


def _cmd_fetch(sources: list[Source], data_dir: Path, args: argparse.Namespace) -> int:
    if args.dry_run:
        for s in sources:
            print(f"{s.id}: {len(s.assets)} assets, {human_size(s.total_size)} -> {data_dir / s.id}")
        total = sum(s.total_size or 0 for s in sources)
        unknown = [s.id for s in sources if s.total_size is None]
        print(f"\nTotal: {human_size(total)}" + (f" plus unknown size for {', '.join(unknown)}" if unknown else ""))
        return 0

    def progress(r: FetchResult) -> None:
        if r.outcome != "present":
            print(f"  {r.outcome:<10} {r.source_id}  {r.path}" + (f"  ({r.detail})" if r.detail else ""), flush=True)

    results = fetch_sources(sources, data_dir, jobs=args.jobs, retries=args.retries, progress=progress)
    counts = {k: sum(1 for r in results if r.outcome == k) for k in ("downloaded", "present", "failed")}
    print(f"\n{counts['downloaded']} downloaded, {counts['present']} already present, {counts['failed']} failed")
    return 1 if counts["failed"] else 0


def _cmd_verify(sources: list[Source], data_dir: Path, args: argparse.Namespace) -> int:
    report = verify_sources(sources, data_dir, quick=args.quick)
    print(format_report(report, verbose=args.verbose))
    if args.json:
        write_json(report, args.json)
    if not report.integrity_ok:
        return 1
    if args.require_license and report.license_issues():
        print("Failing because --require-license was given and some sources lack a usable license.")
        return 1
    return 0


def _cmd_pin(reg: Registry, sources: list[Source], data_dir: Path, path: Path) -> int:
    results = pin_sources(reg, sources, data_dir)
    for r in results:
        if r.outcome in ("pinned", "refused"):
            print(f"  {r.outcome:<8} {r.source_id}  {r.path}  {r.detail}")
    if any(r.outcome == "pinned" for r in results):
        reg_mod.save(reg, path)
        print(f"Updated {path}")
    missing = sum(1 for r in results if r.outcome == "missing")
    if missing:
        print(f"{missing} unpinned assets are not downloaded yet; run `fetch` first.")
    return 1 if any(r.outcome == "refused" for r in results) else 0


def _cmd_refresh(reg: Registry, sources: list[Source], path: Path, accept: bool) -> int:
    pending = False
    for s in sources:
        print(f"{s.id}: querying {s.provider} ...", flush=True)
        r = refresh_source(s, accept=accept)
        reg.sources[reg.sources.index(s)] = r.source
        if r.added:
            print(f"  + {len(r.added)} new assets")
        for c in r.changed:
            print(f"  ! changed upstream: {c}")
        for p in r.removed:
            print(f"  - no longer listed upstream: {p}")
        if r.license_changed:
            print(f"  ! license changed: {r.license_changed}")
        if r.needs_acceptance and not accept:
            pending = True
    reg_mod.save(reg, path)
    print(f"Updated {path}")
    if pending:
        print("Upstream changes were reported but NOT applied. Review them, then rerun with --accept-upstream-changes.")
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    path: Path = args.registry or find_root(Path.cwd()) / REGISTRY_REL
    root = path.resolve().parent.parent.parent
    data_dir: Path = args.data_dir or root / "data" / "raw"
    try:
        reg = reg_mod.load(path)
    except (OSError, RegistryError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    sources = _select(reg, args)
    if args.cmd == "list":
        return _cmd_list(reg, sources)
    if args.cmd == "fetch":
        return _cmd_fetch(sources, data_dir, args)
    if args.cmd == "verify":
        return _cmd_verify(sources, data_dir, args)
    if args.cmd == "pin":
        return _cmd_pin(reg, sources, data_dir, path)
    if args.cmd == "refresh":
        return _cmd_refresh(reg, sources, path, args.accept_upstream_changes)
    raise AssertionError(args.cmd)
