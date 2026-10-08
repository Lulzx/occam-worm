"""Download registered assets into ``data/raw/<source>/`` and check them against the registry.

Downloads are written to a ``.part`` file, hashed while streaming, checked against the expected size and
checksum, and only then moved into place. Existing files are never modified (spec §2.2).
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from occamworm import __version__
from occamworm.sources.registry import RETRIEVAL_FILE, Asset, Source

USER_AGENT = f"occamworm-sources/{__version__} (+https://github.com/Lulzx/occam-worm)"
CHUNK = 1 << 20


@dataclass(frozen=True)
class Digest:
    size: int
    sha256: str
    md5: str


class ChecksumMismatch(Exception):
    pass


class DownloadError(Exception):
    pass


def now_utc() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def digest_file(path: Path) -> Digest:
    sha, md5, n = hashlib.sha256(), hashlib.md5(usedforsecurity=False), 0
    with path.open("rb") as f:
        while chunk := f.read(CHUNK):
            sha.update(chunk)
            md5.update(chunk)
            n += len(chunk)
    return Digest(n, sha.hexdigest(), md5.hexdigest())


def check(asset: Asset, d: Digest) -> str | None:
    """Return a description of the first mismatch against the registry, or None if consistent.
    sha256 is authoritative; md5 is used only when the upstream provides nothing stronger."""
    if asset.size is not None and d.size != asset.size:
        return f"size {d.size} != expected {asset.size}"
    if asset.sha256 is not None:
        return None if d.sha256 == asset.sha256 else f"sha256 {d.sha256} != expected {asset.sha256}"
    if asset.md5 is not None:
        return None if d.md5 == asset.md5 else f"md5 {d.md5} != expected {asset.md5}"
    return None


def asset_path(data_dir: Path, source: Source, asset: Asset) -> Path:
    return data_dir / source.id / Path(*asset.path.split("/"))


def _stream(url: str, dest: Path, timeout: float) -> Digest:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    sha, md5, n = hashlib.sha256(), hashlib.md5(usedforsecurity=False), 0
    with urllib.request.urlopen(req, timeout=timeout) as r, dest.open("wb") as f:
        while chunk := r.read(CHUNK):
            f.write(chunk)
            sha.update(chunk)
            md5.update(chunk)
            n += len(chunk)
    return Digest(n, sha.hexdigest(), md5.hexdigest())


def download(url: str, dest: Path, *, retries: int = 3, timeout: float = 60.0, backoff: float = 2.0) -> Digest:
    """Download ``url`` to ``dest.part`` and return its digest. The caller decides whether to keep it."""
    part = dest.with_name(dest.name + ".part")
    part.parent.mkdir(parents=True, exist_ok=True)
    last: Exception | None = None
    for attempt in range(retries):
        try:
            return _stream(url, part, timeout)
        except urllib.error.HTTPError as e:
            last = e
            if 400 <= e.code < 500 and e.code != 429:
                break  # not transient
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            last = e
        if attempt < retries - 1:
            time.sleep(backoff * 2**attempt)
    part.unlink(missing_ok=True)
    raise DownloadError(f"{url}: {last}")


@dataclass
class FetchResult:
    source_id: str
    path: str
    outcome: str  # downloaded | present | failed
    detail: str = ""


def _record(data_dir: Path, source: Source, entries: dict[str, dict[str, object]]) -> None:
    if not entries:
        return
    f = data_dir / source.id / RETRIEVAL_FILE
    f.parent.mkdir(parents=True, exist_ok=True)
    existing: dict[str, object] = json.loads(f.read_text()) if f.exists() else {}
    files = existing.get("files", {})
    assert isinstance(files, dict)
    files.update(entries)
    doc = {
        "source_id": source.id,
        "version": source.version,
        "note": "Local retrieval record written by occamworm.sources; not committed.",
        "files": dict(sorted(files.items())),
    }
    f.write_text(json.dumps(doc, indent=2) + "\n")


def fetch_asset(
    source: Source, asset: Asset, data_dir: Path, *, force: bool = False, retries: int = 3
) -> tuple[FetchResult, dict[str, object] | None]:
    dest = asset_path(data_dir, source, asset)
    if dest.exists() and not force:
        problem = check(asset, digest_file(dest))
        if problem is None:
            return FetchResult(source.id, asset.path, "present"), None
        return FetchResult(
            source.id,
            asset.path,
            "failed",
            f"existing file does not match registry: {problem}; "
            "raw files are never overwritten automatically, inspect and remove it",
        ), None
    try:
        d = download(asset.url, dest, retries=retries)
    except DownloadError as e:
        return FetchResult(source.id, asset.path, "failed", str(e)), None
    part = dest.with_name(dest.name + ".part")
    problem = check(asset, d)
    if problem is not None:
        part.unlink(missing_ok=True)
        return FetchResult(source.id, asset.path, "failed", f"checksum mismatch: {problem}"), None
    os.replace(part, dest)
    record: dict[str, object] = {
        "url": asset.url,
        "retrieved_at": now_utc(),
        "size": d.size,
        "sha256": d.sha256,
        "md5": d.md5,
        "tool": USER_AGENT,
    }
    detail = "" if asset.pinned else "no expected checksum in registry; run `pin` to record it"
    return FetchResult(source.id, asset.path, "downloaded", detail), record


def fetch_sources(
    sources: list[Source],
    data_dir: Path,
    *,
    jobs: int = 4,
    force: bool = False,
    retries: int = 3,
    progress: Callable[[FetchResult], None] | None = None,
) -> list[FetchResult]:
    results: list[FetchResult] = []
    records: dict[str, dict[str, dict[str, object]]] = {s.id: {} for s in sources}
    lock = threading.Lock()

    def work(source: Source, asset: Asset) -> None:
        res, rec = fetch_asset(source, asset, data_dir, force=force, retries=retries)
        with lock:
            results.append(res)
            if rec is not None:
                records[source.id][asset.path] = rec
            if progress:
                progress(res)

    with ThreadPoolExecutor(max_workers=max(1, jobs)) as pool:
        futures = [pool.submit(work, s, a) for s in sources for a in s.assets]
        for fut in futures:
            fut.result()  # surface unexpected errors instead of dropping them
    for s in sources:
        _record(data_dir, s, records[s.id])
    order = {(s.id, a.path): i for i, (s, a) in enumerate((s, a) for s in sources for a in s.assets)}
    return sorted(results, key=lambda r: order[(r.source_id, r.path)])
