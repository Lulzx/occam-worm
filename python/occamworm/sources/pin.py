"""Record sha256 and size for downloaded assets whose upstream gives only md5 or no checksum at all.

Pinning is trust on first use, so it is an explicit step: it never overwrites an existing sha256, and it
refuses to pin a file that contradicts a known md5 or size.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

from occamworm.sources.fetch import asset_path, check, digest_file
from occamworm.sources.registry import Registry, Source


@dataclass
class PinResult:
    source_id: str
    path: str
    outcome: str  # pinned | already | missing | refused
    detail: str = ""


def pin_sources(registry: Registry, sources: list[Source], data_dir: Path) -> list[PinResult]:
    results: list[PinResult] = []
    for s in sources:
        new_assets = []
        for a in s.assets:
            if a.sha256 is not None:
                results.append(PinResult(s.id, a.path, "already"))
                new_assets.append(a)
                continue
            p = asset_path(data_dir, s, a)
            if not p.is_file():
                results.append(PinResult(s.id, a.path, "missing"))
                new_assets.append(a)
                continue
            d = digest_file(p)
            problem = check(a, d)
            if problem is not None:
                results.append(PinResult(s.id, a.path, "refused", problem))
                new_assets.append(a)
                continue
            results.append(PinResult(s.id, a.path, "pinned", d.sha256))
            new_assets.append(replace(a, sha256=d.sha256, size=d.size))
        idx = registry.sources.index(s)
        registry.sources[idx] = replace(s, assets=new_assets)
    return results
