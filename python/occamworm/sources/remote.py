"""Rebuild asset lists and license evidence from upstream APIs (OSF, Zenodo, GitHub).

Refreshing never silently changes a pinned checksum: if upstream bytes, a git ref or a license changed,
the difference is reported and only applied with ``accept=True``.
"""

from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Any

from occamworm.sources.fetch import USER_AGENT, now_utc
from occamworm.sources.registry import Asset, License, Source

JsonGetter = Callable[[str], Any]


def get_json(url: str) -> Any:
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    token = os.environ.get("GITHUB_TOKEN")
    if token and urllib.parse.urlparse(url).hostname == "api.github.com":
        headers["Authorization"] = f"Bearer {token}"
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=60) as r:
        return json.load(r)


@dataclass
class Upstream:
    assets: list[Asset]
    license: License
    version: str | None = None
    remote: dict[str, Any] | None = None  # updated remote spec, e.g. a moved git ref


def _included(path: str, include: list[str]) -> bool:
    return any(path == p or (p.endswith("/") and path.startswith(p)) for p in include)


# --------------------------------------------------------------------------- OSF


def _osf(remote: dict[str, Any], get: JsonGetter, today: str) -> Upstream:
    node = remote["node"]
    include = [p.lstrip("/") for p in remote["include"]]
    files: dict[str, Asset] = {}

    def walk(url: str) -> None:
        # OSF pagination repeats entries unless sorted; dedupe by path regardless.
        sep = "&" if "?" in url else "?"
        next_url: str | None = f"{url}{sep}sort=name&page%5Bsize%5D=100"
        while next_url:
            page = get(next_url)
            for item in page["data"]:
                a = item["attributes"]
                if a["kind"] == "folder":
                    folder = a["materialized_path"].lstrip("/")
                    if any(p.startswith(folder) or folder.startswith(p) for p in include):
                        walk(item["relationships"]["files"]["links"]["related"]["href"])
                    continue
                path = a["materialized_path"].lstrip("/")
                if not _included(path, include):
                    continue
                hashes = (a.get("extra") or {}).get("hashes") or {}
                files[path] = Asset(
                    path=path,
                    url=item["links"]["download"],
                    size=a["size"],
                    sha256=hashes.get("sha256"),
                    md5=hashes.get("md5"),
                    upstream_modified=a.get("date_modified"),
                )
            next_url = page["links"].get("next")

    walk(f"https://api.osf.io/v2/nodes/{node}/files/osfstorage/")
    attrs = get(f"https://api.osf.io/v2/nodes/{node}/")["data"]["attributes"]
    if attrs.get("node_license"):
        lic_data = get(f"https://api.osf.io/v2/nodes/{node}/license/")["data"]["attributes"]
        name = lic_data.get("name") or "unknown"
        spdx = _OSF_LICENSES.get(name)
        lic = License(spdx, "declared" if spdx else "unknown", f"OSF node license: {name}", today)
    else:
        lic = License(None, "none_declared", f"OSF node {node} has no license (node_license is null)", today)
    return Upstream(sorted(files.values(), key=lambda x: x.path), lic)


_OSF_LICENSES = {
    "CC-By Attribution 4.0 International": "CC-BY-4.0",
    "CC0 1.0 Universal": "CC0-1.0",
    "MIT License": "MIT",
    "Apache License 2.0": "Apache-2.0",
    'BSD 3-Clause "New"/"Revised" License': "BSD-3-Clause",
    "GNU General Public License (GPL) 3.0": "GPL-3.0-only",
}


# --------------------------------------------------------------------------- Zenodo


def _zenodo(remote: dict[str, Any], get: JsonGetter, today: str) -> Upstream:
    rec = get(f"https://zenodo.org/api/records/{remote['record']}")
    include = set(remote["include"])
    assets = []
    for f in rec.get("files", []):
        if f["key"] not in include:
            continue
        algo, _, value = f["checksum"].partition(":")
        assets.append(
            Asset(path=f["key"], url=f["links"]["self"], size=f["size"], md5=value if algo == "md5" else None)
        )
    lic_id = (rec["metadata"].get("license") or {}).get("id")
    if lic_id:
        spdx = _ZENODO_LICENSES.get(lic_id.lower(), lic_id)
        lic = License(spdx, "declared", f"Zenodo record {rec['id']} license: {lic_id}", today)
    else:
        lic = License(None, "none_declared", f"Zenodo record {rec['id']} declares no license", today)
    return Upstream(sorted(assets, key=lambda x: x.path), lic, version=f"Zenodo record {rec['id']}")


_ZENODO_LICENSES = {"cc-by-4.0": "CC-BY-4.0", "cc0-1.0": "CC0-1.0", "mit": "MIT", "apache-2.0": "Apache-2.0"}


# --------------------------------------------------------------------------- GitHub


def github_asset(repo: str, commit: str) -> Asset:
    name = repo.split("/")[1]
    return Asset(path=f"{name}-{commit[:12]}.tar.gz", url=f"https://codeload.github.com/{repo}/tar.gz/{commit}")


def _github(remote: dict[str, Any], get: JsonGetter, today: str) -> Upstream:
    repo, ref = remote["repo"], remote["ref"]
    info = get(f"https://api.github.com/repos/{repo}")
    commit = get(f"https://api.github.com/repos/{repo}/commits/{urllib.parse.quote(ref, safe='')}")["sha"]
    spdx = (info.get("license") or {}).get("spdx_id")
    if spdx and spdx != "NOASSERTION":
        lic = License(spdx, "declared", f"GitHub license detection for {repo}: {spdx}", today)
    elif spdx == "NOASSERTION":
        lic = License(None, "unknown", f"GitHub found a license file in {repo} but could not identify it", today)
    else:
        lic = License(None, "none_declared", f"GitHub repository {repo} has no license file", today)
    new_remote = {**remote, "commit": commit}
    return Upstream([github_asset(repo, commit)], lic, version=f"{ref} @ {commit}", remote=new_remote)


PROVIDERS: dict[str, Callable[[dict[str, Any], JsonGetter, str], Upstream]] = {
    "osf": _osf,
    "zenodo": _zenodo,
    "github": _github,
}


# --------------------------------------------------------------------------- merge


@dataclass
class RefreshResult:
    source: Source  # the source as it should be written
    added: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    changed: list[str] = field(default_factory=list)  # pinned checksum or git commit differs upstream
    license_changed: str | None = None

    @property
    def needs_acceptance(self) -> bool:
        return bool(self.removed or self.changed or self.license_changed)


def _conflicts(old: Asset, new: Asset) -> bool:
    return bool(
        (old.sha256 and new.sha256 and old.sha256 != new.sha256)
        or (old.md5 and new.md5 and old.md5 != new.md5)
        or (old.size is not None and new.size is not None and old.size != new.size)
    )


def refresh_source(source: Source, *, get: JsonGetter = get_json, accept: bool = False) -> RefreshResult:
    """Compare ``source`` with upstream. Additions are always applied. Changed checksums, removed assets,
    a moved git ref and license changes are reported and applied only when ``accept`` is true."""
    up = PROVIDERS[source.provider](source.remote, get, now_utc()[:10])
    result = RefreshResult(source)
    old = {a.path: a for a in source.assets}

    pinned_commit = source.remote.get("commit")
    ref_moved = up.remote is not None and pinned_commit is not None and up.remote["commit"] != pinned_commit
    if ref_moved:
        assert up.remote is not None
        result.changed.append(f"git ref {source.remote['ref']} moved: {pinned_commit} -> {up.remote['commit']}")
        assets = up.assets if accept else source.assets
    else:
        merged: dict[str, Asset] = {}
        for a in up.assets:
            prev = old.get(a.path)
            if prev is None:
                result.added.append(a.path)
                merged[a.path] = a
            elif _conflicts(prev, a):
                result.changed.append(a.path)
                merged[a.path] = a if accept else prev
            else:
                # Keep values pinned locally that the upstream cannot provide (sha256 for md5-only hosts).
                merged[a.path] = replace(
                    a, sha256=a.sha256 or prev.sha256, size=a.size if a.size is not None else prev.size
                )
        result.removed = sorted(set(old) - set(merged))
        if not accept:
            merged.update({p: old[p] for p in result.removed})
        assets = sorted(merged.values(), key=lambda x: x.path)

    first_check = source.license.checked_at is None
    if not first_check and (source.license.spdx, source.license.status) != (up.license.spdx, up.license.status):
        result.license_changed = f"{source.license.label()} -> {up.license.label()}"
    take_license = accept or result.license_changed is None
    take_remote = up.remote is not None and (accept or not ref_moved)

    result.source = replace(
        source,
        assets=assets,
        license=up.license if take_license else source.license,
        version=(up.version or source.version) if (accept or not ref_moved) else source.version,
        remote=up.remote if take_remote and up.remote is not None else source.remote,
        metadata_retrieved_at=now_utc(),
    )
    return result
