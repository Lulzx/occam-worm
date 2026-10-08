"""The source registry: every upstream dataset and code repository the project uses, pinned by checksum.

The registry lives in ``configs/datasets/sources.json``. It is the only place where upstream locations,
expected checksums and license evidence are recorded (spec §2.2, ticket OW-001).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

SCHEMA_VERSION = "0.1.0"

_ID = re.compile(r"^[a-z0-9][a-z0-9-]*$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_MD5 = re.compile(r"^[0-9a-f]{32}$")

KINDS = ("dataset", "code")
STATUSES = ("current", "superseded")
LICENSE_STATUSES = ("declared", "none_declared", "unknown")
PERMISSIVE = frozenset(
    {"MIT", "Apache-2.0", "BSD-2-Clause", "BSD-3-Clause", "ISC", "CC-BY-4.0", "CC0-1.0", "Unlicense"}
)
COPYLEFT_PREFIXES = ("GPL-", "LGPL-", "AGPL-", "MPL-", "CC-BY-SA-", "EUPL-")

# Local bookkeeping file written next to downloaded assets; asset paths may not collide with it.
RETRIEVAL_FILE = "_retrieval.json"


class RegistryError(ValueError):
    """The registry file is malformed or violates an invariant."""


@dataclass
class License:
    spdx: str | None
    status: str
    evidence: str
    checked_at: str | None = None

    @property
    def category(self) -> str:
        """One of: permissive, copyleft, other, unlicensed, unknown."""
        if self.status == "none_declared":
            return "unlicensed"
        if self.status != "declared" or not self.spdx:
            return "unknown"
        if self.spdx in PERMISSIVE:
            return "permissive"
        if self.spdx.startswith(COPYLEFT_PREFIXES):
            return "copyleft"
        return "other"

    def label(self) -> str:
        if self.status == "none_declared":
            return "NONE DECLARED"
        if self.status == "unknown":
            return "UNKNOWN"
        return self.spdx or "UNKNOWN"


@dataclass
class Asset:
    path: str
    url: str
    size: int | None = None
    sha256: str | None = None
    md5: str | None = None
    upstream_modified: str | None = None

    @property
    def pinned(self) -> bool:
        return self.sha256 is not None or self.md5 is not None


@dataclass
class Source:
    id: str
    title: str
    kind: str
    provider: str
    url: str
    license: License
    remote: dict[str, Any]
    doi: str | None = None
    citation: str | None = None
    version: str | None = None
    milestones: list[str] = field(default_factory=list)
    default: bool = False
    status: str = "current"
    superseded_by: str | None = None
    notes: str | None = None
    metadata_retrieved_at: str | None = None
    assets: list[Asset] = field(default_factory=list)

    @property
    def total_size(self) -> int | None:
        sizes = [a.size for a in self.assets]
        if any(s is None for s in sizes):
            return None
        return sum(s for s in sizes if s is not None)


@dataclass
class Registry:
    sources: list[Source]
    schema_version: str = SCHEMA_VERSION

    def get(self, source_id: str) -> Source:
        for s in self.sources:
            if s.id == source_id:
                return s
        raise KeyError(source_id)

    def select(
        self,
        ids: list[str] | None = None,
        milestone: str | None = None,
        everything: bool = False,
    ) -> list[Source]:
        """Explicit ids win; otherwise --all, then a milestone, then the default set. Superseded sources
        are only included when named explicitly or with ``everything``."""
        if ids:
            return [self.get(i) for i in ids]
        if everything:
            return list(self.sources)
        current = [s for s in self.sources if s.status == "current"]
        if milestone:
            return [s for s in current if milestone in s.milestones]
        return [s for s in current if s.default]


# --------------------------------------------------------------------------- parsing


def _req(d: dict[str, Any], key: str, where: str) -> Any:
    if key not in d:
        raise RegistryError(f"{where}: missing required field '{key}'")
    return d[key]


def _asset_from(d: dict[str, Any], where: str) -> Asset:
    a = Asset(
        path=_req(d, "path", where),
        url=_req(d, "url", where),
        size=d.get("size"),
        sha256=d.get("sha256"),
        md5=d.get("md5"),
        upstream_modified=d.get("upstream_modified"),
    )
    p = PurePosixPath(a.path)
    if p.is_absolute() or ".." in p.parts or not a.path or a.path.startswith("_"):
        raise RegistryError(f"{where}: asset path must be relative, without '..', not starting with '_'")
    if a.sha256 is not None and not _SHA256.match(a.sha256):
        raise RegistryError(f"{where}: sha256 must be 64 lowercase hex characters")
    if a.md5 is not None and not _MD5.match(a.md5):
        raise RegistryError(f"{where}: md5 must be 32 lowercase hex characters")
    if a.size is not None and (not isinstance(a.size, int) or a.size < 0):
        raise RegistryError(f"{where}: size must be a non-negative integer")
    return a


def _source_from(d: dict[str, Any]) -> Source:
    sid = _req(d, "id", "source")
    where = f"source '{sid}'"
    lic = _req(d, "license", where)
    license_ = License(
        spdx=lic.get("spdx"),
        status=_req(lic, "status", f"{where} license"),
        evidence=_req(lic, "evidence", f"{where} license"),
        checked_at=lic.get("checked_at"),
    )
    s = Source(
        id=sid,
        title=_req(d, "title", where),
        kind=_req(d, "kind", where),
        provider=_req(d, "provider", where),
        url=_req(d, "url", where),
        license=license_,
        remote=_req(d, "remote", where),
        doi=d.get("doi"),
        citation=d.get("citation"),
        version=d.get("version"),
        milestones=list(d.get("milestones", [])),
        default=bool(d.get("default", False)),
        status=d.get("status", "current"),
        superseded_by=d.get("superseded_by"),
        notes=d.get("notes"),
        metadata_retrieved_at=d.get("metadata_retrieved_at"),
        assets=[_asset_from(a, f"{where} asset {i}") for i, a in enumerate(d.get("assets", []))],
    )
    if not _ID.match(s.id):
        raise RegistryError(f"{where}: id must match {_ID.pattern}")
    if s.kind not in KINDS:
        raise RegistryError(f"{where}: kind must be one of {KINDS}")
    if s.status not in STATUSES:
        raise RegistryError(f"{where}: status must be one of {STATUSES}")
    if license_.status not in LICENSE_STATUSES:
        raise RegistryError(f"{where}: license status must be one of {LICENSE_STATUSES}")
    if license_.status == "declared" and not license_.spdx:
        raise RegistryError(f"{where}: a declared license needs an SPDX identifier")
    if s.remote.get("provider") != s.provider:
        raise RegistryError(f"{where}: remote.provider must equal provider")
    paths = [a.path for a in s.assets]
    if len(paths) != len(set(paths)):
        raise RegistryError(f"{where}: duplicate asset paths")
    return s


def from_dict(d: dict[str, Any]) -> Registry:
    version = d.get("schema_version")
    if version != SCHEMA_VERSION:
        raise RegistryError(f"unsupported schema_version {version!r}; expected {SCHEMA_VERSION}")
    reg = Registry(sources=[_source_from(s) for s in d.get("sources", [])], schema_version=version)
    ids = [s.id for s in reg.sources]
    dupes = sorted({i for i in ids if ids.count(i) > 1})
    if dupes:
        raise RegistryError(f"duplicate source ids: {', '.join(dupes)}")
    for s in reg.sources:
        if s.superseded_by is not None and s.superseded_by not in ids:
            raise RegistryError(f"source '{s.id}': superseded_by '{s.superseded_by}' is not a known source")
    return reg


def load(path: Path) -> Registry:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise RegistryError(f"{path}: invalid JSON: {e}") from e
    return from_dict(data)


# --------------------------------------------------------------------------- serialization


def _license_dict(lic: License) -> dict[str, Any]:
    return {"spdx": lic.spdx, "status": lic.status, "evidence": lic.evidence, "checked_at": lic.checked_at}


def _asset_dict(a: Asset) -> dict[str, Any]:
    d: dict[str, Any] = {"path": a.path, "url": a.url, "size": a.size, "sha256": a.sha256}
    if a.md5 is not None:
        d["md5"] = a.md5
    if a.upstream_modified is not None:
        d["upstream_modified"] = a.upstream_modified
    return d


def _source_dict(s: Source) -> dict[str, Any]:
    return {
        "id": s.id,
        "title": s.title,
        "kind": s.kind,
        "provider": s.provider,
        "url": s.url,
        "doi": s.doi,
        "citation": s.citation,
        "version": s.version,
        "license": _license_dict(s.license),
        "milestones": s.milestones,
        "default": s.default,
        "status": s.status,
        "superseded_by": s.superseded_by,
        "notes": s.notes,
        "remote": s.remote,
        "metadata_retrieved_at": s.metadata_retrieved_at,
    }


def dumps(reg: Registry) -> str:
    """Stable, diff-friendly JSON: sources indented, one asset per line."""
    out = ["{", f'  "schema_version": {json.dumps(reg.schema_version)},', '  "sources": [']
    for i, s in enumerate(reg.sources):
        body = json.dumps(_source_dict(s), indent=2, ensure_ascii=False).split("\n")
        body = ["    " + line for line in body[:-1]]  # drop closing brace; re-added below
        body[-1] += ","
        if s.assets:
            body.append('      "assets": [')
            rows = [
                "        " + json.dumps(_asset_dict(a), ensure_ascii=False, separators=(", ", ": ")) for a in s.assets
            ]
            body.append(",\n".join(rows))
            body.append("      ]")
        else:
            body.append('      "assets": []')
        body.append("    }" + ("," if i < len(reg.sources) - 1 else ""))
        out.extend(body)
    out += ["  ]", "}"]
    return "\n".join(out) + "\n"


def save(reg: Registry, path: Path) -> None:
    text = dumps(reg)
    from_dict(json.loads(text))  # never write something we cannot read back
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)
