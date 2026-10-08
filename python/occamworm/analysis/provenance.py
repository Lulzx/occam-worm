"""Artifact identity and immutability (§14.7, OW-015).

A run directory gets ``run.json`` when it starts: the hashes of everything the result depends on, and a
content-addressed ``run_id`` derived from them. When the run is complete, ``freeze`` writes ``frozen.json`` with
the sha256 of every file; reports are built only from frozen runs whose files still match (``verify_frozen``), so
a report can never silently recompute against changed data.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

FROZEN = "frozen.json"
RUN = "run.json"


class ProvenanceError(RuntimeError):
    """Raised when an artifact is not frozen, was modified, or belongs to a different run."""


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_revision(root: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True, check=True).stdout

    try:
        rev = run("rev-parse", "HEAD").strip()
        dirty = bool(run("status", "--porcelain", "--untracked-files=no", "python", "configs", "libs").strip())
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}
    return {"commit": rev, "dirty": dirty}


def run_id(inputs: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()


def write_run_manifest(out: Path, inputs: dict[str, Any]) -> dict[str, Any]:
    """Create ``run.json``; if it exists, its inputs must be identical (a resumed run), else refuse."""
    body = {"run_id": run_id(inputs), "inputs": inputs}
    path = out / RUN
    if path.exists():
        old = json.loads(path.read_text())
        if old["run_id"] != body["run_id"]:
            raise ProvenanceError(f"{path} belongs to run {old['run_id'][:12]}; inputs changed — use a new directory")
        return dict(old)
    out.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n")
    return body


def freeze(out: Path) -> dict[str, Any]:
    if (out / FROZEN).exists():
        raise ProvenanceError(f"{out} is already frozen")
    files = {
        str(p.relative_to(out)): file_sha256(p) for p in sorted(out.rglob("*")) if p.is_file() and p.name != FROZEN
    }
    run = json.loads((out / RUN).read_text()) if (out / RUN).exists() else None
    body = {"run_id": run["run_id"] if run else None, "files": files}
    (out / FROZEN).write_text(json.dumps(body, indent=2, sort_keys=True) + "\n")
    return body


def verify_frozen(out: Path) -> dict[str, Any]:
    path = out / FROZEN
    if not path.exists():
        raise ProvenanceError(f"{out} is not frozen; reports are built from frozen runs only")
    body = json.loads(path.read_text())
    for rel, sha in body["files"].items():
        p = out / rel
        if not p.exists() or file_sha256(p) != sha:
            raise ProvenanceError(f"{p} changed after the run was frozen")
    return dict(body)
