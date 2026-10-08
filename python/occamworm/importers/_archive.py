"""Read selected members of a source tarball into memory without extracting anything to disk.

GitHub tarballs have one top-level directory (``<repo>-<commit>/``); member keys are relative to it.
"""

from __future__ import annotations

import tarfile
from collections.abc import Iterable
from pathlib import Path, PurePosixPath


class ArchiveError(Exception):
    """A required member is missing, or the archive has an unexpected layout."""


def read_members(archive: Path, wanted: Iterable[str]) -> dict[str, bytes]:
    """Return ``{relative path: bytes}`` for every wanted member; a missing member is an error."""
    want = set(wanted)
    found: dict[str, bytes] = {}
    with tarfile.open(archive, "r|gz") as tar:
        for info in tar:
            if not info.isfile():
                continue
            parts = PurePosixPath(info.name).parts
            if len(parts) < 2 or ".." in parts:
                continue
            rel = PurePosixPath(*parts[1:]).as_posix()
            if rel in want and rel not in found:
                f = tar.extractfile(info)
                if f is None:
                    raise ArchiveError(f"{archive.name}: cannot read {rel}")
                found[rel] = f.read()
    missing = sorted(want - set(found))
    if missing:
        raise ArchiveError(f"{archive.name}: missing members {missing}")
    return found
