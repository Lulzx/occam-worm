from __future__ import annotations

import hashlib
import http.server
import threading
from collections.abc import Callable, Iterator
from functools import partial
from pathlib import Path
from typing import Any

import pytest

from occamworm.sources.registry import Registry, from_dict


@pytest.fixture
def http_root(tmp_path: Path) -> Iterator[tuple[Path, str]]:
    """Serve a temporary directory over HTTP on localhost; yields (directory, base_url)."""
    root = tmp_path / "upstream"
    root.mkdir()

    class Quiet(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *args: Any) -> None:
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), partial(Quiet, directory=str(root)))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield root, f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()


def sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def md5(b: bytes) -> str:
    return hashlib.md5(b).hexdigest()


def source_dict(sid: str = "demo", *, assets: list[dict[str, Any]] | None = None, **kw: Any) -> dict[str, Any]:
    d: dict[str, Any] = {
        "id": sid,
        "title": "Demo source",
        "kind": "dataset",
        "provider": "osf",
        "url": "https://example.org/",
        "license": {"spdx": "CC-BY-4.0", "status": "declared", "evidence": "test", "checked_at": None},
        "remote": {"provider": "osf", "node": "x", "include": ["/"]},
        "milestones": ["M0"],
        "default": True,
        "assets": assets or [],
    }
    d.update(kw)
    return d


@pytest.fixture
def make_registry() -> Callable[..., Registry]:
    def make(*sources: dict[str, Any]) -> Registry:
        return from_dict({"schema_version": "0.1.0", "sources": list(sources)})

    return make
