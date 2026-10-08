from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pytest
from conftest import sha256, source_dict

from occamworm.sources import registry as reg_mod
from occamworm.sources.registry import Registry, RegistryError

REPO = Path(__file__).resolve().parents[3]
REGISTRY = REPO / "configs" / "datasets" / "sources.json"


def test_repository_registry_is_valid() -> None:
    reg = reg_mod.load(REGISTRY)
    assert reg.sources, "registry is empty"
    for s in reg.sources:
        assert s.assets, f"{s.id} has no assets; run refresh"
        assert s.metadata_retrieved_at, f"{s.id} was never refreshed"
        assert s.license.checked_at, f"{s.id} license was never checked"
        if s.provider == "osf":
            assert all(a.sha256 and a.size is not None for a in s.assets), f"{s.id}: OSF provides sha256 and size"
        if s.provider == "github":
            assert len(s.remote.get("commit", "")) == 40, f"{s.id}: git sources must pin a full commit"


def test_repository_registry_is_in_canonical_format() -> None:
    text = REGISTRY.read_text()
    assert reg_mod.dumps(reg_mod.from_dict(json.loads(text))) == text, "reformat with registry.save()"


def test_m0_is_covered_by_default_set() -> None:
    reg = reg_mod.load(REGISTRY)
    assert {s.id for s in reg.select(milestone="M0")} <= {s.id for s in reg.select()}


def test_roundtrip(make_registry: Callable[..., Registry]) -> None:
    reg = make_registry(source_dict(assets=[{"path": "a/b.txt", "url": "u", "size": 3, "sha256": sha256(b"abc")}]))
    again = reg_mod.from_dict(json.loads(reg_mod.dumps(reg)))
    assert again == reg


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda d: d["assets"][0].update(sha256="ABC"), "sha256"),
        (lambda d: d["assets"][0].update(path="../x"), "relative"),
        (lambda d: d["assets"][0].update(path="_retrieval.json"), "relative"),
        (lambda d: d.update(kind="paper"), "kind"),
        (lambda d: d.update(id="Bad_Id"), "id must match"),
        (lambda d: d["license"].update(spdx=None), "SPDX"),
        (lambda d: d["remote"].update(provider="zenodo"), "remote.provider"),
        (lambda d: d["assets"].append(dict(d["assets"][0])), "duplicate asset"),
    ],
)
def test_validation_errors(mutate: Callable[[dict[str, object]], None], message: str) -> None:
    d = source_dict(assets=[{"path": "a.txt", "url": "u"}])
    mutate(d)
    with pytest.raises(RegistryError, match=message):
        reg_mod.from_dict({"schema_version": "0.1.0", "sources": [d]})


def test_duplicate_source_ids(make_registry: Callable[..., Registry]) -> None:
    with pytest.raises(RegistryError, match="duplicate source ids"):
        make_registry(source_dict("a"), source_dict("a"))


def test_selection(make_registry: Callable[..., Registry]) -> None:
    reg = make_registry(
        source_dict("core", milestones=["M0"], default=True),
        source_dict("later", milestones=["M6"], default=False),
        source_dict("old", milestones=["M0"], default=True, status="superseded"),
    )
    assert [s.id for s in reg.select()] == ["core"]
    assert [s.id for s in reg.select(milestone="M6")] == ["later"]
    assert [s.id for s in reg.select(everything=True)] == ["core", "later", "old"]
    assert [s.id for s in reg.select(ids=["old"])] == ["old"]
    with pytest.raises(KeyError):
        reg.select(ids=["nope"])


@pytest.mark.parametrize(
    "spdx, status, category",
    [
        ("MIT", "declared", "permissive"),
        ("GPL-3.0", "declared", "copyleft"),
        ("Custom-1", "declared", "other"),
        (None, "none_declared", "unlicensed"),
        (None, "unknown", "unknown"),
    ],
)
def test_license_category(spdx: str | None, status: str, category: str) -> None:
    assert reg_mod.License(spdx, status, "e").category == category
