from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pytest
from conftest import md5, sha256, source_dict

from occamworm.sources import registry as reg_mod
from occamworm.sources.cli import main
from occamworm.sources.fetch import fetch_sources
from occamworm.sources.pin import pin_sources
from occamworm.sources.registry import Registry
from occamworm.sources.verify import format_report, verify_sources

HTTP = tuple[Path, str]


def publish(root: Path, name: str, data: bytes) -> None:
    (root / name).parent.mkdir(parents=True, exist_ok=True)
    (root / name).write_bytes(data)


def test_fetch_then_verify_ok(http_root: HTTP, tmp_path: Path, make_registry: Callable[..., Registry]) -> None:
    root, base = http_root
    publish(root, "a.txt", b"alpha")
    publish(root, "dir/b.txt", b"beta")
    reg = make_registry(
        source_dict(
            assets=[
                {"path": "a.txt", "url": f"{base}/a.txt", "size": 5, "sha256": sha256(b"alpha")},
                {"path": "dir/b.txt", "url": f"{base}/dir/b.txt", "size": 4, "sha256": sha256(b"beta")},
            ]
        )
    )
    data = tmp_path / "raw"

    results = fetch_sources(reg.sources, data)
    assert [r.outcome for r in results] == ["downloaded", "downloaded"]
    assert (data / "demo" / "dir" / "b.txt").read_bytes() == b"beta"
    record = json.loads((data / "demo" / "_retrieval.json").read_text())
    assert record["files"]["a.txt"]["sha256"] == sha256(b"alpha")
    assert not list(data.rglob("*.part"))

    assert [r.outcome for r in fetch_sources(reg.sources, data)] == ["present", "present"]
    report = verify_sources(reg.sources, data)
    assert report.integrity_ok
    assert "Result: OK" in format_report(report)


def test_fetch_rejects_wrong_bytes(http_root: HTTP, tmp_path: Path, make_registry: Callable[..., Registry]) -> None:
    root, base = http_root
    publish(root, "a.txt", b"tampered")
    reg = make_registry(source_dict(assets=[{"path": "a.txt", "url": f"{base}/a.txt", "sha256": sha256(b"alpha")}]))
    data = tmp_path / "raw"
    [res] = fetch_sources(reg.sources, data)
    assert res.outcome == "failed" and "checksum mismatch" in res.detail
    assert not (data / "demo" / "a.txt").exists()
    assert not list(data.rglob("*.part"))


def test_fetch_never_overwrites_existing_raw_file(
    http_root: HTTP, tmp_path: Path, make_registry: Callable[..., Registry]
) -> None:
    root, base = http_root
    publish(root, "a.txt", b"alpha")
    reg = make_registry(source_dict(assets=[{"path": "a.txt", "url": f"{base}/a.txt", "sha256": sha256(b"alpha")}]))
    local = tmp_path / "raw" / "demo" / "a.txt"
    local.parent.mkdir(parents=True)
    local.write_bytes(b"corrupt")
    [res] = fetch_sources(reg.sources, tmp_path / "raw")
    assert res.outcome == "failed" and "never overwritten" in res.detail
    assert local.read_bytes() == b"corrupt"


def test_http_404_fails_without_retrying_forever(
    http_root: HTTP, tmp_path: Path, make_registry: Callable[..., Registry]
) -> None:
    _, base = http_root
    reg = make_registry(source_dict(assets=[{"path": "gone.txt", "url": f"{base}/gone.txt", "md5": md5(b"x")}]))
    [res] = fetch_sources(reg.sources, tmp_path / "raw", retries=5)
    assert res.outcome == "failed" and "404" in res.detail


def test_verify_reports_every_problem(tmp_path: Path, make_registry: Callable[..., Registry]) -> None:
    reg = make_registry(
        source_dict(
            "good",
            assets=[
                {"path": "ok.txt", "url": "u", "sha256": sha256(b"ok")},
                {"path": "bad.txt", "url": "u", "sha256": sha256(b"good")},
                {"path": "gone.txt", "url": "u", "size": 2048, "sha256": sha256(b"z")},
                {"path": "loose.tar.gz", "url": "u"},
            ],
        ),
        source_dict(
            "closed",
            license={"spdx": None, "status": "none_declared", "evidence": "e"},
            assets=[{"path": "x", "url": "u", "md5": md5(b"x")}],
        ),
        source_dict(
            "gpl",
            license={"spdx": "GPL-3.0", "status": "declared", "evidence": "e"},
            assets=[{"path": "y", "url": "u", "sha256": sha256(b"y")}],
        ),
    )
    data = tmp_path / "raw"
    for sid, name, content in [
        ("good", "ok.txt", b"ok"),
        ("good", "bad.txt", b"evil"),
        ("good", "loose.tar.gz", b"t"),
        ("good", "stray.bin", b"s"),
        ("closed", "x", b"x"),
        ("gpl", "y", b"y"),
    ]:
        (data / sid).mkdir(parents=True, exist_ok=True)
        (data / sid / name).write_bytes(content)

    report = verify_sources(reg.sources, data)
    statuses = {(c.source.id, a.path): a.status for c in report.checks for a in c.assets}
    assert statuses == {
        ("good", "ok.txt"): "ok",
        ("good", "bad.txt"): "mismatch",
        ("good", "gone.txt"): "missing",
        ("good", "loose.tar.gz"): "unpinned",
        ("closed", "x"): "ok",
        ("gpl", "y"): "ok",
    }
    assert not report.integrity_ok
    assert report.checks[0].untracked == ["stray.bin"]
    assert {c.source.id for c in report.license_issues()} == {"closed", "gpl"}
    text = format_report(report)
    for needle in (
        "Checksum mismatch (1",
        "Missing (1",
        "Unpinned (1",
        "Untracked",
        "stray.bin",
        "NONE DECLARED",
        "copyleft",
        "Verified by md5 only (1",
        "Result: FAIL",
    ):
        assert needle in text


def test_quick_verify_checks_sizes_only(tmp_path: Path, make_registry: Callable[..., Registry]) -> None:
    reg = make_registry(source_dict(assets=[{"path": "a", "url": "u", "size": 3, "sha256": sha256(b"abc")}]))
    (tmp_path / "demo").mkdir()
    (tmp_path / "demo" / "a").write_bytes(b"xyz")
    assert verify_sources(reg.sources, tmp_path, quick=True).integrity_ok
    assert not verify_sources(reg.sources, tmp_path).integrity_ok


def test_pin_records_sha256_and_refuses_md5_conflicts(tmp_path: Path, make_registry: Callable[..., Registry]) -> None:
    reg = make_registry(
        source_dict(
            assets=[
                {"path": "tar", "url": "u"},
                {"path": "zen", "url": "u", "md5": md5(b"zen")},
                {"path": "liar", "url": "u", "md5": md5(b"truth")},
                {"path": "absent", "url": "u"},
            ]
        )
    )
    (tmp_path / "demo").mkdir()
    for name, content in [("tar", b"tarball"), ("zen", b"zen"), ("liar", b"lie")]:
        (tmp_path / "demo" / name).write_bytes(content)

    outcomes = {r.path: r.outcome for r in pin_sources(reg, reg.sources, tmp_path)}
    assert outcomes == {"tar": "pinned", "zen": "pinned", "liar": "refused", "absent": "missing"}
    assets = {a.path: a for a in reg.sources[0].assets}
    assert assets["tar"].sha256 == sha256(b"tarball") and assets["tar"].size == 7
    assert assets["zen"].sha256 == sha256(b"zen")
    assert assets["liar"].sha256 is None
    assert {r.outcome for r in pin_sources(reg, reg.sources, tmp_path) if r.path == "tar"} == {"already"}


def test_cli_end_to_end(
    http_root: HTTP, tmp_path: Path, make_registry: Callable[..., Registry], capsys: pytest.CaptureFixture[str]
) -> None:
    root, base = http_root
    publish(root, "code.tar.gz", b"source code")
    reg = make_registry(
        source_dict(
            "code",
            kind="code",
            license={"spdx": None, "status": "none_declared", "evidence": "e"},
            assets=[{"path": "code.tar.gz", "url": f"{base}/code.tar.gz"}],
        )
    )
    path = tmp_path / "configs" / "datasets" / "sources.json"
    path.parent.mkdir(parents=True)
    reg_mod.save(reg, path)
    common = ["--registry", str(path), "--data-dir", str(tmp_path / "raw")]

    assert main([*common, "verify"]) == 1  # missing
    assert main([*common, "fetch", "--dry-run"]) == 0
    assert not (tmp_path / "raw").exists()
    assert main([*common, "fetch"]) == 0
    assert main([*common, "verify"]) == 1  # unpinned
    assert main([*common, "pin"]) == 0
    assert reg_mod.load(path).sources[0].assets[0].sha256 == sha256(b"source code")
    assert main([*common, "verify", "--json", str(tmp_path / "report.json")]) == 0
    assert main([*common, "verify", "--require-license"]) == 1
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["integrity_ok"] and report["sources"][0]["license"]["category"] == "unlicensed"
    assert "NONE DECLARED" in capsys.readouterr().out


def test_cli_unknown_source_exits(tmp_path: Path, make_registry: Callable[..., Registry]) -> None:
    path = tmp_path / "sources.json"
    reg_mod.save(make_registry(source_dict()), path)
    with pytest.raises(SystemExit, match="unknown source id"):
        main(["--registry", str(path), "--data-dir", str(tmp_path), "verify", "--source", "nope"])
