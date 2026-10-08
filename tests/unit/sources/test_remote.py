from __future__ import annotations

from typing import Any

from conftest import source_dict

from occamworm.sources.registry import License, from_dict
from occamworm.sources.remote import refresh_source

SHA_A, SHA_B = "a" * 64, "b" * 64
COMMIT_1, COMMIT_2 = "1" * 40, "2" * 40


def osf_file(path: str, sha: str, size: int = 10) -> dict[str, Any]:
    return {
        "attributes": {
            "kind": "file",
            "materialized_path": path,
            "size": size,
            "extra": {"hashes": {"sha256": sha, "md5": "c" * 32}},
            "date_modified": "2024-01-01",
        },
        "links": {"download": f"https://osf.io/download{path}"},
    }


def osf_api(files: list[dict[str, Any]], license_: bool = False) -> dict[str, Any]:
    folder = {
        "attributes": {"kind": "folder", "materialized_path": "/data/"},
        "relationships": {"files": {"links": {"related": {"href": "FOLDER"}}}},
    }
    skipped = {
        "attributes": {"kind": "folder", "materialized_path": "/other/"},
        "relationships": {"files": {"links": {"related": {"href": "OTHER"}}}},
    }
    # Page 2 repeats an entry from page 1, as the OSF API does with unstable sort orders.
    return {
        "ROOT?sort=name&page%5Bsize%5D=100": {"data": [folder, skipped], "links": {"next": None}},
        "FOLDER?sort=name&page%5Bsize%5D=100": {"data": files[:1], "links": {"next": "PAGE2"}},
        "PAGE2": {"data": files, "links": {"next": None}},
        "NODE": {"data": {"attributes": {"node_license": {"id": "x"} if license_ else None}}},
        "LICENSE": {"data": {"attributes": {"name": "CC-By Attribution 4.0 International"}}},
    }


def osf_getter(pages: dict[str, Any]) -> Any:
    def get(url: str) -> Any:
        url = (
            url.replace("https://api.osf.io/v2/nodes/e2syt/files/osfstorage/", "ROOT")
            .replace("https://api.osf.io/v2/nodes/e2syt/license/", "LICENSE")
            .replace("https://api.osf.io/v2/nodes/e2syt/", "NODE")
        )
        assert not url.startswith("OTHER"), "walked a folder outside the include list"
        return pages[url]

    return get


def osf_source(**kw: Any) -> Any:
    d = source_dict(
        provider="osf",
        remote={"provider": "osf", "node": "e2syt", "include": ["/data/"]},
        license={"spdx": None, "status": "none_declared", "evidence": "e", "checked_at": "2026-01-01"},
        **kw,
    )
    return from_dict({"schema_version": "0.1.0", "sources": [d]}).sources[0]


def test_osf_refresh_adds_dedupes_and_records_license() -> None:
    files = [osf_file("/data/1.txt", SHA_A), osf_file("/data/2.txt", SHA_B)]
    r = refresh_source(osf_source(), get=osf_getter(osf_api(files)))
    assert [a.path for a in r.source.assets] == ["data/1.txt", "data/2.txt"]
    assert r.added == ["data/1.txt", "data/2.txt"] and not r.needs_acceptance
    assert r.source.license.status == "none_declared" and r.source.metadata_retrieved_at


def test_changed_upstream_hash_is_reported_not_applied() -> None:
    src = osf_source(assets=[{"path": "data/1.txt", "url": "u", "size": 10, "sha256": SHA_A}])
    get = osf_getter(osf_api([osf_file("/data/1.txt", SHA_B)]))
    r = refresh_source(src, get=get)
    assert r.changed == ["data/1.txt"] and r.needs_acceptance
    assert r.source.assets[0].sha256 == SHA_A
    accepted = refresh_source(src, get=get, accept=True)
    assert accepted.source.assets[0].sha256 == SHA_B


def test_removed_upstream_is_kept_until_accepted() -> None:
    src = osf_source(assets=[{"path": "data/old.txt", "url": "u", "sha256": SHA_A}])
    get = osf_getter(osf_api([osf_file("/data/1.txt", SHA_B)]))
    r = refresh_source(src, get=get)
    assert r.removed == ["data/old.txt"] and r.added == ["data/1.txt"]
    assert {a.path for a in r.source.assets} == {"data/old.txt", "data/1.txt"}
    assert {a.path for a in refresh_source(src, get=get, accept=True).source.assets} == {"data/1.txt"}


def test_license_change_needs_acceptance() -> None:
    get = osf_getter(osf_api([osf_file("/data/1.txt", SHA_A)], license_=True))
    r = refresh_source(osf_source(), get=get)
    assert r.license_changed == "NONE DECLARED -> CC-BY-4.0"
    assert r.source.license.status == "none_declared"
    assert refresh_source(osf_source(), get=get, accept=True).source.license.spdx == "CC-BY-4.0"


def github_getter(commit: str, spdx: str | None) -> Any:
    def get(url: str) -> Any:
        if url.endswith("/commits/main"):
            return {"sha": commit}
        return {"license": {"spdx_id": spdx} if spdx else None}

    return get


def github_source(**remote: Any) -> Any:
    d = source_dict(
        "lib",
        kind="code",
        provider="github",
        license={"spdx": None, "status": "unknown", "evidence": "never checked", "checked_at": None},
        remote={"provider": "github", "repo": "org/lib", "ref": "main", **remote},
    )
    return from_dict({"schema_version": "0.1.0", "sources": [d]}).sources[0]


def test_github_first_refresh_pins_commit() -> None:
    r = refresh_source(github_source(), get=github_getter(COMMIT_1, "GPL-3.0"))
    assert r.source.remote["commit"] == COMMIT_1
    [asset] = r.source.assets
    assert asset.url == f"https://codeload.github.com/org/lib/tar.gz/{COMMIT_1}" and asset.sha256 is None
    assert r.source.license == License("GPL-3.0", "declared", r.source.license.evidence, r.source.license.checked_at)


def test_github_moved_ref_keeps_pinned_commit() -> None:
    first = refresh_source(github_source(), get=github_getter(COMMIT_1, None)).source
    moved = refresh_source(first, get=github_getter(COMMIT_2, None))
    assert moved.needs_acceptance and "moved" in moved.changed[0]
    assert moved.source.remote["commit"] == COMMIT_1 and moved.source.assets == first.assets
    accepted = refresh_source(first, get=github_getter(COMMIT_2, None), accept=True)
    assert accepted.source.remote["commit"] == COMMIT_2
    assert COMMIT_2 in accepted.source.assets[0].url


def test_zenodo_md5_only_keeps_local_sha256_pin() -> None:
    d = source_dict(
        "z",
        provider="zenodo",
        remote={"provider": "zenodo", "record": "9", "include": ["f.h5"]},
        license={"spdx": None, "status": "none_declared", "evidence": "e"},
        assets=[{"path": "f.h5", "url": "u", "size": 5, "sha256": SHA_A, "md5": "d" * 32}],
    )
    src = from_dict({"schema_version": "0.1.0", "sources": [d]}).sources[0]
    rec = {
        "id": 9,
        "metadata": {},
        "files": [
            {"key": "f.h5", "size": 5, "checksum": "md5:" + "d" * 32, "links": {"self": "https://z/f.h5"}},
            {"key": "huge.jld2", "size": 10**12, "checksum": "md5:" + "e" * 32, "links": {"self": "x"}},
        ],
    }
    r = refresh_source(src, get=lambda url: rec)
    assert not r.needs_acceptance
    [a] = r.source.assets
    assert a.sha256 == SHA_A and a.md5 == "d" * 32 and a.url == "https://z/f.h5"


def test_first_license_check_is_applied_without_acceptance() -> None:
    r = refresh_source(github_source(), get=github_getter(COMMIT_1, "MIT"))
    assert r.license_changed is None and r.source.license.spdx == "MIT" and r.source.license.checked_at
