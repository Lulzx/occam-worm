"""Shared fixtures: programs are compiled by the ``ow`` binary (tests needing it are skipped when it is absent)."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from occamworm.sim.ir import Program, compile_source, find_ow


@pytest.fixture(scope="session")
def compile_wrl() -> Callable[[str], Program]:
    if find_ow(required=False) is None:
        pytest.skip("the ow binary was not found: build it (cmake --preset clang) or set OW_CLI")
    cache: dict[str, Program] = {}

    def compile_(source: str) -> Program:
        if source not in cache:
            cache[source] = compile_source(source)
        return cache[source]

    return compile_
