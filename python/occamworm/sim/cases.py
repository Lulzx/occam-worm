"""Conformance case loader (WRL_SYNTAX.md §7)."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from occamworm.sim.graph import SimInput, sim_input_from_json


@dataclass(frozen=True)
class Tolerance:
    abs: float = 1e-12
    rel: float = 0.0

    def close(self, actual: float, expected: float) -> bool:
        return abs(actual - expected) <= self.abs + self.rel * abs(expected)


@dataclass(frozen=True)
class Case:
    path: Path
    name: str
    source: str
    check: Mapping[str, Any]
    raw_input: Mapping[str, Any]
    program_hash: str | None

    @property
    def kind(self) -> str:
        return str(self.check["type"])

    @property
    def tolerance(self) -> Tolerance:
        t = self.check.get("tolerance", {})
        return Tolerance(float(t.get("abs", 1e-12)), float(t.get("rel", 0.0)))

    def sim_input(self) -> SimInput:
        return sim_input_from_json(self.raw_input)


def program_source(program: Mapping[str, Any], base: Path) -> str:
    """Inline ``source`` (string or list of lines) or a ``path`` relative to the case file."""
    if "source" in program:
        src = program["source"]
        return src if isinstance(src, str) else "".join(line + "\n" for line in src)
    return (base / str(program["path"])).read_text()


def load_case(path: Path | str) -> Case:
    p = Path(path)
    data = json.loads(p.read_text())
    return Case(
        path=p,
        name=str(data.get("name", p.stem)),
        source=program_source(data["program"], p.parent),
        check=data["check"],
        raw_input=data.get("input", {}),
        program_hash=data.get("program_hash"),
    )


def load_suite(directory: Path | str) -> list[Case]:
    return [load_case(p) for p in sorted(Path(directory).glob("[0-9]*.json"))]
