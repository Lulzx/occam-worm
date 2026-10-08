"""Run a program through the C++ reference interpreter (``ow sim run``) and parse its result (§6.8, §11)."""

from __future__ import annotations

import json
import subprocess
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from occamworm.sim.graph import InputError
from occamworm.sim.ir import find_ow
from occamworm.sim.result import SimResult, result_from_json


def simulate_cpp(source: str, raw_input: Mapping[str, Any], ow: Path | None = None) -> SimResult:
    """``ow sim run`` on WRL ``source`` and a simulation input (§6.1) given as a JSON-compatible mapping."""
    binary = ow if ow is not None else find_ow()
    assert binary is not None
    with tempfile.TemporaryDirectory() as tmp:
        wrl = Path(tmp) / "program.wrl"
        wrl.write_text(source)
        case = Path(tmp) / "case.json"
        case.write_text(json.dumps({"input": raw_input}))
        out = Path(tmp) / "out.json"
        proc = subprocess.run(
            [str(binary), "sim", "run", "--program", str(wrl), "--case", str(case), "--out", str(out)],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            raise InputError(proc.stderr.strip())
        return result_from_json(json.loads(out.read_text()))
