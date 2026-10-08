"""Canonical WRL IR: loader, validator and access to the ``ow`` compiler (WRL_SYNTAX.md §5).

The Python simulators consume the IR JSON printed by ``ow rule inspect``; they never parse WRL themselves, so a
program accepted here was accepted by the C++ compiler. The IR is obtained from the ``ow`` binary (the
``OW_CLI`` environment variable, else ``build/clang/libs/ow-cli/ow`` under the repository root) or from a cached
copy in a directory of ``<sha256(source)>.json`` files (``compile_source(..., cache_dir=...)``).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

IR_SCHEMA = "occamworm.wrl.ir/0.1"
GRAMMAR_VERSION = "0.1"

# op -> (min arity, max arity or None for n-ary)
OP_ARITY: dict[str, tuple[int, int | None]] = {
    "const": (0, 0),
    "param": (0, 0),
    "state": (0, 0),
    "stimulus": (0, 0),
    "type_mask": (0, 0),
    "sum_in": (0, 0),
    "count_in": (0, 0),
    "delay": (0, 0),
    "add": (2, None),
    "mul": (2, None),
    "neg": (1, 1),
    "abs": (1, 1),
    "min": (2, None),
    "max": (2, None),
    "clamp": (3, 3),
    "relu": (1, 1),
    "tanh": (1, 1),
    "sigmoid": (1, 1),
    "threshold": (2, 2),
    "select": (3, 3),
    "lut": (1, 1),
    "leaky_integrate": (3, 3),
    "euler_leak": (3, 3),
}
OP_ATTRS: dict[str, tuple[str, ...]] = {
    "const": ("value",),
    "param": ("param",),
    "state": ("register",),
    "type_mask": ("type",),
    "sum_in": ("register", "select"),
    "count_in": ("register", "k"),
    "delay": ("register", "ticks"),
    "lut": ("table",),
}
SUM_SELECTS = ("exc", "inh", "all")
OBSERVATION_OPERATORS = ("identity_v1", "calcium_linear_v1")


class IrError(ValueError):
    """The IR JSON is malformed or violates the schema."""


class CompilerError(RuntimeError):
    """The ``ow`` compiler rejected a program (``code`` is its stable diagnostic code, e.g. ``E_UNIT``)."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


class CompilerNotFoundError(RuntimeError):
    """No ``ow`` binary was found; build it or set ``OW_CLI``."""


@dataclass(frozen=True)
class Register:
    index: int
    name: str
    source_name: str
    unit: str
    init: float


@dataclass(frozen=True)
class Parameter:
    index: int
    name: str
    source_name: str
    unit: str
    value: float
    lower: float
    upper: float
    trainable: bool
    bits: int


@dataclass(frozen=True)
class Instruction:
    id: int
    op: str
    args: tuple[int, ...]
    attrs: Mapping[str, Any]
    unit: str


@dataclass(frozen=True)
class Gap:
    register: int
    scale_param: int | None


@dataclass(frozen=True)
class Observation:
    operator: str
    register: int
    tau_param: int | None
    tau_const: float | None


@dataclass(frozen=True)
class Program:
    """A validated canonical program. ``raw`` keeps the full IR JSON (including ``l_struct``, ``l_params``)."""

    program_hash: str
    grammar_version: str
    compiler_build: str
    rule: str
    tier: str
    dt_max: float | None
    stimulus_unit: str
    registers: tuple[Register, ...]
    parameters: tuple[Parameter, ...]
    instructions: tuple[Instruction, ...]
    writes: tuple[int, ...]
    gap: Gap | None
    observation: Observation
    eliminated_registers: tuple[str, ...]
    eliminated_parameters: tuple[str, ...]
    raw: Mapping[str, Any] = field(repr=False, compare=False)

    def register_index(self, source_name: str) -> int | None:
        for reg in self.registers:
            if reg.source_name == source_name:
                return reg.index
        return None

    def parameter_index(self, source_name: str) -> int | None:
        for par in self.parameters:
            if par.source_name == source_name:
                return par.index
        return None

    @property
    def max_delay_ticks(self) -> int:
        """Largest ``delay(r, n)`` instruction delay (edge delays come from the graph)."""
        return max((int(i.attrs["ticks"]) for i in self.instructions if i.op == "delay"), default=0)

    @property
    def l_struct_bits(self) -> int:
        return int(self.raw["l_struct"]["total_bits"])

    @property
    def l_params_bits(self) -> int:
        return int(self.raw["l_params"]["total_bits"])

    def default_theta(self) -> list[float]:
        """Declared parameter values (midpoint of the bounds for trainable ones), in IR order."""
        return [p.value for p in self.parameters]

    def theta_from_dict(self, overrides: Mapping[str, float]) -> list[float]:
        """Parameter vector from overrides by source name; bounds-checked like the C++ interpreter.

        Names of parameters removed as dead code are ignored, anything else unknown is an error.
        """
        theta = self.default_theta()
        for name, value in overrides.items():
            idx = self.parameter_index(name)
            if idx is None:
                if name in self.eliminated_parameters:
                    continue
                raise ValueError(f"unknown parameter '{name}'")
            par = self.parameters[idx]
            if not math.isfinite(value) or value < par.lower or value > par.upper:
                raise ValueError(f"parameter '{name}' lies outside its declared bounds")
            theta[idx] = float(value)
        return theta


def _number(value: Any, what: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise IrError(f"{what} must be a number, got {value!r}")
    return float(value)


def _integer(value: Any, what: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise IrError(f"{what} must be an integer, got {value!r}")
    return value


def _require(obj: Mapping[str, Any], key: str, what: str) -> Any:
    if not isinstance(obj, Mapping) or key not in obj:
        raise IrError(f"{what}: missing key '{key}'")
    return obj[key]


def _check_instruction(ins: Instruction, n_registers: int, n_params: int) -> None:
    if ins.op not in OP_ARITY:
        raise IrError(f"instruction n{ins.id}: unknown op '{ins.op}'")
    lo, hi = OP_ARITY[ins.op]
    if len(ins.args) < lo or (hi is not None and len(ins.args) > hi):
        raise IrError(f"instruction n{ins.id}: op '{ins.op}' has {len(ins.args)} arguments")
    for a in ins.args:
        if not 0 <= a < ins.id:
            raise IrError(f"instruction n{ins.id}: argument {a} does not refer to an earlier instruction")
    for key in OP_ATTRS.get(ins.op, ()):
        if key not in ins.attrs:
            raise IrError(f"instruction n{ins.id}: op '{ins.op}' lacks attribute '{key}'")
    for key in ("register",):
        if key in ins.attrs and not 0 <= _integer(ins.attrs[key], "register") < n_registers:
            raise IrError(f"instruction n{ins.id}: register index out of range")
    if ins.op == "param" and not 0 <= _integer(ins.attrs["param"], "param") < n_params:
        raise IrError(f"instruction n{ins.id}: parameter index out of range")
    if ins.op == "sum_in" and ins.attrs["select"] not in SUM_SELECTS:
        raise IrError(f"instruction n{ins.id}: bad sum_in selector {ins.attrs['select']!r}")
    if ins.op == "delay" and not 0 <= _integer(ins.attrs["ticks"], "ticks") <= 100000:
        raise IrError(f"instruction n{ins.id}: delay ticks out of range")
    if ins.op == "count_in" and _integer(ins.attrs["k"], "k") < 0:
        raise IrError(f"instruction n{ins.id}: count_in k must be >= 0")
    if ins.op == "lut":
        table = ins.attrs["table"]
        if not isinstance(table, list) or not 1 <= len(table) <= 4096:
            raise IrError(f"instruction n{ins.id}: lut table must have 1..4096 entries")
    if ins.op == "const":
        _number(ins.attrs["value"], "const value")


def load_ir(data: Mapping[str, Any]) -> Program:
    """Validate an IR JSON object (``ow rule inspect`` output) and return a :class:`Program`."""
    if _require(data, "schema", "ir") != IR_SCHEMA:
        raise IrError(f"unsupported IR schema {data['schema']!r}, expected {IR_SCHEMA!r}")
    if _require(data, "grammar_version", "ir") != GRAMMAR_VERSION:
        raise IrError(f"unsupported grammar version {data['grammar_version']!r}")
    program_hash = str(_require(data, "program_hash", "ir"))
    if not re.fullmatch(r"[0-9a-f]{64}", program_hash):
        raise IrError("program_hash must be 64 lowercase hex digits")
    tier = _require(data, "tier", "ir")
    if tier not in ("G0", "G1"):
        raise IrError(f"unsupported tier {tier!r}")

    registers = []
    for k, r in enumerate(_require(data, "registers", "ir")):
        if _integer(r["index"], "register index") != k:
            raise IrError("register indices must be 0..R-1 in order")
        registers.append(Register(k, str(r["name"]), str(r["source_name"]), str(r["unit"]), _number(r["init"], "init")))
    if not registers:
        raise IrError("a program needs at least one register")

    parameters = []
    for k, p in enumerate(_require(data, "parameters", "ir")):
        if _integer(p["index"], "parameter index") != k:
            raise IrError("parameter indices must be 0..P-1 in order")
        par = Parameter(
            k,
            str(p["name"]),
            str(p["source_name"]),
            str(p["unit"]),
            _number(p["value"], "value"),
            _number(p["lower"], "lower"),
            _number(p["upper"], "upper"),
            bool(p["trainable"]),
            _integer(p["bits"], "bits"),
        )
        if not par.lower <= par.value <= par.upper:
            raise IrError(f"parameter {par.source_name}: value outside bounds")
        if par.trainable and not par.lower < par.upper:
            raise IrError(f"parameter {par.source_name}: trainable bounds must satisfy lower < upper")
        parameters.append(par)

    instructions = []
    for k, i in enumerate(_require(data, "instructions", "ir")):
        if _integer(i["id"], "instruction id") != k:
            raise IrError("instruction ids must be 0..n-1 in evaluation order")
        ins = Instruction(
            k, str(i["op"]), tuple(_integer(a, "argument") for a in i["args"]), dict(i.get("attrs", {})), str(i["unit"])
        )
        _check_instruction(ins, len(registers), len(parameters))
        instructions.append(ins)

    writes = []
    for w in _require(data, "writes", "ir"):
        if not 0 <= _integer(w["value"], "write value") < len(instructions):
            raise IrError("write refers to an unknown instruction")
        writes.append(int(w["value"]))
    if len(writes) != len(registers) or [int(w["register"]) for w in data["writes"]] != list(range(len(registers))):
        raise IrError("writes must list every register in index order")

    gap = None
    if data.get("gap") is not None:
        g = data["gap"]
        scale = g.get("scale_param")
        reg = _integer(g["register"], "gap register")
        if not 0 <= reg < len(registers) or (scale is not None and not 0 <= _integer(scale, "scale") < len(parameters)):
            raise IrError("gap refers to an unknown register or parameter")
        gap = Gap(reg, None if scale is None else int(scale))

    o = _require(data, "observation", "ir")
    if o["operator"] not in OBSERVATION_OPERATORS:
        raise IrError(f"unknown observation operator {o['operator']!r}")
    obs_reg = _integer(o["register"], "observation register")
    if not 0 <= obs_reg < len(registers):
        raise IrError("observation register out of range")
    tau = o.get("tau")
    tau_param: int | None = None
    tau_const: float | None = None
    if o["operator"] == "calcium_linear_v1":
        if not isinstance(tau, Mapping) or ("param" in tau) == ("const" in tau):
            raise IrError("calcium_linear_v1 needs tau as exactly one of {'param': i} or {'const': x}")
        if "param" in tau:
            tau_param = _integer(tau["param"], "tau param")
            if not 0 <= tau_param < len(parameters):
                raise IrError("observation tau parameter out of range")
        else:
            tau_const = _number(tau["const"], "tau const")
    observation = Observation(str(o["operator"]), obs_reg, tau_param, tau_const)

    eliminated = data.get("eliminated", {})
    dt_max = data.get("dt_max")
    return Program(
        program_hash=program_hash,
        grammar_version=str(data["grammar_version"]),
        compiler_build=str(data.get("compiler_build", "")),
        rule=str(data.get("rule", "")),
        tier=str(tier),
        dt_max=None if dt_max is None else _number(dt_max, "dt_max"),
        stimulus_unit=str(data.get("stimulus_unit", "1")),
        registers=tuple(registers),
        parameters=tuple(parameters),
        instructions=tuple(instructions),
        writes=tuple(writes),
        gap=gap,
        observation=observation,
        eliminated_registers=tuple(eliminated.get("registers", ())),
        eliminated_parameters=tuple(eliminated.get("parameters", ())),
        raw=data,
    )


# ---------------------------------------------------------------------------------------------------------------
# The ow binary


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def find_ow(required: bool = True) -> Path | None:
    """``$OW_CLI``, else ``<repo>/build/clang/libs/ow-cli/ow``. Raises :class:`CompilerNotFoundError` if absent."""
    env = os.environ.get("OW_CLI")
    candidate = Path(env) if env else repo_root() / "build" / "clang" / "libs" / "ow-cli" / "ow"
    if candidate.is_file() and os.access(candidate, os.X_OK):
        return candidate
    if not required:
        return None
    where = f"$OW_CLI={env}" if env else str(candidate)
    raise CompilerNotFoundError(
        f"the ow compiler was not found at {where}. Build it with `cmake --preset clang && cmake --build --preset "
        "clang` or point the OW_CLI environment variable at the binary."
    )


def run_ow(args: list[str], ow: Path | None = None) -> subprocess.CompletedProcess[str]:
    binary = ow if ow is not None else find_ow()
    assert binary is not None
    return subprocess.run([str(binary), *args], capture_output=True, text=True, check=False)


_DIAGNOSTIC = re.compile(r"error:\s*(E_[A-Z]+)(?:\s+at\s+\d+:\d+)?:?\s*(.*)")


def _raise_compiler_error(stderr: str) -> None:
    m = _DIAGNOSTIC.search(stderr)
    if m:
        raise CompilerError(m.group(1), m.group(2).strip())
    raise CompilerError("E_UNKNOWN", stderr.strip())


def source_digest(source: str) -> str:
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def inspect_source(source: str, ow: Path | None = None) -> str:
    """The IR JSON text printed by ``ow rule inspect`` for a WRL source; raises :class:`CompilerError`."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "program.wrl"
        path.write_text(source)
        proc = run_ow(["rule", "inspect", str(path)], ow)
    if proc.returncode != 0:
        _raise_compiler_error(proc.stderr)
    return proc.stdout


def compile_source(source: str, cache_dir: Path | None = None, ow: Path | None = None) -> Program:
    """Compile WRL source text through ``ow rule inspect``.

    With ``cache_dir`` the IR is read from ``<cache_dir>/<sha256(source)>.json`` when the binary is absent. The
    binary, when present, is always used (the cache never overrides the compiler).
    """
    cached = None if cache_dir is None else cache_dir / f"{source_digest(source)}.json"
    binary = ow if ow is not None else find_ow(required=cached is None or not cached.is_file())
    if binary is None:
        assert cached is not None
        return load_ir(json.loads(cached.read_text()))
    return load_ir(json.loads(inspect_source(source, binary)))


def compile_file(path: Path | str, ow: Path | None = None) -> Program:
    """Compile a ``.wrl`` file, or load a ``.json`` IR file (an ``ow rule inspect`` dump)."""
    p = Path(path)
    if p.suffix == ".json":
        return load_ir(json.loads(p.read_text()))
    proc = run_ow(["rule", "inspect", str(p)], ow)
    if proc.returncode != 0:
        _raise_compiler_error(proc.stderr)
    return load_ir(json.loads(proc.stdout))


def check_rejected(source: str, ow: Path | None = None) -> str:
    """Compile ``source`` and return the diagnostic code; raises if the compiler accepts the program."""
    try:
        compile_source(source, ow=ow)
    except CompilerError as error:
        return error.code
    raise AssertionError("the ow compiler accepted a program that was expected to be rejected")
