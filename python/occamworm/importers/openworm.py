"""Readers for the OpenWorm ConnectomeToolbox tarball (MIT code; bundled datasets keep their own terms).

Source ``openworm-connectome-toolbox`` (registry id). Only data files are read, from the verified tarball and
without extracting it: JSON adjacency caches written by the Toolbox readers, CSV and XLSX tables, and the list
literals of ``cect/Cells.py``, which is parsed statically (``ast``) and never imported or executed.

Every function here returns upstream values as found; name normalization and conflict handling belong to
:mod:`occamworm.importers.annotations`. See docs/data/ANNOTATIONS.md.
"""

from __future__ import annotations

import ast
import csv
import json
from dataclasses import dataclass
from io import StringIO
from pathlib import Path

import numpy as np

from occamworm.importers._xlsx import read_workbook

SOURCE = "openworm-connectome-toolbox"

CELLS_PY = "cect/Cells.py"
CELL_INFO = "cect/data/all_cell_info.csv"
WANG_NT = "cect/data/elife-95402-supp2-v1.xlsx"
RIPOLL_NEURONS = "cect/data/1-s2.0-S0896627323007560-mmc7.xlsx"
BENTLEY_MA = "cect/data/edgelist_MA.csv"
BENTLEY_NP = "cect/data/edgelist_NP.csv"
RIPOLL_MATRICES = {
    "short": "cect/data/01022024_neuropeptide_connectome_short_range_model.csv",
    "mid": "cect/data/01022024_neuropeptide_connectome_mid_range_model.csv",
    "long": "cect/data/01022024_neuropeptide_connectome_long_range_model.csv",
}

# Synapse classes of the Toolbox that carry no neurotransmitter assignment.
CHEMICAL_CLASS = "Generic_CS"
GAP_CLASS = "Generic_GJ"

NEURON_LISTS = (
    "PREFERRED_HERM_NEURON_NAMES",
    "SENSORY_NEURONS_COOK",
    "INTERNEURONS_COOK",
    "MOTORNEURONS_COOK",
    "PHARYNGEAL_POLYMODAL_NEURONS",
    "UNKNOWN_FUNCTION_NEURONS",
    "PHARYNGEAL_NEURONS",
    "PREFERRED_MUSCLE_NAMES",
)


class ToolboxError(Exception):
    """A Toolbox file does not have the expected structure."""


def cache_member(reader: str) -> str:
    return f"cect/cache/{reader}.json"


def archive_path(raw_dir: Path) -> Path:
    found = sorted((raw_dir / SOURCE).glob("*.tar.gz"))
    if len(found) != 1:
        raise ToolboxError(f"expected exactly one tarball in {raw_dir / SOURCE}, found {len(found)}")
    return found[0]


# --------------------------------------------------------------------------- adjacency caches


@dataclass(frozen=True)
class Cache:
    """A Toolbox reader cache: square matrices indexed by ``nodes``, rows are presynaptic cells."""

    nodes: list[str]
    connections: dict[str, np.ndarray]


def parse_cache(data: bytes) -> Cache:
    raw = json.loads(data)
    nodes = [str(n) for n in raw["nodes"]]
    conns: dict[str, np.ndarray] = {}
    for synclass, matrix in raw["connections"].items():
        arr = np.asarray(matrix, dtype=np.float64)
        if arr.shape != (len(nodes), len(nodes)):
            raise ToolboxError(f"cache matrix {synclass} has shape {arr.shape} for {len(nodes)} nodes")
        conns[str(synclass)] = arr
    return Cache(nodes, conns)


def cache_edges(cache: Cache, synclass: str) -> list[tuple[str, str, float]]:
    """Nonzero entries ``(pre, post, weight)``, pre = matrix row."""
    arr = cache.connections[synclass]
    rows, cols = np.nonzero(arr)
    return [(cache.nodes[i], cache.nodes[j], float(arr[i, j])) for i, j in zip(rows, cols, strict=True)]


# --------------------------------------------------------------------------- Cells.py list literals


def _eval_list(node: ast.expr, lists: dict[str, list[str]], strings: dict[str, str]) -> list[str]:
    if isinstance(node, ast.List):
        out: list[str] = []
        for e in node.elts:
            if isinstance(e, ast.Constant) and isinstance(e.value, str):
                out.append(e.value)
            elif isinstance(e, ast.Name):
                out.append(strings[e.id])
            else:
                raise ValueError("not a list of strings")
        return out
    if isinstance(node, ast.Name):
        return lists[node.id]
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return _eval_list(node.left, lists, strings) + _eval_list(node.right, lists, strings)
    raise ValueError("unsupported expression")


def static_lists(cells_py: bytes) -> dict[str, list[str]]:
    """Module-level assignments of string lists (literals joined with ``+``) from ``cect/Cells.py``."""
    lists: dict[str, list[str]] = {}
    strings: dict[str, str] = {}
    for node in ast.parse(cells_py.decode("utf-8")).body:
        if not (isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)):
            continue
        name = node.targets[0].id
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            strings[name] = node.value.value
            continue
        try:
            lists[name] = _eval_list(node.value, lists, strings)
        except (ValueError, KeyError):
            continue
    missing = [k for k in NEURON_LISTS if k not in lists]
    if missing:
        raise ToolboxError(f"{CELLS_PY}: lists not found: {missing}")
    return {k: lists[k] for k in NEURON_LISTS}


# --------------------------------------------------------------------------- neuron tables


@dataclass(frozen=True)
class CellInfoRow:
    name: str
    cell_type: str
    classification: str


def read_cell_info(data: bytes) -> dict[str, CellInfoRow]:
    """WormAtlas cell table (``all_cell_info.csv``): name, type, classification."""
    rows = list(csv.reader(StringIO(data.decode("utf-8"))))
    if not rows or rows[0][:2] != ["Cell name", "Type"]:
        raise ToolboxError(f"{CELL_INFO}: unexpected header {rows[0] if rows else None}")
    return {r[0]: CellInfoRow(r[0], r[1], r[4] if len(r) > 4 else "") for r in rows[1:] if len(r) >= 2 and r[0]}


@dataclass(frozen=True)
class WangRow:
    neuron: str
    cell_class: str | None
    neurotransmitter_raw: str | None
    comments: str | None


def read_wang_nt(data: bytes) -> list[WangRow]:
    """Wang et al. 2024 (eLife 95402) supplementary file 2: transmitter pathway expression, hermaphrodite."""
    sheets = read_workbook(data)
    rows = next(iter(sheets.values()))
    header_at = next((i for i, r in enumerate(rows) if len(r) > 2 and r[2] == "Neuron"), None)
    if header_at is None:
        raise ToolboxError(f"{WANG_NT}: no header row with 'Neuron'")
    header = [(h or "").strip() for h in rows[header_at]]
    nt_col = next((i for i, h in enumerate(header) if h.startswith("Neurotransmitter")), None)
    if nt_col is None:
        raise ToolboxError(f"{WANG_NT}: no 'Neurotransmitter(s)' column")
    class_cols = [i for i, h in enumerate(header) if h == "Class"]
    comments = header.index("Comments") if "Comments" in header else None

    def cell(r: list[str | None], i: int | None) -> str | None:
        if i is None or i >= len(r) or r[i] is None:
            return None
        v = str(r[i]).strip()
        return v or None

    out: list[WangRow] = []
    for r in rows[header_at + 1 :]:
        neuron = cell(r, 2)
        if neuron is None:
            continue
        cls = cell(r, class_cols[-1]) or cell(r, class_cols[0])
        out.append(WangRow(neuron, cls, cell(r, nt_col), cell(r, comments)))
    return out


@dataclass(frozen=True)
class RipollNeuronRow:
    neuron: str
    cell_type: str | None
    segment: str | None


def read_ripoll_neurons(data: bytes) -> list[RipollNeuronRow]:
    """Ripoll-Sanchez et al. 2023 (Neuron 111) table S7 ``Neuron information``: type and body segment."""
    rows = read_workbook(data).get("Neuron information")
    if not rows or not rows[0] or rows[0][0] != "Neuron":
        raise ToolboxError(f"{RIPOLL_NEURONS}: unexpected layout")

    def cell(r: list[str | None], i: int) -> str | None:
        v = r[i] if i < len(r) else None
        return v.strip() or None if v else None

    return [RipollNeuronRow(str(r[0]).strip(), cell(r, 2), cell(r, 3)) for r in rows[2:] if r and r[0]]


# --------------------------------------------------------------------------- extrasynaptic tables


def read_edgelist(data: bytes) -> list[tuple[str, str, str, str]]:
    """Bentley et al. 2016 edge lists: ``pre, post, transmitter, receptor`` per line, ``#`` comments."""
    out: list[tuple[str, str, str, str]] = []
    for r in csv.reader(StringIO(data.decode("utf-8"))):
        if len(r) < 4 or not r[0] or r[0].startswith("#"):
            continue
        out.append((r[0].strip(), r[1].strip(), r[2].strip(), r[3].strip()))
    return out


def read_matrix_csv(data: bytes) -> list[tuple[str, str, float]]:
    """A square CSV adjacency matrix with a header row and a leading row-label column (rows are presynaptic)."""
    rows = list(csv.reader(StringIO(data.decode("utf-8"))))
    if not rows:
        raise ToolboxError("empty matrix file")
    cols = [c.strip() for c in rows[0][1:]]
    out: list[tuple[str, str, float]] = []
    for r in rows[1:]:
        if not r or not r[0]:
            continue
        if len(r) != len(cols) + 1:
            raise ToolboxError(f"matrix row {r[0]!r} has {len(r) - 1} values for {len(cols)} columns")
        for c, v in zip(cols, r[1:], strict=True):
            w = float(v)
            if w != 0.0:
                out.append((r[0].strip(), c, w))
    return out
