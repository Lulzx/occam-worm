"""Readers for the data files bundled in the Worm Neuro Atlas repository (source ``wormneuroatlas``).

LICENSE RULE: the repository's code is GPL-3.0. Nothing from it is imported, vendored or copied here; these
functions parse its plain-text, JSON and XLSX *data* files, which keep the terms of their original
publications (see docs/data/ANNOTATIONS.md). The data files are read from the verified tarball without
extracting it.
"""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from io import StringIO
from pathlib import Path

from occamworm.importers._xlsx import read_workbook

SOURCE = "wormneuroatlas"

_BASE = "wormneuroatlas/data/"
NEURON_IDS = _BASE + "neuron_ids.txt"
GANGLIA = _BASE + "aconnectome_ids_ganglia.json"
POSITIONS = _BASE + "anatlas_neuron_positions.txt"
LINEAGE = _BASE + "cell_lineage.txt"
FENYVES = _BASE + "journal.pcbi.1007974.s003.xlsx"
WHITE_WHOLE_CSV = _BASE + "aconnectome_white_1986_whole.csv"
WITVLIET_CSV = {7: _BASE + "aconnectome_witvliet_2020_7.csv", 8: _BASE + "aconnectome_witvliet_2020_8.csv"}
BENTLEY_MA = _BASE + "esconnectome_monoamines_Bentley_2016.csv"
BENTLEY_NP = _BASE + "esconnectome_neuropeptides_Bentley_2016.csv"

_NAMELIKE = re.compile(r"^[A-Za-z][A-Za-z0-9]*$")


class AtlasDataError(Exception):
    """A Worm Neuro Atlas data file does not have the expected structure."""


def archive_path(raw_dir: Path) -> Path:
    found = sorted((raw_dir / SOURCE).glob("*.tar.gz"))
    if len(found) != 1:
        raise AtlasDataError(f"expected exactly one tarball in {raw_dir / SOURCE}, found {len(found)}")
    return found[0]


def read_neuron_ids(data: bytes) -> list[str]:
    """Atlas node order: ``index<TAB>name`` per line."""
    out = []
    for line in data.decode("utf-8").splitlines():
        parts = line.split("\t")
        if len(parts) == 2 and parts[1].strip():
            out.append(parts[1].strip())
    return out


@dataclass(frozen=True)
class Ganglia:
    groups: dict[str, list[str]]  # ganglion or region name -> neuron names
    pharyngeal: frozenset[str]  # group names inside the pharynx
    head: frozenset[str]  # group names inside the head


def read_ganglia(data: bytes) -> Ganglia:
    raw = json.loads(data)
    head = frozenset(raw["head"])
    pharynx = frozenset(raw["pharynx"])
    groups = {k: list(v) for k, v in raw.items() if k not in ("head", "pharynx")}
    return Ganglia(groups, pharynx, head)


@dataclass(frozen=True)
class Position:
    name: str
    x: float
    y: float
    z: float


def read_positions(data: bytes) -> list[Position]:
    """``#`` header line lists names in order; each following line is ``x y z`` for the same index."""
    lines = data.decode("utf-8").splitlines()
    if not lines or not lines[0].startswith("#"):
        raise AtlasDataError("positions file: expected a '#' header listing neuron names")
    names = lines[0].lstrip("#").split()
    coords = [ln.split() for ln in lines[1:] if ln.strip()]
    if len(coords) != len(names):
        raise AtlasDataError(f"positions file: {len(names)} names but {len(coords)} coordinate rows")
    return [Position(n, float(c[0]), float(c[1]), float(c[2])) for n, c in zip(names, coords, strict=True)]


def read_lineage(data: bytes) -> dict[str, str]:
    """Cell name -> description (first occurrence) from ``cell_lineage.txt``; covers neurons, glia and others."""
    out: dict[str, str] = {}
    for line in data.decode("utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        name = parts[0].strip()
        if name and name not in out:
            out[name] = parts[2].strip() if len(parts) > 2 else ""
    return out


def read_aconnectome_csv(data: bytes) -> list[tuple[str, str, str, float]]:
    """``pre<TAB>post<TAB>type<TAB>synapses`` rows (``chemical`` or ``electrical``)."""
    rows = list(csv.reader(StringIO(data.decode("utf-8")), delimiter="\t"))
    if not rows or rows[0][:4] != ["pre", "post", "type", "synapses"]:
        raise AtlasDataError(f"unexpected connectome header {rows[0] if rows else None}")
    return [(r[0], r[1], r[2], float(r[3])) for r in rows[1:] if len(r) >= 4]


# --------------------------------------------------------------------------- Fenyves et al. 2020


@dataclass(frozen=True)
class FenyvesNT:
    neuron: str
    dominant: str | None
    alternative: str | None


@dataclass(frozen=True)
class FenyvesEdge:
    pre: str
    post: str
    weight: float
    edge_type: str
    polarity: str  # '+', '-', 'complex' or 'no pred'


@dataclass(frozen=True)
class Fenyves:
    transmitters: list[FenyvesNT]
    types: dict[str, str]  # neuron -> raw 'Neuron type'
    receptors: dict[str, list[str]]  # neuron -> ionotropic receptor genes
    edges: list[FenyvesEdge]


def _sheet(sheets: dict[str, list[list[str | None]]], prefix: str) -> list[list[str | None]]:
    for name, rows in sheets.items():
        if name.startswith(prefix):
            return rows
    raise AtlasDataError(f"Fenyves workbook: no sheet starting with {prefix!r}")


def _cell(row: list[str | None], i: int) -> str | None:
    if i >= len(row):
        return None
    v = row[i]
    if v is None:
        return None
    v = v.strip()
    return v or None


def read_fenyves(data: bytes) -> Fenyves:
    """Fenyves et al. 2020 (PLoS Comput Biol 16:e1007974) S3: transmitter expression and sign predictions.

    The sign column is a published *prediction* from transmitter and receptor expression, not a measurement.
    """
    sheets = read_workbook(data)
    nt_rows = _sheet(sheets, "1.")
    transmitters = [
        FenyvesNT(n, _cell(r, 1), _cell(r, 2)) for r in nt_rows[1:] if (n := _cell(r, 0)) and _NAMELIKE.match(n)
    ]
    receptors: dict[str, list[str]] = {}
    for r in _sheet(sheets, "3."):
        n = _cell(r, 0)
        if n and _NAMELIKE.match(n) and not n.startswith("Neuron"):
            receptors[n] = [g for i in range(1, len(r)) if (g := _cell(r, i))]
    types: dict[str, str] = {}
    for r in _sheet(sheets, "4.")[2:]:
        n = _cell(r, 0)
        t = _cell(r, 7)
        if n and t and _NAMELIKE.match(n):
            types[n] = t
    edges: list[FenyvesEdge] = []
    for r in _sheet(sheets, "5.")[2:]:
        pre, post, w, kind, pol = _cell(r, 0), _cell(r, 3), _cell(r, 4), _cell(r, 5), _cell(r, 16)
        if pre and post and w and kind and pol:
            edges.append(FenyvesEdge(pre, post, float(w), kind, pol))
    if not transmitters or not edges:
        raise AtlasDataError("Fenyves workbook: empty transmitter or sign sheet")
    return Fenyves(transmitters, types, receptors, edges)
