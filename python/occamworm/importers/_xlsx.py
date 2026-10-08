"""Minimal reader for ``.xlsx`` workbooks (stdlib only): cell values as text, sheet by sheet.

Only what the annotation sources need: shared and inline strings, numbers as their stored text, no formulas
or formatting. Rows are returned as lists padded with ``None`` for empty cells.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
import zipfile
from io import BytesIO

_MAIN = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
_REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
_COL = re.compile(r"[A-Z]+")


def _column(ref: str) -> int:
    m = _COL.match(ref)
    if m is None:
        raise ValueError(f"bad cell reference {ref!r}")
    n = 0
    for ch in m.group(0):
        n = n * 26 + ord(ch) - 64
    return n - 1


def _text(node: ET.Element) -> str:
    return "".join(t.text or "" for t in node.iter(f"{{{_MAIN}}}t"))


def read_workbook(data: bytes) -> dict[str, list[list[str | None]]]:
    z = zipfile.ZipFile(BytesIO(data))
    shared: list[str] = []
    if "xl/sharedStrings.xml" in z.namelist():
        shared = [_text(si) for si in ET.fromstring(z.read("xl/sharedStrings.xml")).findall(f"{{{_MAIN}}}si")]
    workbook = ET.fromstring(z.read("xl/workbook.xml"))
    rels = {r.get("Id"): r.get("Target", "") for r in ET.fromstring(z.read("xl/_rels/workbook.xml.rels"))}
    sheets = workbook.find(f"{{{_MAIN}}}sheets")
    out: dict[str, list[list[str | None]]] = {}
    for sheet in sheets if sheets is not None else []:
        target = rels[sheet.get(f"{{{_REL}}}id", "")].lstrip("/")
        path = target if target.startswith("xl/") else f"xl/{target}"
        rows: list[list[str | None]] = []
        for row in ET.fromstring(z.read(path)).iter(f"{{{_MAIN}}}row"):
            cells: dict[int, str] = {}
            for c in row.findall(f"{{{_MAIN}}}c"):
                kind = c.get("t")
                v = c.find(f"{{{_MAIN}}}v")
                if kind == "inlineStr":
                    value = _text(c)
                elif v is None or v.text is None:
                    continue
                elif kind == "s":
                    value = shared[int(v.text)]
                else:
                    value = v.text
                cells[_column(c.get("r", "A1"))] = value
            rows.append([cells.get(i) for i in range(max(cells) + 1)] if cells else [])
        out[sheet.get("name", "")] = rows
    return out
