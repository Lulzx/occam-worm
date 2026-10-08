"""Explicit, versioned alias table from observed cell labels to canonical hermaphrodite neuron IDs (OW-013).

The canonical set is the 302 hermaphrodite neuron names of the OpenWorm Toolbox. A label is resolved only if
it is canonical, or listed in the alias table with a recorded kind and reason. Nothing is normalized silently:

* ``zero_padded``  ``VA01`` -> ``VA1`` (unique, resolved)
* ``case_variant`` differs from exactly one canonical name only by letter case (resolved)
* ``convention``   labels a source uses for a pair or a state (``DB1/3``, ``AWCON``); never resolved to one cell
* ``class_label``  a class or stem such as ``AVA``; ambiguous, one row per candidate member

Typos are not aliases. :func:`suggestions` proposes a single nearest canonical name for the audit table, but a
suggestion never resolves a label.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

MAPPING_VERSION = "ow013-aliases-v1"

# label -> (candidates, reason). Candidates that are not canonical names are dropped when the table is built.
CONVENTIONS: dict[str, tuple[tuple[str, ...], str]] = {
    "AWCON": (
        ("AWCL", "AWCR"),
        "AWC ON/OFF names a per-animal asymmetric state, not a side; the side is not inferred "
        "(wormneuroatlas pairs OFF with AWCL and ON with AWCR by convention)",
    ),
    "AWCOFF": (
        ("AWCL", "AWCR"),
        "AWC ON/OFF names a per-animal asymmetric state, not a side; the side is not inferred "
        "(wormneuroatlas pairs OFF with AWCL and ON with AWCR by convention)",
    ),
    "DB1/3": (
        ("DB1", "DB3"),
        "Wang 2024 pair label; the table does not say which row is DB1 and which is DB3 (the Toolbox maps "
        "the first to DB1 by position)",
    ),
    "DB3/1": (
        ("DB1", "DB3"),
        "Wang 2024 pair label; the table does not say which row is DB1 and which is DB3 (the Toolbox maps "
        "the first to DB1 by position)",
    ),
}

_ZERO_PADDED = re.compile(r"^([A-Za-z]+)0(\d+)$")
_STEM_SUFFIX = re.compile(r"^(\d+)?([DV])?([LR])?$")


@dataclass(frozen=True)
class AliasRow:
    alias: str
    canonical_neuron_id: str
    kind: str
    ambiguous: bool
    n_candidates: int
    reason: str
    observed_in: tuple[str, ...]


@dataclass(frozen=True)
class Resolution:
    status: str  # exact | alias | ambiguous | unresolved
    candidates: tuple[str, ...]
    kind: str | None
    reason: str

    @property
    def neuron_id(self) -> str | None:
        return self.candidates[0] if self.status in ("exact", "alias") else None


def stem_candidates(label: str, canonical: Iterable[str]) -> list[str]:
    """Canonical names that extend ``label`` by a side, a dorsal/ventral letter or a member number."""
    return sorted(
        c
        for c in canonical
        if c != label and c.startswith(label) and len(c) > len(label) and _STEM_SUFFIX.match(c[len(label) :])
    )


def _stem_reason(label: str, candidates: Sequence[str]) -> str:
    sufs = [c[len(label) :] for c in candidates]
    if all(s in ("L", "R") for s in sufs):
        return "left/right not specified"
    if all(re.fullmatch(r"[DV][LR]", s) for s in sufs):
        return "dorsal/ventral and left/right not specified"
    if all(s.isdigit() for s in sufs):
        return "member number of a numbered series not specified"
    return f"class label with {len(candidates)} members"


def _classify(label: str, canonical: frozenset[str]) -> tuple[str, bool, tuple[str, ...], str] | None:
    """(kind, ambiguous, candidates, reason) for a non-canonical label, or None if no explicit rule applies."""
    if label in CONVENTIONS:
        cands, reason = CONVENTIONS[label]
        kept = tuple(c for c in cands if c in canonical)
        if kept:
            return "convention", True, kept, reason
    m = _ZERO_PADDED.match(label)
    if m and (target := m.group(1) + m.group(2)) in canonical:
        return "zero_padded", False, (target,), "leading zero in the series number"
    folded = [c for c in canonical if c.casefold() == label.casefold()]
    if len(folded) == 1:
        return "case_variant", False, (folded[0],), "differs only in letter case"
    stems = stem_candidates(label, canonical)
    if len(stems) >= 2:
        return "class_label", True, tuple(stems), _stem_reason(label, stems)
    return None


def derive_aliases(canonical: Iterable[str], observed: Mapping[str, Iterable[str]]) -> list[AliasRow]:
    """Alias rows for every observed non-canonical label that an explicit rule maps; others are left out."""
    canon = frozenset(canonical)
    rows: list[AliasRow] = []
    for label in sorted(observed):
        if label in canon:
            continue
        hit = _classify(label, canon)
        if hit is None:
            continue
        kind, ambiguous, cands, reason = hit
        sources = tuple(sorted(set(observed[label])))
        rows.extend(AliasRow(label, c, kind, ambiguous, len(cands), reason, sources) for c in cands)
    return rows


def class_rows(canonical: Iterable[str], classes: Mapping[str, Iterable[str]], source: str) -> list[AliasRow]:
    """Alias rows for source-declared cell classes (e.g. ``AVA`` -> AVAL, AVAR); single-member classes are skipped."""
    canon = frozenset(canonical)
    rows: list[AliasRow] = []
    for name in sorted(classes):
        members = sorted(set(classes[name]) & canon)
        if name in canon or len(members) < 2:
            continue
        reason = f"cell class of {len(members)} neurons ({source}); no member is implied"
        rows.extend(AliasRow(name, m, "class_label", True, len(members), reason, (f"class:{source}",)) for m in members)
    return rows


def merge_rows(*groups: Iterable[AliasRow]) -> list[AliasRow]:
    """Union by (alias, canonical id); observed_in is merged, the first kind and reason win."""
    merged: dict[tuple[str, str], AliasRow] = {}
    for group in groups:
        for r in group:
            key = (r.alias, r.canonical_neuron_id)
            if key in merged:
                old = merged[key]
                merged[key] = AliasRow(
                    old.alias,
                    old.canonical_neuron_id,
                    old.kind,
                    old.ambiguous,
                    old.n_candidates,
                    old.reason,
                    tuple(sorted(set(old.observed_in) | set(r.observed_in))),
                )
            else:
                merged[key] = r
    by_alias: dict[str, int] = {}
    for alias, _ in merged:
        by_alias[alias] = by_alias.get(alias, 0) + 1
    out = []
    for key in sorted(merged):
        r = merged[key]
        n = by_alias[r.alias]
        out.append(
            r
            if r.n_candidates == n
            else AliasRow(r.alias, r.canonical_neuron_id, r.kind, n > 1, n, r.reason, r.observed_in)
        )
    return out


class AliasTable:
    def __init__(self, canonical: Iterable[str], rows: Iterable[AliasRow]) -> None:
        self.canonical = frozenset(canonical)
        self.rows = list(rows)
        self._by_alias: dict[str, list[AliasRow]] = {}
        for r in self.rows:
            if r.alias in self.canonical:
                raise ValueError(f"alias {r.alias!r} is itself a canonical name")
            if r.canonical_neuron_id not in self.canonical:
                raise ValueError(f"alias {r.alias!r} points at unknown neuron {r.canonical_neuron_id!r}")
            self._by_alias.setdefault(r.alias, []).append(r)

    def resolve(self, label: str) -> Resolution:
        if label in self.canonical:
            return Resolution("exact", (label,), None, "canonical neuron name")
        rows = self._by_alias.get(label)
        if not rows:
            return Resolution("unresolved", (), None, "")
        cands = tuple(sorted(r.canonical_neuron_id for r in rows))
        first = rows[0]
        if len(cands) == 1 and not first.ambiguous:
            return Resolution("alias", cands, first.kind, first.reason)
        return Resolution("ambiguous", cands, first.kind, first.reason)


# --------------------------------------------------------------------------- audit helpers


def _within_one_edit(a: str, b: str) -> bool:
    """True if ``a`` and ``b`` differ by one substitution, insertion, deletion or adjacent transposition."""
    if a == b or abs(len(a) - len(b)) > 1:
        return False
    if len(a) == len(b):
        diffs = [i for i in range(len(a)) if a[i] != b[i]]
        if len(diffs) == 1:
            return True
        return (
            len(diffs) == 2 and diffs[1] == diffs[0] + 1 and a[diffs[0]] == b[diffs[1]] and a[diffs[1]] == b[diffs[0]]
        )
    short, long_ = (a, b) if len(a) < len(b) else (b, a)
    return any(long_[:i] + long_[i + 1 :] == short for i in range(len(long_)))


def suggestions(label: str, canonical: Iterable[str]) -> list[str]:
    """Canonical names one edit away from ``label``; labels shorter than 3 characters get none."""
    if len(label) < 3:
        return []
    return sorted(c for c in canonical if _within_one_edit(label, c))


def non_neuron_reason(label: str, cells: Mapping[str, str]) -> str | None:
    """Explain a label that names a known non-neuron cell (glia, socket, muscle...), else None."""
    folded = label.casefold()
    for n in sorted(cells):
        if n.casefold() == folded:
            note = "" if n == label else f" (case differs from {n})"
            return f"known non-neuron cell: {cells[n] or 'no description'}{note}"
    if len(label) < 3:
        return None
    stems = [
        n
        for n in sorted(cells)
        if len(n) > len(label) and n.casefold().startswith(folded) and _STEM_SUFFIX.match(n[len(label) :])
    ]
    if len(stems) >= 2:
        return f"class-level label of non-neuron cells ({', '.join(stems[:4])}): {cells[stems[0]] or 'no description'}"
    return None
