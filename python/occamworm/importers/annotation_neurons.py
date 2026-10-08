"""Neuron annotation evidence for the annotation build: vocabularies, evidence rows, conflicts (OW-013).

Every attribute value of a neuron is kept as one evidence row with its source; nothing is chosen between
sources. :func:`detect_conflicts` turns disagreements into explicit conflict records.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, field

from occamworm.importers import openworm as ow
from occamworm.importers import wormneuroatlas as wa
from occamworm.importers.aliases import AliasTable

# --------------------------------------------------------------------------- vocabularies

# WormAtlas cell types (all_cell_info.csv) of hermaphrodite neurons -> functional type; a note that White 1986
# classified a cell differently stays in the raw value.
WORMATLAS_TYPES: dict[str, str] = {
    "Ventral cord motor neuron": "motor",
    "Layer 3 interneuron": "interneuron",
    "Layer 2 interneuron": "interneuron",
    "Layer 1 interneuron": "interneuron",
    "Layer 1 interneuron; motorneuron in White et al., 1986": "interneuron",
    "Category 4 interneuron": "interneuron",
    "Pharyngeal interneuron": "interneuron",
    "Linker to pharynx": "interneuron",
    "Cephalic": "sensory",
    "Amphid": "sensory",
    "Amphid, nociceptive": "sensory",
    "Mechanosensory": "sensory",
    "Touch": "sensory",
    "Phasmid": "sensory",
    "O2, CO2, social signals, touch": "sensory",
    "Head motor neuron": "motor",
    "Sublateral motor neuron": "motor",
    "Sublateral motor neuron; interneuron in White et al., 1986": "motor",
    "Hermaphrodite specific motor neuron": "motor",
    "Pharyngeal motor neuron": "motor",
    "Pharyngeal polymodal neuron": "polymodal",
    "Canal neuron": "unknown",
}
_FUNCTION_WORDS = {
    "motor neuron": "motor",
    "interneuron": "interneuron",
    "sensory neuron": "sensory",
    "unknown": "unknown",
    "pharyngeal motor neuron": "motor",
    "pharyngeal interneuron": "interneuron",
}
CLASSICAL = frozenset({"ACh", "Glu", "GABA"})
MONOAMINES = frozenset({"dopamine", "serotonin", "octopamine", "tyramine"})
_TRANSMITTER_WORDS = {
    "ach": "ACh",
    "glu": "Glu",
    "gaba": "GABA",
    "da": "dopamine",
    "dopamine": "dopamine",
    "5-ht": "serotonin",
    "serotonin": "serotonin",
    "octopamine": "octopamine",
    "tyramine": "tyramine",
    "betaine": "betaine",
}


def normalize_transmitter(raw: str) -> list[str]:
    """Transmitter tokens of a Wang 2024 entry: ``*Glu - NEW`` -> Glu, ``GABA (uptake)`` -> GABA_uptake."""
    out: list[str] = []
    for part in re.split(r",(?![^()]*\))", raw):  # commas inside parentheses stay
        s = re.sub(r"\s*-\s*NEW\b", "", part, flags=re.IGNORECASE)
        s = re.sub(r"\(new\)", "", s, flags=re.IGNORECASE).replace("*", "").strip().lower()
        if not s:
            continue
        if s.startswith("unknown"):
            out.append("unknown_monoamine" if "amine" in s else "unknown_orphan")
        elif "uptake" in s:
            base = re.sub(r"\(.*?\)", "", s).strip()
            out.append(f"{_TRANSMITTER_WORDS.get(base, base)}_uptake")
        else:
            out.append(_TRANSMITTER_WORDS.get(s, f"other:{s}"))
    return out


def function_types(raw: str) -> list[str]:
    """Functional types of a comma-separated type string (Fenyves, Ripoll-Sanchez), lower-cased."""
    out = []
    for part in raw.split(","):
        key = part.strip().lower()
        if key in _FUNCTION_WORDS:
            out.append(_FUNCTION_WORDS[key])
        elif key:
            out.append(f"other:{key}")
    return out


# --------------------------------------------------------------------------- evidence


@dataclass(frozen=True)
class Evidence:
    neuron_id: str
    attribute: str
    value: str
    role: str
    source: str
    member: str
    raw_value: str | None
    raw_label: str
    via_alias: bool
    ambiguous_assignment: bool
    note: str | None


@dataclass
class EvidenceBook:
    table: AliasTable
    rows: list[Evidence] = field(default_factory=list)
    unmapped: dict[str, Counter[str]] = field(default_factory=lambda: defaultdict(Counter))

    def add(
        self,
        label: str,
        attribute: str,
        value: str,
        *,
        role: str,
        source: str,
        member: str,
        raw_value: str | None = None,
        note: str | None = None,
    ) -> None:
        res = self.table.resolve(label)
        if not res.candidates:
            self.unmapped[source][label] += 1
            return
        for nid in res.candidates:
            self.rows.append(
                Evidence(
                    nid,
                    attribute,
                    value,
                    role,
                    source,
                    member,
                    raw_value,
                    label,
                    res.status != "exact",
                    res.status == "ambiguous",
                    note,
                )
            )


@dataclass(frozen=True)
class Conflict:
    entity_type: str
    entity_id: str
    attribute: str
    kind: str
    severity: str  # hard: sources disagree; soft: partial overlap or a single source lists several values
    values: tuple[str, ...]
    sources: tuple[str, ...]  # 'value@source' for every piece of evidence
    raw_values: tuple[str, ...]
    note: str


def _by_source(rows: list[Evidence], *, roles: tuple[str, ...] | None = None) -> dict[str, set[str]]:
    out: dict[str, set[str]] = defaultdict(set)
    for r in rows:
        if roles is None or r.role in roles:
            out[r.source].add(r.value)
    return out


def _conflict(neuron_id: str, attribute: str, kind: str, severity: str, rows: list[Evidence], note: str) -> Conflict:
    return Conflict(
        "neuron",
        neuron_id,
        attribute,
        kind,
        severity,
        tuple(sorted({r.value for r in rows})),
        tuple(sorted({f"{r.value}@{r.source}" for r in rows})),
        tuple(sorted({f"{r.raw_value}@{r.source}" for r in rows if r.raw_value is not None})),
        note,
    )


def _single_valued(neuron_id: str, attribute: str, rows: list[Evidence]) -> list[Conflict]:
    values = {r.value for r in rows}
    if len(values) < 2:
        return []
    across = any(a.source != b.source and a.value != b.value for a in rows for b in rows)
    if across:
        return [_conflict(neuron_id, attribute, "values_differ_across_sources", "hard", rows, "")]
    return [_conflict(neuron_id, attribute, "several_values_in_one_source", "soft", rows, "")]


def _type_conflicts(neuron_id: str, rows: list[Evidence]) -> list[Conflict]:
    sets = {s: v - {"unknown"} for s, v in _by_source(rows).items()}
    sets = {s: v for s, v in sets.items() if v}
    names = sorted(sets)
    disjoint = differ = False
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            if sets[a] == sets[b]:
                continue
            wildcard = "polymodal" in sets[a] or "polymodal" in sets[b]
            if wildcard or sets[a] & sets[b]:
                differ = True
            else:
                disjoint = True
    if disjoint:
        return [_conflict(neuron_id, "neuron_type", "type_sets_disjoint", "hard", rows, "")]
    if differ:
        return [
            _conflict(
                neuron_id,
                "neuron_type",
                "type_sets_overlap_but_differ",
                "soft",
                rows,
                "polymodal counts as overlapping with any type",
            )
        ]
    return []


def _transmitter_conflicts(neuron_id: str, rows: list[Evidence]) -> list[Conflict]:
    out: list[Conflict] = []
    main = {s: v & CLASSICAL for s, v in _by_source(rows, roles=("dominant", "reported")).items()}
    alt = {s: v & CLASSICAL for s, v in _by_source(rows, roles=("alternative",)).items()}
    main = {s: v for s, v in main.items() if v}
    names = sorted(main)
    verdict: tuple[str, str] | None = None
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            if main[a] == main[b]:
                continue
            if main[a] & main[b]:
                cand = ("classical_sets_overlap_but_differ", "soft")
            elif main[b] <= alt.get(a, set()) or main[a] <= alt.get(b, set()):
                cand = ("dominant_matches_alternative_only", "soft")
            else:
                cand = ("classical_transmitter_differs", "hard")
            if verdict is None or cand[1] == "hard":
                verdict = cand
    if verdict:
        out.append(_conflict(neuron_id, "transmitter", verdict[0], verdict[1], rows, "classical: ACh, Glu, GABA"))
    by_src = _by_source(rows, roles=("dominant", "reported", "alternative", "monoamine_source"))
    if "wang2024-nt-atlas" in by_src:  # Bentley lists every monoamine source, so its absence is informative
        wang_m = by_src["wang2024-nt-atlas"] & MONOAMINES
        bentley_m = by_src.get("bentley2016-monoamine", set()) & MONOAMINES
        if wang_m != bentley_m:
            disjoint = bool(wang_m and bentley_m and not wang_m & bentley_m)
            kind = "monoamine_sets_disjoint" if disjoint else "monoamine_reported_by_one_source_only"
            out.append(
                _conflict(neuron_id, "transmitter", kind, "hard" if disjoint else "soft", rows, "monoamines compared")
            )
    return out


def detect_conflicts(evidence: list[Evidence]) -> list[Conflict]:
    grouped: dict[tuple[str, str], list[Evidence]] = defaultdict(list)
    for e in evidence:
        grouped[(e.neuron_id, e.attribute)].append(e)
    out: list[Conflict] = []
    for (nid, attribute), rows in sorted(grouped.items()):
        if attribute == "neuron_type":
            out.extend(_type_conflicts(nid, rows))
        elif attribute == "transmitter":
            out.extend(_transmitter_conflicts(nid, rows))
        elif attribute != "ionotropic_receptor_gene":
            out.extend(_single_valued(nid, attribute, rows))
    return out


# --------------------------------------------------------------------------- neuron attribute gathering


def gather_neuron_evidence(
    book: EvidenceBook,
    wn: Mapping[str, bytes],
    lists: dict[str, list[str]],
    fenyves: wa.Fenyves,
    cell_info: dict[str, ow.CellInfoRow],
    wang: list[ow.WangRow],
    ripoll: list[ow.RipollNeuronRow],
    bentley_ma: list[tuple[str, str, str, str]],
) -> None:
    canon = sorted(book.table.canonical)
    cook_src, cook_member = "cook2019-categories", ow.CELLS_PY
    cook_types = (
        ("sensory", lists["SENSORY_NEURONS_COOK"]),
        ("interneuron", lists["INTERNEURONS_COOK"]),
        ("motor", lists["MOTORNEURONS_COOK"]),
        ("polymodal", lists["PHARYNGEAL_POLYMODAL_NEURONS"]),
        ("unknown", lists["UNKNOWN_FUNCTION_NEURONS"]),
    )
    pharyngeal = set(lists["PHARYNGEAL_NEURONS"])
    for value, names in cook_types:
        for n in names:
            book.add(n, "neuron_type", value, role="reported", source=cook_src, member=cook_member, raw_value=value)
    for n in canon:
        book.add(n, "is_pharyngeal", str(n in pharyngeal).lower(), role="reported", source=cook_src, member=cook_member)

    wa_src, wa_member = "wormatlas-cellinfo", ow.CELL_INFO
    for n in canon:
        row = cell_info.get(n)
        if row is None:
            continue
        mapped = WORMATLAS_TYPES.get(row.cell_type)
        if mapped is None:
            mapped = f"other:{row.cell_type}"
        book.add(n, "neuron_type", mapped, role="reported", source=wa_src, member=wa_member, raw_value=row.cell_type)
        book.add(
            n,
            "is_pharyngeal",
            str(row.cell_type.startswith("Pharyngeal")).lower(),
            role="reported",
            source=wa_src,
            member=wa_member,
            raw_value=row.cell_type,
        )

    w_src, w_member = "wang2024-nt-atlas", ow.WANG_NT
    for wr in wang:
        if wr.cell_class:
            book.add(
                wr.neuron,
                "cell_class",
                wr.cell_class,
                role="reported",
                source=w_src,
                member=w_member,
                note="class cell is merged upstream; value filled down from the group's first row"
                if wr.class_filled_down
                else None,
            )
        if wr.neurotransmitter_raw:
            for v in normalize_transmitter(wr.neurotransmitter_raw):
                book.add(
                    wr.neuron,
                    "transmitter",
                    v,
                    role="reported",
                    source=w_src,
                    member=w_member,
                    raw_value=wr.neurotransmitter_raw,
                    note=wr.comments,
                )

    r_src, r_member = "ripoll2023-table-s7", ow.RIPOLL_NEURONS
    for rr in ripoll:
        if rr.cell_type:
            if rr.cell_type == "Pharynx":
                book.add(
                    rr.neuron,
                    "is_pharyngeal",
                    "true",
                    role="reported",
                    source=r_src,
                    member=r_member,
                    raw_value="Pharynx",
                )
            else:
                book.add(
                    rr.neuron,
                    "is_pharyngeal",
                    "false",
                    role="reported",
                    source=r_src,
                    member=r_member,
                    raw_value=rr.cell_type,
                )
                for v in function_types(rr.cell_type):
                    book.add(
                        rr.neuron,
                        "neuron_type",
                        v,
                        role="reported",
                        source=r_src,
                        member=r_member,
                        raw_value=rr.cell_type,
                    )
        if rr.segment:
            book.add(rr.neuron, "body_segment", rr.segment, role="reported", source=r_src, member=r_member)

    f_src, f_member = "fenyves2020-s3", wa.FENYVES
    for nt in fenyves.transmitters:
        for role, raw in (("dominant", nt.dominant), ("alternative", nt.alternative)):
            if raw:
                for v in normalize_transmitter(raw):
                    book.add(nt.neuron, "transmitter", v, role=role, source=f_src, member=f_member, raw_value=raw)
    for neuron, raw_type in fenyves.types.items():
        book.add(
            neuron,
            "is_pharyngeal",
            str("pharyngeal" in raw_type.lower()).lower(),
            role="reported",
            source=f_src,
            member=f_member,
            raw_value=raw_type,
        )
        for v in function_types(raw_type):
            book.add(neuron, "neuron_type", v, role="reported", source=f_src, member=f_member, raw_value=raw_type)
    for neuron, genes in fenyves.receptors.items():
        for g in genes:
            book.add(neuron, "ionotropic_receptor_gene", g, role="expressed", source=f_src, member=f_member)

    ganglia = wa.read_ganglia(wn[wa.GANGLIA])
    g_src, g_member = "wna-ganglia", wa.GANGLIA
    for group, names in sorted(ganglia.groups.items()):
        for n in names:
            book.add(n, "region", group, role="reported", source=g_src, member=g_member)
            book.add(
                n,
                "is_pharyngeal",
                str(group in ganglia.pharyngeal).lower(),
                role="reported",
                source=g_src,
                member=g_member,
                raw_value=group,
            )

    p_src, p_member = "wna-positions", wa.POSITIONS
    for pos in wa.read_positions(wn[wa.POSITIONS]):
        book.add(
            pos.name,
            "position_xyz",
            f"{pos.x!r},{pos.y!r},{pos.z!r}",
            role="reported",
            source=p_src,
            member=p_member,
            note="coordinate origin and units are not documented upstream",
        )

    b_src, b_member = "bentley2016-monoamine", ow.BENTLEY_MA
    seen: set[tuple[str, str]] = set()
    for pre, _post, transmitter, _receptor in bentley_ma:
        v = _TRANSMITTER_WORDS.get(transmitter.strip().lower(), f"other:{transmitter.lower()}")
        if (pre, v) not in seen:
            seen.add((pre, v))
            book.add(
                pre,
                "transmitter",
                v,
                role="monoamine_source",
                source=b_src,
                member=b_member,
                raw_value=transmitter,
                note="presynaptic cell of a monoamine-receptor pair in the Bentley edge list",
            )
