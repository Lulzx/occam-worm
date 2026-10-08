"""Typed edge extraction for the annotation build: reconstructions, endpoint mapping, kinds, sign (OW-013).

Each reconstruction is read from its own file and mapped into ``edges.parquet`` rows (spec §14.4) on its own;
reconstructions are never merged. Gap junctions are stored symmetrically: one directed row per direction with
the same count. Sign is ``unknown`` for chemical, neuromuscular and modulatory edges unless the dataset states
it for that edge (Fenyves 2020: a published prediction), and ``null`` (not applicable) for gap junctions.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Collection, Iterable, Mapping
from dataclasses import dataclass, field

from occamworm.importers import openworm as ow
from occamworm.importers import wormneuroatlas as wa
from occamworm.importers.aliases import AliasTable, non_neuron_reason

KIND_ORDER = {"chem": 0, "gap": 1, "nmj": 2, "putative_mod": 3}


class EdgeError(Exception):
    """An edge file contains something the importer does not know how to type."""


# --------------------------------------------------------------------------- reconstruction registry


@dataclass(frozen=True)
class Recon:
    id: str
    loader: str  # toolbox_cache | wna_csv | fenyves | bentley_ma | bentley_np | ripoll
    source: str  # registry source id
    member: str  # archive member the edges come from
    title: str
    life_stage: str | None
    weights: str  # what the native weight counts
    reader: str | None = None  # Toolbox cache reader name
    pair: str | None = None  # the same upstream dataset as bundled by the other source


_SYN = "synapse or junction count as reported by the reconstruction"
_COMBINED = "adult (N2U) and L4 (JSH) series combined"
_WITVLIET = "Witvliet et al. 2021, Nature 596:257-261 (WormWiring spreadsheets, "
_STAGES = {1: "L1", 2: "L1", 3: "L1", 4: "L1", 5: "L2", 6: "L3", 7: "adult", 8: "adult"}

RECONSTRUCTIONS: tuple[Recon, ...] = (
    Recon(
        "white1986-whole",
        "toolbox_cache",
        ow.SOURCE,
        ow.cache_member("White_whole"),
        "White et al. 1986, Phil Trans R Soc B 314:1-340 (whole-animal wiring, WormAtlas/Varshney compilation)",
        _COMBINED,
        _SYN,
        reader="White_whole",
        pair="white1986-whole:wna",
    ),
    Recon(
        "white1986-n2u",
        "toolbox_cache",
        ow.SOURCE,
        ow.cache_member("DurbinN2UDataReader"),
        "White et al. 1986 animal N2U as compiled in Durbin 1987 (neurodata.txt, updated)",
        "adult",
        _SYN,
        reader="DurbinN2UDataReader",
    ),
    Recon(
        "white1986-jsh",
        "toolbox_cache",
        ow.SOURCE,
        ow.cache_member("DurbinJSHDataReader"),
        "White et al. 1986 animal JSH as compiled in Durbin 1987 (neurodata.txt, updated)",
        "L4",
        _SYN,
        reader="DurbinJSHDataReader",
    ),
    Recon(
        "varshney2011",
        "toolbox_cache",
        ow.SOURCE,
        ow.cache_member("VarshneyDataReader"),
        "Varshney et al. 2011, PLoS Comput Biol 7:e1001066 (NeuronConnect.xls)",
        _COMBINED,
        _SYN,
        reader="VarshneyDataReader",
    ),
    Recon(
        "cook2019-herm",
        "toolbox_cache",
        ow.SOURCE,
        ow.cache_member("Cook2019HermReader"),
        "Cook et al. 2019, Nature 571:63-71 (hermaphrodite, WormWiring SI 5 corrected July 2020)",
        "adult",
        _SYN,
        reader="Cook2019HermReader",
    ),
    Recon(
        "cook2020-pharynx",
        "toolbox_cache",
        ow.SOURCE,
        ow.cache_member("Cook2020DataReader"),
        "Cook et al. 2020, J Comp Neurol (pharynx connectome)",
        "adult",
        _SYN,
        reader="Cook2020DataReader",
    ),
    *(
        Recon(
            f"witvliet2021-{i}",
            "toolbox_cache",
            ow.SOURCE,
            ow.cache_member(f"WitvlietDataReader{i}"),
            f"{_WITVLIET}OpenWorm Toolbox), dataset {i}",
            _STAGES[i],
            _SYN,
            reader=f"WitvlietDataReader{i}",
            pair=f"witvliet2021-{i}:wna" if i in wa.WITVLIET_CSV else None,
        )
        for i in range(1, 9)
    ),
    Recon(
        "white1986-whole:wna",
        "wna_csv",
        wa.SOURCE,
        wa.WHITE_WHOLE_CSV,
        "White et al. 1986 whole-animal wiring, aconnectome_white_1986_whole.csv (Worm Neuro Atlas bundle)",
        _COMBINED,
        _SYN,
        pair="white1986-whole",
    ),
    *(
        Recon(
            f"witvliet2021-{i}:wna",
            "wna_csv",
            wa.SOURCE,
            wa.WITVLIET_CSV[i],
            f"{_WITVLIET}Worm Neuro Atlas bundle), dataset {i}",
            _STAGES[i],
            _SYN,
            pair=f"witvliet2021-{i}",
        )
        for i in sorted(wa.WITVLIET_CSV)
    ),
    Recon(
        "fenyves2020-sign-prediction",
        "fenyves",
        wa.SOURCE,
        wa.FENYVES,
        "Fenyves et al. 2020, PLoS Comput Biol 16:e1007974, S3 sheet '5. Sign prediction' (WormWiring chemical "
        "edges with sign predicted from transmitter and receptor expression)",
        None,
        "chemical synapse count",
    ),
    Recon(
        "bentley2016-monoamine",
        "bentley_ma",
        ow.SOURCE,
        ow.BENTLEY_MA,
        "Bentley et al. 2016, PLoS Comput Biol 12:e1005283 (monoamine signalling, extrasynaptic)",
        None,
        "number of distinct transmitter-receptor pairs",
    ),
    Recon(
        "bentley2016-neuropeptide",
        "bentley_np",
        ow.SOURCE,
        ow.BENTLEY_NP,
        "Bentley et al. 2016, PLoS Comput Biol 12:e1005283 (neuropeptide signalling, extrasynaptic)",
        None,
        "number of distinct peptide-receptor pairs",
    ),
    *(
        Recon(
            f"ripoll-sanchez2023-{rng}",
            "ripoll",
            ow.SOURCE,
            ow.RIPOLL_MATRICES[rng],
            f"Ripoll-Sanchez et al. 2023, Neuron 111:3570-3589 ({rng}-range neuropeptide model)",
            "adult",
            "number of NPP-GPCR pathways",
        )
        for rng in ("short", "mid", "long")
    ),
)


def members_needed(recons: Collection[Recon], source: str) -> set[str]:
    return {r.member for r in recons if r.source == source}


# --------------------------------------------------------------------------- muscle aliases

MUSCLE_MAPPING_VERSION = "ow013-muscle-aliases-v1"
_BWM = re.compile(r"^BWM-([DV])([LR])(\d\d)$")
# label -> (canonical muscle, reason)
MUSCLE_CONVENTIONS: dict[str, tuple[str, str]] = {
    "LegacyBodyWallMuscles": (
        "BWM",
        "White 1986 does not resolve individual body-wall muscles; the Toolbox names this group BWM",
    ),
    "pm4": (
        "pm4_UNSPECIFIED",
        "pharyngeal muscle pm4 without a side; the Toolbox names this group pm4_UNSPECIFIED",
    ),
}


@dataclass(frozen=True)
class MuscleAlias:
    alias: str
    canonical_muscle_id: str
    kind: str
    reason: str
    observed_in: tuple[str, ...]


def derive_muscle_aliases(muscles: Collection[str], observed: Mapping[str, Iterable[str]]) -> list[MuscleAlias]:
    """Aliases of muscle labels seen in the sources: ``BWM-DL01`` -> ``MDL01`` (Witvliet naming) and the
    explicit conventions above. Only targets that are Toolbox muscle names are kept."""
    out: list[MuscleAlias] = []
    for label in sorted(observed):
        if label in muscles:
            continue
        m = _BWM.match(label)
        if m and (target := f"M{m.group(1)}{m.group(2)}{m.group(3)}") in muscles:
            reason = "body-wall muscle named by quadrant and number (WormWiring Witvliet naming)"
            out.append(MuscleAlias(label, target, "bwm_prefix", reason, tuple(sorted(set(observed[label])))))
        elif label in MUSCLE_CONVENTIONS and MUSCLE_CONVENTIONS[label][0] in muscles:
            target, reason = MUSCLE_CONVENTIONS[label]
            out.append(MuscleAlias(label, target, "convention", reason, tuple(sorted(set(observed[label])))))
    return out


# --------------------------------------------------------------------------- extraction


@dataclass
class RawEdge:
    pre: str
    post: str
    layer: str  # chem | gap | mod
    weight: float
    synclass: str | None = None
    transmitters: tuple[str, ...] = ()
    receptors: tuple[str, ...] = ()
    sign_raw: str | None = None


_LAYER_OF_CLASS = {ow.CHEMICAL_CLASS: "chem", ow.GAP_CLASS: "gap"}
_LAYER_OF_TYPE = {"chemical": "chem", "electrical": "gap"}


def _aggregate_pairs(rows: list[tuple[str, str, str, str]]) -> list[RawEdge]:
    """One modulatory edge per (pre, post); its weight is the number of distinct transmitter-receptor pairs."""
    pairs: dict[tuple[str, str], set[tuple[str, str]]] = defaultdict(set)
    for pre, post, transmitter, receptor in rows:
        pairs[(pre, post)].add((transmitter, receptor))
    return [
        RawEdge(
            pre,
            post,
            "mod",
            float(len(p)),
            transmitters=tuple(sorted({t for t, _ in p})),
            receptors=tuple(sorted({r for _, r in p})),
        )
        for (pre, post), p in sorted(pairs.items())
    ]


def extract(recon: Recon, members: Mapping[str, bytes]) -> list[RawEdge]:
    data = members[recon.member]
    out: list[RawEdge] = []
    if recon.loader == "toolbox_cache":
        cache = ow.parse_cache(data)
        for synclass in sorted(cache.connections):
            layer = _LAYER_OF_CLASS.get(synclass)
            if layer is None:
                raise EdgeError(f"{recon.id}: unexpected synapse class {synclass!r}")
            out.extend(RawEdge(a, b, layer, w, synclass) for a, b, w in ow.cache_edges(cache, synclass))
    elif recon.loader == "wna_csv":
        for pre, post, typ, w in wa.read_aconnectome_csv(data):
            layer = _LAYER_OF_TYPE.get(typ)
            if layer is None:
                raise EdgeError(f"{recon.id}: unexpected edge type {typ!r}")
            out.append(RawEdge(pre, post, layer, w, typ))
    elif recon.loader == "fenyves":
        for fe in wa.read_fenyves(data).edges:
            if fe.edge_type != "chemical":
                raise EdgeError(f"{recon.id}: unexpected edge type {fe.edge_type!r}")
            out.append(RawEdge(fe.pre, fe.post, "chem", fe.weight, fe.edge_type, sign_raw=fe.polarity))
    elif recon.loader in ("bentley_ma", "bentley_np"):
        out = _aggregate_pairs(ow.read_edgelist(data))
    elif recon.loader == "ripoll":
        out = [RawEdge(a, b, "mod", w) for a, b, w in ow.read_matrix_csv(data)]
    else:
        raise EdgeError(f"{recon.id}: unknown loader {recon.loader!r}")
    return out


# --------------------------------------------------------------------------- mapping


@dataclass(frozen=True)
class EdgeRow:
    source: str
    target: str
    kind: str
    synapse_count: float | None
    sign: str | None
    reconstruction: str
    life_stage: str | None
    provenance: str
    native_weight: float
    weight_semantics: str
    synclass: str | None
    transmitters: tuple[str, ...]
    receptors: tuple[str, ...]
    n_raw_rows: int
    raw_source: str
    raw_target: str
    symmetrised: bool
    sign_raw: str | None
    sign_basis: str
    target_kind: str  # neuron | muscle


@dataclass
class _Acc:
    weight: float = 0.0
    n: int = 0
    transmitters: set[str] = field(default_factory=set)
    receptors: set[str] = field(default_factory=set)
    signs: set[str] = field(default_factory=set)
    raw_pre: str = ""
    raw_post: str = ""
    synclass: str | None = None


_FENYVES_SIGN = {"+": "excitatory", "-": "inhibitory"}


def sign_for(recon: Recon, kind: str, sign_raws: Collection[str]) -> tuple[str | None, str]:
    """(sign, basis). Only an explicit per-edge value in the dataset can produce a sign other than unknown."""
    if kind == "gap":
        return None, "not_applicable"
    if recon.loader == "fenyves" and kind == "chem":
        mapped = {_FENYVES_SIGN.get(s, "unknown") for s in sign_raws}
        return (mapped.pop() if len(mapped) == 1 else "unknown"), "fenyves2020_nt_r_prediction"
    return "unknown", "not_supplied"


def map_recon(
    recon: Recon,
    raw: list[RawEdge],
    table: AliasTable,
    muscles: Collection[str],
    muscle_aliases: Mapping[str, str],
    non_neuron: Mapping[str, str],
) -> tuple[list[EdgeRow], Counter[str], Counter[str]]:
    """Map raw labels to canonical IDs, type the edges and count what is dropped (by reason and by label)."""
    dropped: Counter[str] = Counter()
    dropped_labels: Counter[str] = Counter()

    def endpoint(label: str) -> tuple[str, str | None, str]:
        res = table.resolve(label)
        if res.status in ("exact", "alias"):
            return "neuron", res.candidates[0], ""
        if res.status == "ambiguous":
            return "drop", None, "ambiguous_label"
        if label in muscles:
            return "muscle", label, ""
        if label in muscle_aliases:
            return "muscle", muscle_aliases[label], ""
        return "drop", None, "non_neuron_cell" if non_neuron_reason(label, non_neuron) else "unknown_label"

    acc: dict[tuple[str, str, str, str], _Acc] = {}
    for e in raw:
        sk, sid, sreason = endpoint(e.pre)
        tk, tid, treason = endpoint(e.post)
        if sid is None or tid is None:
            dropped[f"{e.layer}:{sreason or treason}"] += 1
            dropped_labels[e.pre if sid is None else e.post] += 1
            continue
        if e.layer == "mod":
            if sk != "neuron" or tk != "neuron":
                dropped[f"mod:{sk}_to_{tk}"] += 1
                continue
            kind = "putative_mod"
        elif sk == "neuron" and tk == "neuron":
            kind = e.layer
        elif sk == "neuron" and tk == "muscle":
            kind = "nmj"
        elif sk == "muscle" and tk == "neuron" and e.layer == "gap":
            dropped["gap:mirror_of_neuron_to_muscle"] += 1
            continue
        else:
            dropped[f"{e.layer}:{sk}_to_{tk}"] += 1
            continue
        a = acc.setdefault((sid, tid, kind, tk), _Acc(raw_pre=e.pre, raw_post=e.post, synclass=e.synclass))
        a.weight += e.weight
        a.n += 1
        a.transmitters.update(e.transmitters)
        a.receptors.update(e.receptors)
        if e.sign_raw is not None:
            a.signs.add(e.sign_raw)

    provenance = f"{recon.title} | {recon.source}:{recon.member}"

    def make(sid: str, tid: str, kind: str, tk: str, a: _Acc, *, symmetrised: bool) -> EdgeRow:
        sign, basis = sign_for(recon, kind, a.signs)
        return EdgeRow(
            sid,
            tid,
            kind,
            None if kind == "putative_mod" else a.weight,
            sign,
            recon.id,
            recon.life_stage,
            provenance,
            a.weight,
            recon.weights,
            a.synclass,
            tuple(sorted(a.transmitters)),
            tuple(sorted(a.receptors)),
            0 if symmetrised else a.n,
            a.raw_post if symmetrised else a.raw_pre,
            a.raw_pre if symmetrised else a.raw_post,
            symmetrised,
            ",".join(sorted(a.signs)) or None,
            basis,
            tk,
        )

    rows = [make(sid, tid, kind, tk, a, symmetrised=False) for (sid, tid, kind, tk), a in acc.items()]
    gap_pairs = {(sid, tid) for (sid, tid, kind, _tk) in acc if kind == "gap"}
    for (sid, tid, kind, tk), a in acc.items():
        if kind == "gap" and (tid, sid) not in gap_pairs:
            rows.append(make(tid, sid, kind, tk, a, symmetrised=True))
    rows.sort(key=lambda r: (KIND_ORDER[r.kind], r.source, r.target))
    return rows, dropped, dropped_labels


# --------------------------------------------------------------------------- comparison


def compare(a: list[EdgeRow], b: list[EdgeRow], kinds: tuple[str, ...] = ("chem", "gap", "nmj")) -> dict[str, int]:
    """Edge-set agreement between two reconstructions of the same kind (neuron targets and muscles alike)."""
    out = {"both": 0, "equal_weight": 0, "only_a": 0, "only_b": 0}
    wa_ = {(r.source, r.target, r.kind): r.native_weight for r in a if r.kind in kinds}
    wb_ = {(r.source, r.target, r.kind): r.native_weight for r in b if r.kind in kinds}
    for k, w in wa_.items():
        if k in wb_:
            out["both"] += 1
            out["equal_weight"] += int(w == wb_[k])
        else:
            out["only_a"] += 1
    out["only_b"] = sum(1 for k in wb_ if k not in wa_)
    return out
