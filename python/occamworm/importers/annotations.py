"""Biological annotation mapping (ticket OW-013): canonical neurons, aliases, typed edges, conflicts, audit.

Inputs: the OpenWorm ConnectomeToolbox and Worm Neuro Atlas tarballs (both verified against the source
registry by the CLI) and the Randi 2023 ``neurons.parquet`` (atlas ROI labels, from OW-002). Output:
``data/normalized/annotations-v1/``. See docs/data/ANNOTATIONS.md for the rules.

Principles (spec §3.2):

* the neuron set is the 302 hermaphrodite neurons of the Toolbox; every other label resolves through the
  explicit alias table (:mod:`occamworm.importers.aliases`) or stays unresolved with a reason;
* each reconstruction is kept as its own set of edge rows; none is merged or preferred;
* synaptic sign is ``unknown`` unless a dataset states it for that edge (only Fenyves 2020, as a published
  prediction); it is never derived from the presynaptic cell name;
* when sources disagree on a neuron attribute or on an edge weight, every value is kept with its source and a
  row in ``conflicts.parquet`` records the disagreement.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from collections.abc import Collection, Iterable, Sequence
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from occamworm import __version__
from occamworm.importers import openworm as ow
from occamworm.importers import wormneuroatlas as wa
from occamworm.importers._archive import read_members
from occamworm.importers.aliases import (
    MAPPING_VERSION,
    AliasRow,
    AliasTable,
    class_rows,
    derive_aliases,
    merge_rows,
    non_neuron_reason,
    suggestions,
)
from occamworm.importers.annotation_edges import (
    MUSCLE_MAPPING_VERSION,
    RECONSTRUCTIONS,
    EdgeRow,
    MuscleAlias,
    Recon,
    compare,
    derive_muscle_aliases,
    extract,
    map_recon,
    members_needed,
)
from occamworm.importers.annotation_neurons import (
    Conflict,
    Evidence,
    EvidenceBook,
    detect_conflicts,
    gather_neuron_evidence,
)
from occamworm.importers.randi2023 import _file_sha, _git_revision

SCHEMA_VERSION = "0.1.0"
OUTPUT_NAME = "annotations-v1"
ATLAS_NAME = "randi2023-wt-v1"
WIDE_ATTRIBUTES = (
    "cell_class",
    "region",
    "body_segment",
    "neuron_type",
    "is_pharyngeal",
    "transmitter",
    "position_xyz",
)
ATLAS_LABEL_STATUSES = ("name", "name_duplicated")
UNKNOWN_TRANSMITTERS = frozenset({"unknown_orphan", "unknown_monoamine"})


class AnnotationError(Exception):
    """The inputs violate an expectation; the build stops rather than guessing."""


_STR = pa.string()
_LIST = pa.list_(pa.string())


def _write(table: pa.Table, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path, compression="zstd")


# --------------------------------------------------------------------------- tables


def aliases_table(rows: Iterable[AliasRow]) -> pa.Table:
    rows = list(rows)
    return pa.table(
        {
            "alias": pa.array([r.alias for r in rows], _STR),
            "canonical_neuron_id": pa.array([r.canonical_neuron_id for r in rows], _STR),
            "kind": pa.array([r.kind for r in rows], _STR),
            "ambiguous": pa.array([r.ambiguous for r in rows], pa.bool_()),
            "n_candidates": pa.array([r.n_candidates for r in rows], pa.int32()),
            "reason": pa.array([r.reason for r in rows], _STR),
            "observed_in": pa.array([list(r.observed_in) for r in rows], _LIST),
            "mapping_version": pa.array([MAPPING_VERSION] * len(rows), _STR),
        }
    )


def muscle_aliases_table(rows: Sequence[MuscleAlias]) -> pa.Table:
    return pa.table(
        {
            "alias": pa.array([r.alias for r in rows], _STR),
            "canonical_muscle_id": pa.array([r.canonical_muscle_id for r in rows], _STR),
            "kind": pa.array([r.kind for r in rows], _STR),
            "reason": pa.array([r.reason for r in rows], _STR),
            "observed_in": pa.array([list(r.observed_in) for r in rows], _LIST),
            "mapping_version": pa.array([MUSCLE_MAPPING_VERSION] * len(rows), _STR),
        }
    )


def attributes_table(evidence: Sequence[Evidence]) -> pa.Table:
    rows = sorted(evidence, key=lambda e: (e.neuron_id, e.attribute, e.source, e.role, e.value))
    return pa.table(
        {
            "neuron_id": pa.array([e.neuron_id for e in rows], _STR),
            "attribute": pa.array([e.attribute for e in rows], _STR),
            "value": pa.array([e.value for e in rows], _STR),
            "role": pa.array([e.role for e in rows], _STR),
            "source": pa.array([e.source for e in rows], _STR),
            "source_member": pa.array([e.member for e in rows], _STR),
            "raw_value": pa.array([e.raw_value for e in rows], _STR),
            "raw_label": pa.array([e.raw_label for e in rows], _STR),
            "via_alias": pa.array([e.via_alias for e in rows], pa.bool_()),
            "ambiguous_assignment": pa.array([e.ambiguous_assignment for e in rows], pa.bool_()),
            "note": pa.array([e.note for e in rows], _STR),
        }
    )


def conflicts_table(conflicts: Sequence[Conflict]) -> pa.Table:
    rows = sorted(conflicts, key=lambda c: (c.entity_type, c.entity_id, c.attribute, c.kind))
    return pa.table(
        {
            "conflict_id": pa.array([f"{c.entity_type}:{c.entity_id}:{c.attribute}:{c.kind}" for c in rows], _STR),
            "entity_type": pa.array([c.entity_type for c in rows], _STR),
            "entity_id": pa.array([c.entity_id for c in rows], _STR),
            "attribute": pa.array([c.attribute for c in rows], _STR),
            "conflict_kind": pa.array([c.kind for c in rows], _STR),
            "severity": pa.array([c.severity for c in rows], _STR),
            "values": pa.array([list(c.values) for c in rows], _LIST),
            "sources": pa.array([list(c.sources) for c in rows], _LIST),
            "raw_values": pa.array([list(c.raw_values) for c in rows], _LIST),
            "note": pa.array([c.note for c in rows], _STR),
        }
    )


def edges_table(rows: Sequence[EdgeRow]) -> pa.Table:
    """Exactly the §14.4 ``edges.parquet`` columns."""
    return pa.table(
        {
            "source_neuron_id": pa.array([r.source for r in rows], _STR),
            "target_neuron_id": pa.array([r.target for r in rows], _STR),
            "edge_kind": pa.array([r.kind for r in rows], _STR),
            "synapse_count": pa.array([r.synapse_count for r in rows], pa.float32()),
            "sign": pa.array([r.sign for r in rows], _STR),
            "confidence": pa.array([None] * len(rows), pa.float32()),
            "source_reconstruction": pa.array([r.reconstruction for r in rows], _STR),
            "life_stage": pa.array([r.life_stage for r in rows], _STR),
            "provenance_reference": pa.array([r.provenance for r in rows], _STR),
        }
    )


def edge_details_table(rows: Sequence[EdgeRow]) -> pa.Table:
    """Extra columns for ``edges.parquet``; row ``i`` here describes row ``i`` there."""
    return pa.table(
        {
            "edge_row": pa.array(range(len(rows)), pa.int64()),
            "native_weight": pa.array([r.native_weight for r in rows], pa.float64()),
            "weight_semantics": pa.array([r.weight_semantics for r in rows], _STR),
            "synclass": pa.array([r.synclass for r in rows], _STR),
            "transmitters": pa.array([list(r.transmitters) for r in rows], _LIST),
            "receptors": pa.array([list(r.receptors) for r in rows], _LIST),
            "n_raw_rows": pa.array([r.n_raw_rows for r in rows], pa.int32()),
            "raw_source_label": pa.array([r.raw_source for r in rows], _STR),
            "raw_target_label": pa.array([r.raw_target for r in rows], _STR),
            "symmetrised": pa.array([r.symmetrised for r in rows], pa.bool_()),
            "sign_raw": pa.array([r.sign_raw for r in rows], _STR),
            "sign_basis": pa.array([r.sign_basis for r in rows], _STR),
            "target_kind": pa.array([r.target_kind for r in rows], _STR),
        }
    )


# --------------------------------------------------------------------------- edge conflicts


def edge_conflicts(by_recon: dict[str, list[EdgeRow]], recons: Sequence[Recon]) -> list[Conflict]:
    """Disagreements between two bundles of the same upstream dataset (``X`` and ``X:wna``)."""
    out: list[Conflict] = []
    for r in recons:
        if r.pair is None or r.pair not in by_recon or r.id not in by_recon or r.id > r.pair:
            continue
        a = {(e.source, e.target, e.kind): e.native_weight for e in by_recon[r.id] if e.kind != "putative_mod"}
        b = {(e.source, e.target, e.kind): e.native_weight for e in by_recon[r.pair] if e.kind != "putative_mod"}
        for key in sorted(set(a) | set(b)):
            wa_, wb_ = a.get(key), b.get(key)
            if wa_ == wb_:
                continue
            kind = "synapse_count_differs" if wa_ is not None and wb_ is not None else "edge_in_one_bundle_only"
            values = tuple(sorted({f"{w:g}" for w in (wa_, wb_) if w is not None}))
            sources = tuple(
                sorted(f"{'absent' if w is None else format(w, 'g')}@{rid}" for w, rid in ((wa_, r.id), (wb_, r.pair)))
            )
            out.append(
                Conflict(
                    "edge",
                    f"{key[0]}>{key[1]}|{key[2]}",
                    "synapse_count",
                    kind,
                    "hard",
                    values,
                    sources,
                    (),
                    f"same upstream dataset bundled by two sources: {r.id} vs {r.pair}",
                )
            )
    return out


# --------------------------------------------------------------------------- atlas label audit


def read_atlas_labels(atlas_dir: Path) -> pa.Table:
    path = atlas_dir / "neurons.parquet"
    if not path.is_file():
        raise AnnotationError(f"atlas ROI table not found: {path} (run `randi2023 import` first)")
    return pq.read_table(path, columns=["recording_id", "label", "label_status", "label_duplicated"])


def audit_atlas_labels(
    atlas: pa.Table, table: AliasTable, non_neuron: dict[str, str]
) -> tuple[list[dict[str, Any]], dict[str, set[str]], dict[str, int], Counter[str]]:
    """One row per distinct atlas label with status name. Returns (rows, neuron -> recordings, neuron -> ROIs, other
    status counts). Only labels that are unique in their recording and resolve to one neuron count as functional."""
    stats: dict[tuple[str, str], dict[str, Any]] = {}
    other: Counter[str] = Counter()
    functional_recs: dict[str, set[str]] = defaultdict(set)
    functional_rois: dict[str, int] = defaultdict(int)
    cols = [atlas[c].to_pylist() for c in ("recording_id", "label", "label_status", "label_duplicated")]
    for rec, label, status, dup in zip(*cols, strict=True):
        if status not in ATLAS_LABEL_STATUSES:
            other[status] += 1
            continue
        s = stats.setdefault((label, status), {"rois": 0, "recs": set(), "dup": 0})
        s["rois"] += 1
        s["recs"].add(rec)
        s["dup"] += int(bool(dup))
        res = table.resolve(label)
        if res.status in ("exact", "alias") and not dup:
            functional_recs[res.candidates[0]].add(rec)
            functional_rois[res.candidates[0]] += 1
    rows: list[dict[str, Any]] = []
    for (label, status), s in sorted(stats.items()):
        res = table.resolve(label)
        suggested: str | None = None
        reason = res.reason
        if res.status == "unresolved":
            reason = non_neuron_reason(label, non_neuron) or "unknown name (no alias, not a canonical neuron)"
            near = suggestions(label, table.canonical)
            if len(near) == 1:
                suggested = near[0]
                reason += f"; one edit from {suggested} (suggestion only, not applied)"
            elif near:
                reason += f"; one edit from several neurons ({', '.join(near)})"
        rows.append(
            {
                "label": label,
                "label_status": status,
                "n_rois": s["rois"],
                "n_recordings": len(s["recs"]),
                "n_rois_label_duplicated": s["dup"],
                "resolved": res.status in ("exact", "alias"),
                "resolution": res.status,
                "neuron_id": res.neuron_id,
                "candidates": list(res.candidates),
                "mapping_kind": res.kind,
                "reason": reason,
                "suggested_neuron_id": suggested,
                "mapping_version": MAPPING_VERSION,
            }
        )
    return rows, functional_recs, dict(functional_rois), other


def unresolved_table(rows: Sequence[dict[str, Any]]) -> pa.Table:
    types = {
        "label": _STR,
        "label_status": _STR,
        "n_rois": pa.int64(),
        "n_recordings": pa.int64(),
        "n_rois_label_duplicated": pa.int64(),
        "resolved": pa.bool_(),
        "resolution": _STR,
        "neuron_id": _STR,
        "candidates": _LIST,
        "mapping_kind": _STR,
        "reason": _STR,
        "suggested_neuron_id": _STR,
        "mapping_version": _STR,
    }
    return pa.table({k: pa.array([r[k] for r in rows], t) for k, t in types.items()})


# --------------------------------------------------------------------------- neurons table


def _unique(values: Iterable[str]) -> str | None:
    s = set(values)
    return next(iter(s)) if len(s) == 1 else None


def neurons_table(
    canonical: Sequence[str],
    evidence: Sequence[Evidence],
    conflicts: Sequence[Conflict],
    *,
    anatomy: dict[str, set[str]],
    atlas_recs: dict[str, set[str]],
    atlas_rois: dict[str, int],
) -> pa.Table:
    by_neuron: dict[str, dict[str, list[Evidence]]] = defaultdict(lambda: defaultdict(list))
    for e in evidence:
        by_neuron[e.neuron_id][e.attribute].append(e)
    flagged = {(c.entity_id, c.attribute) for c in conflicts if c.entity_type == "neuron"}
    n_conf = Counter(c.entity_id for c in conflicts if c.entity_type == "neuron")
    cols: dict[str, list[Any]] = defaultdict(list)
    for nid in sorted(canonical):
        attrs = by_neuron.get(nid, {})
        cols["neuron_id"].append(nid)
        for a in ("cell_class", "region", "body_segment"):
            cols[a].append(_unique(e.value for e in attrs.get(a, [])))
        for a in WIDE_ATTRIBUTES:
            rows = attrs.get(a, [])
            cols[f"{a}_values"].append(sorted({e.value for e in rows}))
            cols[f"{a}_sources"].append(sorted({f"{e.value}@{e.source}" for e in rows}))
            cols[f"{a}_conflict"].append((nid, a) in flagged)
        pos = _unique(e.value for e in attrs.get("position_xyz", []))
        xyz = [float(v) for v in pos.split(",")] if pos else [None, None, None]
        for axis, v in zip("xyz", xyz, strict=True):
            cols[f"position_{axis}"].append(v)
        cols["ionotropic_receptor_genes"].append(sorted({e.value for e in attrs.get("ionotropic_receptor_gene", [])}))
        cols["has_transmitter_assignment"].append(has_transmitter(attrs.get("transmitter", [])))
        cols["n_anatomical_reconstructions"].append(len(anatomy.get(nid, set())))
        cols["atlas_n_recordings"].append(len(atlas_recs.get(nid, set())))
        cols["atlas_n_rois"].append(atlas_rois.get(nid, 0))
        cols["n_conflicts"].append(n_conf.get(nid, 0))
    types: dict[str, pa.DataType] = {"neuron_id": _STR}
    for a in ("cell_class", "region", "body_segment"):
        types[a] = _STR
    for a in WIDE_ATTRIBUTES:
        types[f"{a}_values"] = _LIST
        types[f"{a}_sources"] = _LIST
        types[f"{a}_conflict"] = pa.bool_()
    for axis in "xyz":
        types[f"position_{axis}"] = pa.float64()
    types["ionotropic_receptor_genes"] = _LIST
    types["has_transmitter_assignment"] = pa.bool_()
    types["n_anatomical_reconstructions"] = pa.int32()
    types["atlas_n_recordings"] = pa.int32()
    types["atlas_n_rois"] = pa.int32()
    types["n_conflicts"] = pa.int32()
    return pa.table({k: pa.array(cols[k], t) for k, t in types.items()})


def has_transmitter(rows: Iterable[Evidence]) -> bool:
    """True if some source assigns a transmitter other than an explicit 'unknown'."""
    return any(e.value not in UNKNOWN_TRANSMITTERS and not e.value.endswith("_uptake") for e in rows)


# --------------------------------------------------------------------------- driver


def _reason_key(reason: str) -> str:
    """Group audit reasons: drop details after the first ';', ':' or '(' and the member counts."""
    head = re.split(r"[;:(]", reason)[0].strip()
    return re.sub(r"\b\d+\b", "N", head)


def _counts(rows: Iterable[str]) -> dict[str, int]:
    return dict(sorted(Counter(rows).items()))


def run_build(
    raw_dir: Path,
    out_dir: Path,
    *,
    atlas_dir: Path,
    inputs: list[dict[str, Any]],
    repo_root: Path | None = None,
    recons: Sequence[Recon] = RECONSTRUCTIONS,
) -> dict[str, Any]:
    if out_dir.exists() and any(out_dir.iterdir()):
        raise AnnotationError(f"{out_dir} is not empty; normalized outputs are written once")
    atlas = read_atlas_labels(atlas_dir)

    tb = read_members(
        ow.archive_path(raw_dir),
        {ow.CELLS_PY, ow.CELL_INFO, ow.WANG_NT, ow.RIPOLL_NEURONS, ow.BENTLEY_MA, ow.BENTLEY_NP}
        | members_needed(recons, ow.SOURCE),
    )
    wn = read_members(
        wa.archive_path(raw_dir),
        {wa.NEURON_IDS, wa.GANGLIA, wa.POSITIONS, wa.LINEAGE, wa.FENYVES, wa.BENTLEY_MA, wa.BENTLEY_NP}
        | members_needed(recons, wa.SOURCE),
    )
    lists = ow.static_lists(tb[ow.CELLS_PY])
    canonical = lists["PREFERRED_HERM_NEURON_NAMES"]
    if len(set(canonical)) != len(canonical):
        raise AnnotationError("the canonical neuron list has duplicates")
    muscles = set(lists["PREFERRED_MUSCLE_NAMES"])
    cell_info = ow.read_cell_info(tb[ow.CELL_INFO])
    wang = ow.read_wang_nt(tb[ow.WANG_NT])
    ripoll = ow.read_ripoll_neurons(tb[ow.RIPOLL_NEURONS])
    fenyves = wa.read_fenyves(wn[wa.FENYVES])
    bentley_ma = ow.read_edgelist(tb[ow.BENTLEY_MA])
    bentley_np = ow.read_edgelist(tb[ow.BENTLEY_NP])
    atlas_ids = wa.read_neuron_ids(wn[wa.NEURON_IDS])
    lineage = wa.read_lineage(wn[wa.LINEAGE])
    non_neuron = {n: d for n, d in lineage.items() if n not in set(canonical)}
    members_by_source = {ow.SOURCE: tb, wa.SOURCE: wn}
    raw_edges = {r.id: extract(r, members_by_source[r.source]) for r in recons}

    # Observed labels -> where they were seen; the alias table is derived from these and the rules.
    observed: dict[str, set[str]] = defaultdict(set)

    def see(labels: Iterable[str], source: str) -> None:
        for lab in labels:
            observed[lab].add(source)

    see((r.neuron for r in wang), "wang2024-nt-atlas")
    see((r.neuron for r in ripoll), "ripoll2023-table-s7")
    see((t.neuron for t in fenyves.transmitters), "fenyves2020-s3")
    see(fenyves.types, "fenyves2020-s3")
    see(fenyves.receptors, "fenyves2020-s3")
    see((e.pre for e in fenyves.edges), "fenyves2020-s3")
    see((e.post for e in fenyves.edges), "fenyves2020-s3")
    see(atlas_ids, "wna-neuron-ids")
    ganglia = wa.read_ganglia(wn[wa.GANGLIA])
    see((n for names in ganglia.groups.values() for n in names), "wna-ganglia")
    see((p.name for p in wa.read_positions(wn[wa.POSITIONS])), "wna-positions")
    for rows, src in ((bentley_ma, "bentley2016-monoamine"), (bentley_np, "bentley2016-neuropeptide")):
        see((r[0] for r in rows), src)
        see((r[1] for r in rows), src)
    for r in recons:
        if r.loader != "toolbox_cache":
            see((lab for e in raw_edges[r.id] for lab in (e.pre, e.post) if lab not in muscles), r.id)
    atlas_labels = {
        lab
        for lab, st in zip(atlas["label"].to_pylist(), atlas["label_status"].to_pylist(), strict=True)
        if st in ATLAS_LABEL_STATUSES
    }
    see(atlas_labels, f"atlas:{ATLAS_NAME}")

    provisional = AliasTable(canonical, derive_aliases(canonical, observed))
    classes: dict[str, set[str]] = defaultdict(set)
    for w in wang:
        if w.cell_class:
            classes[w.cell_class].update(provisional.resolve(w.neuron).candidates)
    table = AliasTable(
        canonical, merge_rows(derive_aliases(canonical, observed), class_rows(canonical, classes, "wang2024"))
    )

    # Neuron attributes and their conflicts.
    book = EvidenceBook(table)
    gather_neuron_evidence(book, wn, lists, fenyves, cell_info, wang, ripoll, bentley_ma)
    conflicts = detect_conflicts(book.rows)

    # Edges, one reconstruction at a time.
    muscle_aliases = derive_muscle_aliases(muscles, observed)
    muscle_map = {m.alias: m.canonical_muscle_id for m in muscle_aliases}
    by_recon: dict[str, list[EdgeRow]] = {}
    dropped: dict[str, Counter[str]] = {}
    dropped_labels: dict[str, Counter[str]] = {}
    for r in recons:
        by_recon[r.id], dropped[r.id], dropped_labels[r.id] = map_recon(
            r, raw_edges[r.id], table, muscles, muscle_map, non_neuron
        )
    conflicts += edge_conflicts(by_recon, recons)
    all_edges = [e for r in recons for e in by_recon[r.id]]

    # Atlas audit and the three-way intersection.
    audit_rows, functional_recs, functional_rois, other_status = audit_atlas_labels(atlas, table, non_neuron)
    anatomy: dict[str, set[str]] = defaultdict(set)
    for e in all_edges:
        if e.kind in ("chem", "gap"):
            anatomy[e.source].add(e.reconstruction)
            anatomy[e.target].add(e.reconstruction)
    neuron_attr_rows: dict[str, list[Evidence]] = defaultdict(list)
    for ev in book.rows:
        if ev.attribute == "transmitter":
            neuron_attr_rows[ev.neuron_id].append(ev)
    molecular = {n for n, rows in neuron_attr_rows.items() if has_transmitter(rows)}
    intersection = _intersection(canonical, recons, by_recon, functional_recs, molecular)

    tables = {
        "neurons": neurons_table(
            canonical,
            book.rows,
            conflicts,
            anatomy=anatomy,
            atlas_recs=functional_recs,
            atlas_rois=functional_rois,
        ),
        "neuron_attributes": attributes_table(book.rows),
        "aliases": aliases_table(table.rows),
        "muscle_aliases": muscle_aliases_table(muscle_aliases),
        "edges": edges_table(all_edges),
        "edge_details": edge_details_table(all_edges),
        "conflicts": conflicts_table(conflicts),
        "unresolved": unresolved_table(audit_rows),
    }
    for name, t in tables.items():
        _write(t, out_dir / f"{name}.parquet")

    outputs = {
        p.name: {"sha256": _file_sha(p), "rows": pq.ParquetFile(p).metadata.num_rows}
        for p in sorted(out_dir.glob("*.parquet"))
    }
    atlas_file = atlas_dir / "neurons.parquet"
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "dataset": OUTPUT_NAME,
        "importer": f"occamworm.importers.annotations {__version__}",
        "code_revision": _git_revision(repo_root) if repo_root else None,
        "mapping_version": MAPPING_VERSION,
        "muscle_mapping_version": MUSCLE_MAPPING_VERSION,
        "inputs": inputs,
        "atlas_input": {"dataset": ATLAS_NAME, "file": "neurons.parquet", "sha256": _file_sha(atlas_file)},
        "outputs": outputs,
        "summary": _summary(
            canonical, table, book, conflicts, by_recon, dropped, audit_rows, other_status, intersection, all_edges
        ),
        "reconstructions": _reconstruction_info(recons, by_recon, dropped, dropped_labels),
        "cross_checks": _cross_checks(
            recons, by_recon, fenyves, bentley_ma, bentley_np, wn, canonical, table, atlas_ids
        ),
        "intersection": intersection,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=False) + "\n")
    return manifest


def _intersection(
    canonical: Sequence[str],
    recons: Sequence[Recon],
    by_recon: dict[str, list[EdgeRow]],
    functional_recs: dict[str, set[str]],
    molecular: set[str],
) -> dict[str, Any]:
    canon = set(canonical)
    functional = set(functional_recs)

    def anatomical(ids: Collection[str]) -> set[str]:
        out: set[str] = set()
        for rid in ids:
            for e in by_recon[rid]:
                if e.kind in ("chem", "gap"):
                    out.update((e.source, e.target))
        return out & canon

    typed = [r.id for r in recons if any(e.kind in ("chem", "gap") for e in by_recon[r.id])]
    adult = [r.id for r in recons if r.life_stage == "adult" and r.id in typed]
    anat_any, anat_adult = anatomical(typed), anatomical(adult)
    three_any = functional & anat_any & molecular
    return {
        "definitions": {
            "functional_atlas": "canonical neurons with at least one atlas ROI whose label is unique in its recording "
            "and resolves exactly or through a non-ambiguous alias",
            "anatomical_any": "canonical neurons in at least one chem or gap edge of any reconstruction",
            "anatomical_adult_any": "same, restricted to reconstructions with life_stage 'adult'",
            "molecular_annotated": "canonical neurons with a transmitter assignment other than unknown or uptake-only "
            "from any source",
        },
        "canonical_neurons": len(canon),
        "functional_atlas": len(functional & canon),
        "anatomical_any": len(anat_any),
        "anatomical_adult_any": len(anat_adult),
        "molecular_annotated": len(molecular & canon),
        "functional_and_anatomical_any": len(functional & anat_any),
        "functional_and_molecular": len(functional & molecular),
        "anatomical_any_and_molecular": len(anat_any & molecular),
        "three_way_any": len(three_any),
        "three_way_adult": len(functional & anat_adult & molecular),
        "by_reconstruction": {
            rid: {
                "anatomical": len(a := anatomical([rid])),
                "with_functional": len(a & functional),
                "three_way": len(a & functional & molecular),
            }
            for rid in typed
        },
        "functional_without_anatomy": sorted(functional - anat_any),
        "functional_without_molecular": sorted(functional - molecular),
        "three_way_neuron_ids": sorted(three_any),
    }


def _summary(
    canonical: Sequence[str],
    table: AliasTable,
    book: EvidenceBook,
    conflicts: Sequence[Conflict],
    by_recon: dict[str, list[EdgeRow]],
    dropped: dict[str, Counter[str]],
    audit_rows: Sequence[dict[str, Any]],
    other_status: Counter[str],
    intersection: dict[str, Any],
    all_edges: Sequence[EdgeRow],
) -> dict[str, Any]:
    by_res: Counter[str] = Counter()
    rois_by_res: Counter[str] = Counter()
    reasons: dict[str, Counter[str]] = {"ambiguous": Counter(), "unresolved": Counter()}
    for r in audit_rows:
        by_res[r["resolution"]] += 1
        rois_by_res[r["resolution"]] += r["n_rois"]
        if r["resolution"] in reasons:
            reasons[r["resolution"]][_reason_key(r["reason"])] += r["n_rois"]
    return {
        "neurons": len(canonical),
        "aliases": {
            "rows": len(table.rows),
            "distinct_aliases": len({r.alias for r in table.rows}),
            "by_kind": _counts(r.kind for r in table.rows),
            "ambiguous_aliases": len({r.alias for r in table.rows if r.ambiguous}),
        },
        "neuron_attribute_rows": len(book.rows),
        "unmapped_source_labels": {s: dict(sorted(c.items())) for s, c in sorted(book.unmapped.items())},
        "edges": {
            "rows": len(all_edges),
            "by_kind": _counts(e.kind for e in all_edges),
            "by_sign": _counts(f"{e.kind}:{e.sign}" for e in all_edges),
            "symmetrised_gap_rows": sum(e.symmetrised for e in all_edges),
            "by_reconstruction": {rid: _counts(e.kind for e in rows) for rid, rows in by_recon.items()},
            "dropped_by_reconstruction": {rid: dict(sorted(c.items())) for rid, c in dropped.items() if c},
        },
        "conflicts": {
            "rows": len(conflicts),
            "by_entity_type": _counts(c.entity_type for c in conflicts),
            "by_attribute_kind_severity": _counts(
                f"{c.entity_type}/{c.attribute}/{c.kind}/{c.severity}" for c in conflicts
            ),
        },
        "atlas_labels": {
            "audited_status": list(ATLAS_LABEL_STATUSES),
            "distinct_labels": len(audit_rows),
            "by_resolution": dict(sorted(by_res.items())),
            "rois_by_resolution": dict(sorted(rois_by_res.items())),
            "ambiguous_roi_reasons": dict(reasons["ambiguous"].most_common()),
            "unresolved_roi_reasons": dict(reasons["unresolved"].most_common()),
            "other_label_status_rois_not_audited": dict(sorted(other_status.items())),
        },
        "three_way_intersection": intersection["three_way_any"],
    }


def _reconstruction_info(
    recons: Sequence[Recon],
    by_recon: dict[str, list[EdgeRow]],
    dropped: dict[str, Counter[str]],
    dropped_labels: dict[str, Counter[str]],
) -> list[dict[str, Any]]:
    out = []
    for r in recons:
        rows = by_recon[r.id]
        out.append(
            {
                "id": r.id,
                "title": r.title,
                "source": r.source,
                "member": r.member,
                "life_stage": r.life_stage,
                "native_weight": r.weights,
                "pair": r.pair,
                "edges_by_kind": _counts(e.kind for e in rows),
                "neurons": len({e.source for e in rows} | {e.target for e in rows if e.target_kind == "neuron"}),
                "muscles": len({e.target for e in rows if e.target_kind == "muscle"}),
                "symmetrised_rows": sum(e.symmetrised for e in rows),
                "dropped": dict(sorted(dropped[r.id].items())),
                "dropped_top_labels": dict(dropped_labels[r.id].most_common(12)),
            }
        )
    return out


def _cross_checks(
    recons: Sequence[Recon],
    by_recon: dict[str, list[EdgeRow]],
    fenyves: wa.Fenyves,
    bentley_ma: list[tuple[str, str, str, str]],
    bentley_np: list[tuple[str, str, str, str]],
    wn: dict[str, bytes],
    canonical: Sequence[str],
    table: AliasTable,
    atlas_ids: Sequence[str],
) -> dict[str, Any]:
    out: dict[str, Any] = {"bundle_pairs": {}}
    for r in recons:
        if r.pair and r.pair in by_recon and r.id < r.pair:
            out["bundle_pairs"][f"{r.id} vs {r.pair}"] = compare(by_recon[r.id], by_recon[r.pair])
    if "fenyves2020-sign-prediction" in by_recon and "cook2019-herm" in by_recon:
        out["fenyves_vs_cook2019_herm_chem"] = compare(
            by_recon["fenyves2020-sign-prediction"], by_recon["cook2019-herm"], ("chem",)
        )
    out["fenyves_polarity_labels"] = _counts(e.polarity for e in fenyves.edges)
    for name, member, rows in (
        ("bentley2016_monoamine_toolbox_vs_wna", wa.BENTLEY_MA, bentley_ma),
        ("bentley2016_neuropeptide_toolbox_vs_wna", wa.BENTLEY_NP, bentley_np),
    ):
        other = set(ow.read_edgelist(wn[member]))
        mine = set(rows)
        out[name] = {"both": len(mine & other), "only_toolbox": len(mine - other), "only_wna": len(other - mine)}
    res = [table.resolve(n) for n in atlas_ids]
    out["wna_neuron_ids"] = {
        "ids": len(atlas_ids),
        "resolved": sum(r.status in ("exact", "alias") for r in res),
        "ambiguous": sorted(n for n, r in zip(atlas_ids, res, strict=True) if r.status == "ambiguous"),
        "unresolved": sorted(n for n, r in zip(atlas_ids, res, strict=True) if r.status == "unresolved"),
        "canonical_missing_from_wna_ids": sorted(set(canonical) - {c for r in res for c in r.candidates}),
    }
    return out
