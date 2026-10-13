"""Negative-control constructions of the M4 search (PREREGISTRATION_M4.md §9)."""

from __future__ import annotations

import dataclasses
from collections import Counter

import numpy as np

from occamworm.search.controls import degree_preserving, shuffle_target_labels
from occamworm.search.wrl import Connectome
from occamworm.sim.graph import ChemicalEdge, GapJunction, GraphSpec, build_graph


def _random_connectome(seed: int = 0, n: int = 30) -> Connectome:
    rng = np.random.default_rng(seed)
    names = [f"N{i:02d}" for i in range(n)]
    chem = {(names[a], names[b]) for a, b in rng.integers(0, n, (150, 2)) if a != b}
    sign = {x: (-1 if rng.random() < 0.3 else 1) for x in names}  # one sign per presynaptic neuron
    chemical = tuple(ChemicalEdge(s, t, float(rng.uniform(0.1, 1)), sign[s]) for s, t in sorted(chem))
    gap = {tuple(sorted((names[a], names[b]))) for a, b in rng.integers(0, n, (60, 2)) if a != b}
    gaps = tuple(GapJunction(a, b, float(rng.uniform(0.1, 1))) for a, b in sorted(gap))
    spec = GraphSpec(tuple((x, "neuron") for x in names), chemical, gaps)
    return Connectome(spec, build_graph(spec), {"reconstruction": "test"})


def test_degree_preserving_keeps_degrees_signs_and_weights() -> None:
    conn = _random_connectome()
    out = degree_preserving(conn, seed=3)
    old_c, new_c = conn.spec.chemical, out.spec.chemical
    assert Counter(e.pre for e in old_c) == Counter(e.pre for e in new_c)
    assert Counter(e.post for e in old_c) == Counter(e.post for e in new_c)
    assert Counter((e.pre, e.sign) for e in old_c) == Counter((e.pre, e.sign) for e in new_c)
    assert all(e.pre != e.post for e in new_c)
    assert len({(e.pre, e.post) for e in new_c}) == len(new_c)  # no parallel edges
    deg = Counter[str]()
    for e in conn.spec.gap:
        deg.update((e.a, e.b))
    deg_new = Counter[str]()
    for e in out.spec.gap:
        deg_new.update((e.a, e.b))
    assert deg == deg_new
    assert all(e.a < e.b for e in out.spec.gap) and len({(e.a, e.b) for e in out.spec.gap}) == len(out.spec.gap)
    # most edges move, weights are rescaled to unit max row sum
    moved = len({(e.pre, e.post) for e in old_c} - {(e.pre, e.post) for e in new_c})
    assert moved > 0.5 * len(old_c)
    rows = Counter[str]()
    for e in new_c:
        rows[e.post] += e.weight
    assert abs(max(rows.values()) - 1.0) < 1e-12
    assert out.meta["control"]["kind"] == "degree_preserving"


def test_degree_preserving_is_deterministic_in_seed() -> None:
    conn = _random_connectome()
    assert degree_preserving(conn, 1).spec == degree_preserving(conn, 1).spec
    assert degree_preserving(conn, 1).spec != degree_preserving(conn, 2).spec


@dataclasses.dataclass
class _W:
    trials: dict[str, object]


def test_shuffle_target_labels_permutes_within_recordings_only() -> None:
    rec = ["r1"] * 6 + ["r2"] * 5
    nid = ["A", "B", "C", "A", "B", None, "D", "E", "F", "G", "H"]
    w = _W({"recording_id": rec, "stim_target_neuron_id": nid, "stim_target_label": [x and x.lower() for x in nid]})
    out = shuffle_target_labels(w, seed=0)  # type: ignore[arg-type]
    new = list(out.trials["stim_target_neuron_id"])  # type: ignore[call-overload]
    for r in ("r1", "r2"):
        idx = [i for i, x in enumerate(rec) if x == r]
        assert Counter(new[i] for i in idx) == Counter(nid[i] for i in idx)
    assert new[5] is None  # unlabelled trials stay put
    assert new != nid
    labels = list(out.trials["stim_target_label"])  # type: ignore[call-overload]
    assert all((x and x.lower()) == y for x, y in zip(new, labels, strict=True))  # fields move together
    again = shuffle_target_labels(w, seed=0)  # type: ignore[arg-type]
    assert list(again.trials["stim_target_neuron_id"]) == new  # type: ignore[call-overload]
