"""Negative controls of the M4 search (§11.7; PREREGISTRATION_M4.md §9, OW-017).

* ``degree_preserving`` rewires a connectome by Maslov-Sneppen double-edge swaps. Chemical edges are swapped as
  directed edges ``(a -> b, c -> d) -> (a -> d, c -> b)``; each edge keeps its weight and sign with its presynaptic
  neuron (signs are a transmitter property), so every neuron keeps its in-degree, its out-degree and the weights and
  signs it sends. Gap junctions are swapped as undirected edges and keep their conductances, so every neuron keeps
  its gap degree. Swaps that would create a self-loop or a parallel edge are rejected. Modulatory edges are left as
  they are (their own control is the label shuffle of ``load_connectome``). Weights are rescaled to unit max row sum,
  the convention of ``load_connectome``.
* ``shuffle_target_labels`` permutes the stimulated-neuron identity across the trials of each recording. Traces and
  the measured autoresponse stay with their trial, and stimulation timing is unchanged; only which neuron the model is
  told was stimulated changes.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np

from occamworm.analysis.dataset import Windows
from occamworm.search.wrl import Connectome
from occamworm.sim.graph import ChemicalEdge, GapJunction, GraphSpec, build_graph

TARGET_FIELDS = ("stim_target_neuron_id", "stim_target_label")


def _rescale_chemical(edges: list[ChemicalEdge]) -> tuple[ChemicalEdge, ...]:
    row: dict[str, float] = {}
    for e in edges:
        row[e.post] = row.get(e.post, 0.0) + e.weight
    scale = max(row.values(), default=1.0)
    return tuple(
        sorted((dataclasses.replace(e, weight=e.weight / scale) for e in edges), key=lambda e: (e.pre, e.post))
    )


def _rescale_gap(edges: list[GapJunction]) -> tuple[GapJunction, ...]:
    row: dict[str, float] = {}
    for e in edges:
        row[e.a] = row.get(e.a, 0.0) + e.g
        row[e.b] = row.get(e.b, 0.0) + e.g
    scale = max(row.values(), default=1.0)
    return tuple(sorted((GapJunction(e.a, e.b, e.g / scale) for e in edges), key=lambda e: (e.a, e.b)))


def _swap_directed(edges: list[ChemicalEdge], rng: np.random.Generator, n_swaps: int) -> tuple[list[ChemicalEdge], int]:
    present = {(e.pre, e.post) for e in edges}
    done = 0
    for _ in range(n_swaps):
        i, j = (int(k) for k in rng.integers(0, len(edges), 2))
        e1, e2 = edges[i], edges[j]
        a, b, c, d = e1.pre, e1.post, e2.pre, e2.post
        if i == j or a == d or c == b or (a, d) in present or (c, b) in present:
            continue
        present -= {(a, b), (c, d)}
        present |= {(a, d), (c, b)}
        edges[i] = dataclasses.replace(e1, post=d)
        edges[j] = dataclasses.replace(e2, post=b)
        done += 1
    return edges, done


def _swap_undirected(edges: list[GapJunction], rng: np.random.Generator, n_swaps: int) -> tuple[list[GapJunction], int]:
    def key(x: str, y: str) -> tuple[str, str]:
        return (x, y) if x <= y else (y, x)

    present = {key(e.a, e.b) for e in edges}
    done = 0
    for _ in range(n_swaps):
        i, j = (int(k) for k in rng.integers(0, len(edges), 2))
        e1, e2 = edges[i], edges[j]
        a, b = (e1.a, e1.b) if rng.random() < 0.5 else (e1.b, e1.a)
        c, d = e2.a, e2.b
        new1, new2 = key(a, d), key(c, b)
        if i == j or a == d or c == b or new1 == new2 or new1 in present or new2 in present:
            continue
        present -= {key(e1.a, e1.b), key(e2.a, e2.b)}
        present |= {new1, new2}
        edges[i] = GapJunction(*new1, e1.g)
        edges[j] = GapJunction(*new2, e2.g)
        done += 1
    return edges, done


def degree_preserving(conn: Connectome, seed: int, swaps_per_edge: int = 10) -> Connectome:
    """The connectome with chemical and gap edges rewired by degree-preserving swaps (deterministic in ``seed``)."""
    rng = np.random.default_rng(seed)
    chem, n_chem = _swap_directed(list(conn.spec.chemical), rng, swaps_per_edge * len(conn.spec.chemical))
    gap, n_gap = _swap_undirected(list(conn.spec.gap), rng, swaps_per_edge * len(conn.spec.gap))
    spec = GraphSpec(conn.spec.neurons, _rescale_chemical(chem), _rescale_gap(gap), conn.spec.modulatory)
    meta: dict[str, Any] = {
        **conn.meta,
        "control": {"kind": "degree_preserving", "seed": seed, "chemical_swaps": n_chem, "gap_swaps": n_gap},
    }
    return Connectome(spec, build_graph(spec), meta)


def shuffle_target_labels(w: Windows, seed: int) -> Windows:
    """``w`` with the stimulated-neuron identity permuted across the trials of each recording (deterministic)."""
    t = w.trials
    rec = np.asarray(t["recording_id"], dtype=object)
    labelled = np.array([x is not None for x in t["stim_target_neuron_id"]])
    rng = np.random.default_rng(seed)
    order = np.arange(rec.size)
    for r in sorted({str(x) for x in rec}):
        idx = np.nonzero((rec == r) & labelled)[0]
        order[idx] = idx[rng.permutation(idx.size)]
    trials = dict(t)
    for f in TARGET_FIELDS:
        col = np.asarray(t[f], dtype=object)
        trials[f] = col[order].tolist() if isinstance(t[f], list) else col[order]
    return dataclasses.replace(w, trials=trials)
