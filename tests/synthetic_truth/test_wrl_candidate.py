"""OW-012: a WRL program as a nested-search candidate recovers its generating parameters and wins the search."""

from pathlib import Path

import numpy as np
import pytest

from occamworm.analysis.dataset import POST
from occamworm.analysis.splits import SplitSpec, build_split
from occamworm.baselines.data import KERNEL_KNOTS, aggregate, compute_stats, trial_designs
from occamworm.baselines.synthetic import SyntheticSpec, generate
from occamworm.search.candidates import KernelCandidate
from occamworm.search.nested import run_fold
from occamworm.search.wrl import (
    Connectome,
    GateError,
    ResponseModel,
    WrlCandidate,
    budget_curve,
    enumerate_programs,
    fit_program,
    gate_structure,
    search_volume,
)
from occamworm.sim.graph import ChemicalEdge, GraphSpec, ModulatoryEdge, build_graph
from occamworm.sim.ir import compile_source, find_ow

pytestmark = pytest.mark.skipif(find_ow(required=False) is None, reason="the ow binary is not built")

ROOT = Path(__file__).resolve().parents[2]
SOURCE = """wrl 0.1
rule chem_leak
tier G1
state v : 1 = 0
param tau : s = 1.0 in [0.05, 10] trainable bits 12
param gain : 1 = 1.0 in [0, 5] trainable bits 10
input u = stimulus
input chem = sum_in(v, all)
next v = leaky_integrate(v, u + gain * chem, tau)
observe identity_v1(v)
"""
INERT = SOURCE.replace("u + gain * chem", "gain * chem").replace("input u = stimulus\n", "")
TRUTH = {"tau": 2.0, "gain": 0.6}
SPEC = SyntheticSpec("shared", n_animals=8, n_targets=3, n_responders=4, reps=2, seed=5)


def connectome() -> Connectome:
    neurons = [f"T{j}" for j in range(SPEC.n_targets)] + [
        f"R{j}_{i}" for j in range(SPEC.n_targets) for i in range(SPEC.n_responders)
    ]
    edges = [
        ChemicalEdge(f"T{j}", f"R{j}_{i}", 1.0, 1 if i % 2 == 0 else -1)
        for j in range(SPEC.n_targets)
        for i in range(SPEC.n_responders - 1)  # the last responder of each target is not connected
    ]
    spec = GraphSpec(tuple((n, "neuron") for n in neurons), tuple(edges))
    return Connectome(spec, build_graph(spec), {"synthetic": True})


CONN = connectome()
LABELS = {n: n for n, _ in CONN.spec.neurons}
PROGRAM = compile_source(SOURCE)


def truth_theta() -> list[float]:
    return PROGRAM.theta_from_dict(TRUTH)


def synthetic() -> tuple:
    pairs = [(f"T{j}", f"R{j}_{i}") for j in range(SPEC.n_targets) for i in range(SPEC.n_responders)]
    model = ResponseModel(PROGRAM, CONN, pairs, LABELS, SPEC.dt)
    c = model.dense(truth_theta())
    lags = np.arange(POST)
    k = np.stack([np.interp(lags, KERNEL_KNOTS, row) for row in c]).reshape(SPEC.n_targets, SPEC.n_responders, POST)
    data, _ = generate(SPEC, k)
    return data, model, c


DATA, MODEL, C_TRUE = synthetic()


def test_response_kernels_follow_edge_signs() -> None:
    peak = C_TRUE.reshape(SPEC.n_targets, SPEC.n_responders, -1)
    assert np.all(peak[:, 0].max(axis=1) > 0)  # excitatory edge
    assert np.all(peak[:, 1].min(axis=1) < 0)  # inhibitory edge
    assert np.allclose(peak[:, -1], 0.0)  # no edge, no response
    unmapped = ResponseModel(PROGRAM, CONN, [("T0", "nope")], LABELS, SPEC.dt)
    assert unmapped.empty and not unmapped.dense(truth_theta()).any()


def test_fit_recovers_generating_parameters() -> None:
    designs = trial_designs(DATA, None)
    ps = aggregate(compute_stats(DATA, designs), np.ones(len(DATA.animals), dtype=bool), len(DATA.pairs))
    fit = fit_program(MODEL, ps, starts=2, seed=0)
    theta = dict(zip([p.source_name for p in PROGRAM.parameters], fit.extra["theta"], strict=True))
    assert theta["tau"] == pytest.approx(TRUTH["tau"], rel=0.1)
    assert theta["gain"] == pytest.approx(TRUTH["gain"], rel=0.1)


def test_wrl_program_wins_nested_search_and_inert_program_is_gated(tmp_path: Path) -> None:
    with pytest.raises(GateError, match="inert"):
        gate_structure(compile_source(INERT))
    split = build_split(DATA.animals, SplitSpec("g4", "group_kfold", 4, 2, seed=1))
    wrl = WrlCandidate(SOURCE, ROOT, name="chem_leak", connectome=CONN, labels=LABELS)
    inert = WrlCandidate(INERT, ROOT, name="inert", connectome=CONN, labels=LABELS)
    res = run_fold(DATA, split, 0, [KernelCandidate("B0", 4.0), wrl, inert], tmp_path)
    assert res["winner"] == "wrl:chem_leak"
    assert set(res["animal_nll"]) == set(split.folds[0].test)
    assert "wrl:inert" in res["search_log"]["candidates_rejected"]


def test_search_volume_and_budget_curve() -> None:
    progs = enumerate_programs(ROOT / "configs" / "searches" / "g1-tiny.json", limit=40)
    kept, log = search_volume(progs, ROOT)
    assert log["enumerated"] == 40 and log["kept"] == len(kept) and sum(log["gated"].values()) + len(kept) == 40
    from occamworm.search.candidates import InnerResult

    curve = budget_curve([InnerResult(f"c{k}", 0, float(v), {}, 1, 0, 0, 0) for k, v in enumerate([5, 3, 4, 1])])
    assert curve["enumeration"] == [5, 3, 3, 1] and curve["random_mean"][-1] == 1


MOD_SOURCE = (
    SOURCE.replace("rule chem_leak", "rule mod_leak")
    .replace("input chem = sum_in(v, all)", "input chem = sum_in(v, mod)")
    .replace("in [0, 5]", "in [-5, 5]")
)
MOD_TRUTH = {"tau": 2.0, "gain": -0.6}


def test_modulatory_gain_and_sign_are_recovered() -> None:
    """A rule on modulatory edges only: the unsigned edges get their sign from the fitted gain."""
    spec = CONN.spec
    mods = tuple(ModulatoryEdge(e.pre, e.post, e.weight) for e in spec.chemical)
    mspec = GraphSpec(spec.neurons, (), (), mods)
    conn = Connectome(mspec, build_graph(mspec), {"synthetic": True})
    program = compile_source(MOD_SOURCE)
    pairs = [(f"T{j}", f"R{j}_{i}") for j in range(SPEC.n_targets) for i in range(SPEC.n_responders)]
    model = ResponseModel(program, conn, pairs, LABELS, SPEC.dt)
    c = model.dense(program.theta_from_dict(MOD_TRUTH))
    assert np.all(c.reshape(SPEC.n_targets, SPEC.n_responders, -1)[:, :-1].min(axis=2) < 0)
    lags = np.arange(POST)
    k = np.stack([np.interp(lags, KERNEL_KNOTS, row) for row in c]).reshape(SPEC.n_targets, SPEC.n_responders, POST)
    data, _ = generate(SPEC, k)
    ps = aggregate(compute_stats(data, trial_designs(data, None)), np.ones(len(data.animals), dtype=bool), len(pairs))
    fit = fit_program(model, ps, starts=2, seed=0)
    theta = dict(zip([p.source_name for p in program.parameters], fit.extra["theta"], strict=True))
    assert theta["gain"] == pytest.approx(MOD_TRUTH["gain"], rel=0.1)
    assert theta["tau"] == pytest.approx(MOD_TRUTH["tau"], rel=0.1)
    no_mod = WrlCandidate(MOD_SOURCE, ROOT, name="mod_leak", connectome=CONN, labels=LABELS)
    with pytest.raises(ValueError, match="no modulatory edges"):
        no_mod._model(data)
