"""M3 gate machinery: S1 mutation closure and a small end-to-end planted-truth recovery (OW-012)."""

import json
from pathlib import Path

import numpy as np
import pytest

from occamworm.search import recovery
from occamworm.search.mutate import ARITY, Mutator, parse
from occamworm.search.recovery import Cell, Pool, build_pool, evaluate_cell, hill_climb_order, strategy_curves
from occamworm.sim.ir import compile_source, find_ow

pytestmark = pytest.mark.skipif(find_ow(required=False) is None, reason="the ow binary is not built")

ROOT = Path(__file__).resolve().parents[2]
CFG = json.loads((ROOT / "configs" / "searches" / "recovery-reduced.json").read_text())
GRAMMAR = json.loads((ROOT / CFG["enumerate"]).read_text())
TRUTH_HASH = CFG["truths"][0]["hash"]


@pytest.fixture(scope="module")
def pool() -> Pool:
    return build_pool(ROOT, CFG)


@pytest.fixture(scope="module")
def mutator(pool: Pool) -> Mutator:
    return Mutator(pool.config, pool.programs)


def ops_of(tree: object) -> set[str]:
    return set() if isinstance(tree, str) else {tree[0]}.union(*(ops_of(c) for c in tree[1]))  # type: ignore[index]


def test_parse_render_round_trip_preserves_canonical_hash(pool: Pool, mutator: Mutator) -> None:
    for row in pool.programs[::25]:
        assert mutator.hash_of(parse(row["source"])) == row["hash"]


def test_mutation_closure_valid_in_budget_and_deduplicated(pool: Pool, mutator: Mutator) -> None:
    rng = np.random.default_rng(0)
    seen: set[str] = set()
    produced = 0
    for row in pool.programs[::20]:
        seen.add(row["hash"])
        for _ in range(4):
            child = mutator.propose(row["hash"], rng, seen)
            if child is None:
                continue
            produced += 1
            h = child["hash"]
            assert h not in seen  # canonical dedup against everything already seen, including the parent
            seen.add(h)
            prog = compile_source(child["source"])  # valid WRL
            assert prog.program_hash == h and h in mutator.index  # a member of the enumerated grammar and budget
            assert len(prog.registers) <= GRAMMAR["max_registers"]
            assert sum(p.trainable for p in prog.parameters) <= GRAMMAR["max_parameters"]
            assert ops_of(parse(child["source"])) <= set(GRAMMAR["ops"]) & set(ARITY)
    assert produced > 20


def test_mutation_respects_the_allowed_subset(pool: Pool, mutator: Mutator) -> None:
    rng = np.random.default_rng(1)
    allowed = {r["hash"] for r in pool.kept[:60]}
    start = pool.kept[0]["hash"]
    for _ in range(20):
        child = mutator.propose(start, rng, {start}, allowed=allowed)
        assert child is None or child["hash"] in allowed


def test_hill_climb_order_has_no_duplicates_and_is_seeded(pool: Pool, mutator: Mutator) -> None:
    order = [r["hash"] for r in pool.programs if r["hash"] in {k["hash"] for k in pool.kept}]
    nll = {h: float(i % 17) for i, h in enumerate(order)}
    a = hill_climb_order(order, nll, mutator, np.random.default_rng(3), 40)
    b = hill_climb_order(order, nll, mutator, np.random.default_rng(3), 40)
    assert a == b and len(a) == len(set(a)) == 40 and set(a) <= set(order)


def small_pool(pool: Pool, n: int = 6) -> Pool:
    truth = next(r for r in pool.kept if r["hash"].startswith(TRUTH_HASH))
    others = [r for r in pool.kept if r is not truth][:: max(1, len(pool.kept) // n)][:n]
    return Pool(pool.config, pool.programs, [*others, truth], set(), pool.log)


def test_planted_program_is_recovered_in_favorable_conditions(pool: Pool) -> None:
    cfg = {**CFG, "strategies": {"budgets": [1, 5], "seeds": 3}}
    out = evaluate_cell(Cell("chem-relay", 2, 1.0, 0), cfg, ROOT, small_pool(pool), misspec=False)
    assert out["winner_is_truth"] and out["truth_rank"] == 1 and out["truth_on_pareto"]
    assert out["param_rel_err"]["p0"] < 0.05
    curves = out["strategies"]
    assert curves["k"][-1] == out["evaluated"]
    for name in ("enumeration", "random", "s1"):
        assert curves[name]["p_truth_selected"][-1] == 1.0  # the full pool always selects the truth


def test_misspecified_truth_reports_the_selection(pool: Pool) -> None:
    out = evaluate_cell(Cell("independent", 2, 1.0, 0), CFG, ROOT, small_pool(pool), misspec=True)
    assert "truth_hash" not in out and out["winner"]["hash"] and out["misspecified"]
    assert out["references"]["B3"]["inner_nll"] < out["references"]["B0"]["inner_nll"]


def test_strategy_curves_on_a_synthetic_table(pool: Pool) -> None:
    kept = [r["hash"] for r in pool.kept[:50]]
    truth = kept[7]
    table = [{"hash": h, "nll": 100.0 + i if h != truth else 0.0, "bits": 100.0} for i, h in enumerate(kept)]
    cfg = {"strategies": {"budgets": [1, 10, 25], "seeds": 5}}
    sub = Pool(pool.config, pool.programs, [r for r in pool.kept if r["hash"] in set(kept)], set(), {})
    out = strategy_curves(cfg, sub, table, truth, 0)
    assert out["k"] == [1, 10, 25, 50]
    assert out["enumeration"]["p_truth_selected"] == [0.0, 1.0, 1.0, 1.0]  # the truth is 8th in enumeration order
    assert recovery.EQUIVALENT_NATS > 0
