"""Synthetic truth recovery and search-strategy comparison (M3 gate; §7.2, §7.8, §16; OW-012).

A cell plants one WRL program with known parameters on a seeded synthetic graph, generates T2a-like data from its
pair kernels (the same linear-response kernels the candidates are scored with) plus AR(1) noise of a chosen scale,
and evaluates every gated program of the declared pool on the training view of one outer fold with the nested
search's inner-fold machinery. The cell records where the true program (by canonical hash) ranks, whether it is the
winner or on the (L_total, NLL) Pareto front, how far its fitted parameters are from the planted ones, and how
close the winner's NLL is to the truth's. The full per-candidate NLL table is kept, so the three strategies of
§7.2 (enumeration order, uniform random, S1 hill-climbing over ``mutate.Mutator``) are replayed on it for any
budget k at no extra fit cost: a strategy that has "evaluated" a program reads its cached inner NLL.

Misspecified cells plant a program the pool cannot contain (more registers than the budget, operators outside the
allowlist) or no program at all (independent per-pair kernels); the cell then reports what the search selects and
how much of the B0-B3 NLL gap the best program closes.

Cells are independent, so the sweep runs on any number of processes and resumes from ``cells.jsonl``.
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from occamworm.analysis.dataset import POST
from occamworm.analysis.splits import SplitSpec, build_split
from occamworm.baselines.data import KERNEL_KNOTS, TaskData
from occamworm.baselines.synthetic import SyntheticSpec, generate
from occamworm.search.candidates import CandidateRejected, InnerResult, KernelCandidate
from occamworm.search.mutate import Mutator
from occamworm.search.nested import pareto_front
from occamworm.search.views import views_for_fold
from occamworm.search.wrl import Connectome, ResponseModel, WrlCandidate, enumerate_programs, search_volume
from occamworm.sim.graph import ChemicalEdge, GapJunction, GraphSpec, build_graph
from occamworm.sim.ir import Program, compile_source

FloatArray = npt.NDArray[np.float64]
EQUIVALENT_NATS = 2.0  # a winner within this many nats of the truth's inner NLL counts as functionally equivalent
MIN_GAP_NATS = 100.0  # below this B0-B3 gap (nats) the closed fraction is noise and is not reported
N_RESP = 4  # responders per target; the last one has no edge from its target (a negative control)


@dataclass(frozen=True)
class Cell:
    truth: str
    n_targets: int
    noise_a: float
    seed: int

    @property
    def cell_id(self) -> str:
        return f"{self.truth}|T{self.n_targets}|a{self.noise_a:g}|s{self.seed}"


def synthetic_graph(n_targets: int, n_resp: int, seed: int) -> Connectome:
    """Targets T_j with responders R_j_i: chemical and gap edges from T_j (the last responder is unconnected), a few
    random extra forward chemical edges (acyclic, so a program that feeds ``sum_in`` back stays bounded; gap
    junctions are symmetric and diffusive), weights scaled to unit maximum row sum as in the real connectome loader."""
    rng = np.random.default_rng(seed)
    tg = [f"T{j}" for j in range(n_targets)]
    rs = [[f"R{j}_{i}" for i in range(n_resp)] for j in range(n_targets)]
    chem: dict[tuple[str, str], tuple[float, int]] = {}
    gap: dict[tuple[str, str], float] = {}
    for j in range(n_targets):
        for i in range(n_resp - 1):
            chem[(tg[j], rs[j][i])] = (float(rng.uniform(0.4, 1.0)), -1 if rng.random() < 0.3 else 1)
            if rng.random() < 0.6:
                gap[(tg[j], rs[j][i])] = float(rng.uniform(0.3, 1.0))
        for a in range(n_resp - 1):
            for b in range(a + 1, n_resp - 1):  # forward edges only: the chemical graph stays acyclic and stable
                if rng.random() < 0.15:
                    chem[(rs[j][a], rs[j][b])] = (float(rng.uniform(0.2, 0.6)), 1 if rng.random() < 0.7 else -1)
        if j + 1 < n_targets and rng.random() < 0.5:
            chem[(tg[j], tg[j + 1])] = (float(rng.uniform(0.2, 0.6)), 1)
    row: dict[str, float] = {}
    for (_, post), (w, _) in chem.items():
        row[post] = row.get(post, 0.0) + w
    grow: dict[str, float] = {}
    for (g1, g2), w in gap.items():
        grow[g1] = grow.get(g1, 0.0) + w
        grow[g2] = grow.get(g2, 0.0) + w
    cs, gs = max(row.values(), default=1.0), max(grow.values(), default=1.0)
    names = tg + [n for group in rs for n in group]
    spec = GraphSpec(
        tuple((n, "neuron") for n in names),
        tuple(ChemicalEdge(a, b, w / cs, s) for (a, b), (w, s) in sorted(chem.items())),
        tuple(GapJunction(a, b, w / gs) for (a, b), w in sorted(gap.items())),
    )
    return Connectome(spec, build_graph(spec), {"synthetic": True, "n_targets": n_targets, "n_resp": n_resp})


def _pairs(n_targets: int, n_resp: int) -> list[tuple[str, str]]:
    return [(f"T{j}", f"R{j}_{i}") for j in range(n_targets) for i in range(n_resp)]


def plant(
    program: Program | None, theta: Sequence[float] | None, conn: Connectome, spec: SyntheticSpec
) -> tuple[TaskData, FloatArray]:
    """Synthetic data from a planted program's pair kernels, or from independent per-pair kernels if ``program`` is
    None. Returns the data and the noiseless kernels on the tent knots (zeros for the independent truth)."""
    if program is None:
        data, _ = generate(replace(spec, truth="independent"))
        return data, np.zeros(0)
    labels = {n: n for n, _ in conn.spec.neurons}
    model = ResponseModel(program, conn, _pairs(spec.n_targets, spec.n_responders), labels, spec.dt)
    c = model.dense(theta if theta is not None else program.default_theta())
    lags = np.arange(POST)
    k = np.stack([np.interp(lags, KERNEL_KNOTS, row) for row in c]).reshape(spec.n_targets, spec.n_responders, POST)
    data, _ = generate(spec, k)
    return data, c


def _truth_program(t: Mapping[str, Any], root: Path, index: Mapping[str, Mapping[str, Any]]) -> Program:
    if "rule" in t:
        return compile_source((root / t["rule"]).read_text())
    if "source" in t:
        return compile_source(t["source"])
    (row,) = [p for h, p in index.items() if h.startswith(t["hash"])]
    return compile_source(row["source"])


@dataclass
class Pool:
    """The declared candidate pool: gated enumerated programs plus the hand-written grammar members."""

    config: dict[str, Any]
    programs: list[dict[str, Any]]  # every enumerated program, in enumeration order
    kept: list[dict[str, Any]]  # rows passing the structural gates (enumerated and extra)
    extra: set[str]  # hashes of hand-written members the enumerator cannot produce (e.g. gap terms)
    log: dict[str, Any]

    @property
    def index(self) -> dict[str, dict[str, Any]]:
        return {p["hash"]: p for p in self.programs}


def build_pool(root: Path, cfg: Mapping[str, Any]) -> Pool:
    en_path = root / cfg["enumerate"]
    en = json.loads(en_path.read_text())
    progs = enumerate_programs(en_path)
    kept_cands, log = search_volume(progs, root)
    keep_hashes = {c.program.program_hash for c in kept_cands}
    kept = [p for p in progs if p["hash"] in keep_hashes]
    extra: set[str] = set()
    for rule in cfg.get("extra_rules", []):
        prog = compile_source((root / rule).read_text())
        row = {
            "hash": prog.program_hash,
            "source": (root / rule).read_text(),
            "name": Path(rule).stem,
            "l_struct": prog.l_struct_bits,
        }
        if prog.program_hash not in keep_hashes:
            kept.append(row)
            extra.add(prog.program_hash)
    return Pool(en, progs, kept, extra, {**log, "extra_rules": len(extra)})


def _candidate(row: Mapping[str, Any], root: Path, conn: Connectome, starts: int, seed: int) -> WrlCandidate:
    return WrlCandidate(
        row["source"],
        root,
        name=row.get("name", row["hash"][:12]),
        starts=starts,
        seed=seed,
        connectome=conn,
        labels={n: n for n, _ in conn.spec.neurons},
    )


def _rel_err(theta: Mapping[str, float], fitted: Sequence[float], program: Program) -> dict[str, float]:
    out: dict[str, float] = {}
    for p, value in zip(program.parameters, fitted, strict=True):
        if p.source_name in theta:
            truth = float(theta[p.source_name])
            out[p.source_name] = abs(value - truth) / abs(truth) if truth else abs(value)
    return out


def evaluate_cell(cell: Cell, cfg: Mapping[str, Any], root: Path, pool: Pool, misspec: bool) -> dict[str, Any]:
    t0 = time.time()
    truths = {t["name"]: t for t in cfg["misspecified" if misspec else "truths"]}
    t = truths[cell.truth]
    conn = synthetic_graph(cell.n_targets, N_RESP, cell.seed)
    spec = SyntheticSpec(
        "shared",
        n_animals=cfg["n_animals"],
        n_targets=cell.n_targets,
        n_responders=N_RESP,
        reps=cfg["reps"],
        noise_a=cell.noise_a,
        seed=cell.seed,
    )
    index = pool.index
    program = None if t.get("independent") else _truth_program(t, root, index)
    theta = program.theta_from_dict(t["theta"]) if program is not None and "theta" in t else None
    data, _ = plant(program, theta, conn, spec)
    split = build_split(
        data.animals, SplitSpec("rec", "group_kfold", cfg["outer_folds"], cfg["inner_folds"], cell.seed)
    )
    view, test = views_for_fold(data, split, 0)
    starts = int(cfg["starts"])
    cands: dict[str, WrlCandidate] = {}
    for row in pool.kept:
        cands[row["hash"]] = _candidate(row, root, conn, starts, cell.seed)
    results: dict[str, InnerResult] = {}
    rejected: dict[str, str] = {}
    for h, c in cands.items():
        try:
            results[h] = c.inner_select(view)
        except CandidateRejected as e:
            rejected[h] = str(e)
    refs: dict[str, dict[str, float]] = {}
    for fam, bits in cfg["kernel_families"].items():
        r = KernelCandidate(fam, float(bits)).inner_select(view)
        refs[fam] = {"inner_nll": r.inner_nll, "l_total_bits": r.l_total_bits}
    ranked = sorted(results.items(), key=lambda kv: (kv[1].inner_nll, kv[1].l_total_bits, kv[1].key))
    front = {r.key for r in pareto_front([r for _, r in ranked])}
    winner_hash, winner = ranked[0]
    out: dict[str, Any] = {
        "cell": cell.cell_id,
        **asdict(cell),
        "misspecified": misspec,
        "pool": len(cands),
        "evaluated": len(results),
        "rejected": len(rejected),
        "winner": {"hash": winner_hash, "name": cands[winner_hash].name, **_summ(winner)},
        "references": refs,
        "best_nll": winner.inner_nll,
        "n_train_animals": len(view.data.animals),
        "pairs": len(data.pairs),
        "neurons": conn.graph.n,
    }
    if program is not None:
        th = program.program_hash
        out["truth_hash"] = th
        out["truth_in_pool"] = th in results
        if th in results:
            tr = results[th]
            rank = 1 + next(i for i, (h, _) in enumerate(ranked) if h == th)
            out.update(
                truth_rank=rank,
                winner_is_truth=winner_hash == th,
                truth_on_pareto=tr.key in front,
                truth_nll=tr.inner_nll,
                excess_nll=winner.inner_nll - tr.inner_nll,
                equivalent=tr.inner_nll - winner.inner_nll <= EQUIVALENT_NATS,
            )
            fitted = cands[th].fit_final(view, 0.0)
            out["truth_theta_fit"] = fitted["theta"]
            if theta is not None:
                out["param_rel_err"] = _rel_err(t["theta"], fitted["theta"], program)
                out["theta_true"] = t["theta"]
            out["truth_outer_nll"] = sum(cands[th].score(fitted, test).values())
            wf = cands[winner_hash].fit_final(view, 0.0)
            out["winner_outer_nll"] = sum(cands[winner_hash].score(wf, test).values())
    b0, b3 = refs.get("B0"), refs.get("B3")
    if b0 and b3 and b0["inner_nll"] - b3["inner_nll"] > MIN_GAP_NATS:
        out["gap_closed"] = (b0["inner_nll"] - winner.inner_nll) / (b0["inner_nll"] - b3["inner_nll"])
    out["table"] = [
        {"hash": h, "nll": r.inner_nll, "bits": r.l_total_bits, "enumerated": h in index} for h, r in ranked
    ]
    if program is not None and not misspec and out.get("truth_in_pool") and cfg.get("strategies"):
        out["strategies"] = strategy_curves(cfg, pool, out["table"], out["truth_hash"], cell.seed)
    out["seconds"] = time.time() - t0
    return out


def _summ(r: InnerResult) -> dict[str, float]:
    return {"inner_nll": r.inner_nll, "l_total_bits": r.l_total_bits}


# ---------------------------------------------------------------------------------------------------------------
# Strategies (§7.2): replayed on a cached full-pool evaluation

_POOL_MUT: dict[str, Mutator] = {}


def _ranking_best(seen: Sequence[str], nll: Mapping[str, float], bits: Mapping[str, float]) -> str:
    return min(seen, key=lambda h: (nll[h], bits[h], h))


def enumeration_order(table: Sequence[Mapping[str, Any]], pool: Pool) -> list[str]:
    evaluated = {r["hash"] for r in table}
    return [p["hash"] for p in pool.kept if p["hash"] in evaluated]


def hill_climb_order(
    order: Sequence[str],
    nll: Mapping[str, float],
    mut: Mutator,
    rng: np.random.Generator,
    k_max: int,
    patience: int = 8,
) -> list[str]:
    """Candidates S1 evaluates, in order: start from a random pool member; propose an unseen in-grammar neighbour of
    the best-so-far (equal NLL moves too); restart at a random unseen member after ``patience`` proposals without
    improvement or when no unseen neighbour exists. Neighbours outside the evaluated pool (gated) are not proposed."""
    allowed = set(order)
    seen: list[str] = []
    seen_set: set[str] = set()
    cur: str | None = None
    stale = 0
    while len(seen) < min(k_max, len(order)):
        nxt: str | None = None
        if cur is not None and cur in mut.index and stale < patience:
            row = mut.propose(cur, rng, seen_set, allowed=allowed)
            nxt = None if row is None else str(row["hash"])
        if nxt is None:
            rest = [h for h in order if h not in seen_set]
            nxt = rest[int(rng.integers(len(rest)))]
            cur, stale = None, 0
        seen.append(nxt)
        seen_set.add(nxt)
        improved = cur is None or nll[nxt] < nll[cur]
        if cur is None or nll[nxt] <= nll[cur]:
            cur = nxt
        stale = 0 if improved else stale + 1
    return seen


def strategy_curves(
    cfg: Mapping[str, Any], pool: Pool, table: Sequence[Mapping[str, Any]], truth: str, seed: int
) -> dict[str, Any]:
    """P(true program is the selected winner after k evaluations) and mean excess best NLL, per strategy."""
    s = cfg["strategies"]
    nll = {r["hash"]: float(r["nll"]) for r in table}
    bits = {r["hash"]: float(r["bits"]) for r in table}
    order = enumeration_order(table, pool)
    ks = [k for k in s["budgets"] if k <= len(order)]
    if order[-1:] and ks[-1:] != [len(order)]:
        ks.append(len(order))
    if "mutator" not in _POOL_MUT:
        _POOL_MUT["mutator"] = Mutator(pool.config, pool.programs)
    mut = _POOL_MUT["mutator"]
    floor = nll[truth]

    def score(seq: Sequence[str]) -> list[tuple[bool, float, bool]]:
        out = []
        for k in ks:
            seen = seq[:k]
            best = _ranking_best(seen, nll, bits)
            out.append((best == truth, nll[best] - floor, truth in seen))
        return out

    def agg(runs: list[list[tuple[bool, float, bool]]]) -> dict[str, list[float]]:
        a = np.asarray(runs, dtype=np.float64)  # (runs, k, 3)
        return {
            "p_truth_selected": a[:, :, 0].mean(0).tolist(),
            "mean_excess_nll": a[:, :, 1].mean(0).tolist(),
            "p_truth_evaluated": a[:, :, 2].mean(0).tolist(),
        }

    rng = np.random.default_rng(1000 + seed)
    rand_runs = [score([order[i] for i in rng.permutation(len(order))]) for _ in range(s["seeds"])]
    s1_runs = [score(hill_climb_order(order, nll, mut, rng, ks[-1])) for _ in range(s["seeds"])]
    return {
        "k": ks,
        "pool": len(order),
        "enumeration": agg([score(order)]),
        "random": agg(rand_runs),
        "s1": agg(s1_runs),
    }


# ---------------------------------------------------------------------------------------------------------------
# Sweep driver

_POOL: dict[str, Pool] = {}


def _init_worker() -> None:
    for var in ("VECLIB_MAXIMUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = "1"
    os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")


def _job(args: tuple[str, str, dict[str, Any], bool]) -> dict[str, Any]:
    root_s, cell_json, cfg, misspec = args
    root = Path(root_s)
    if "pool" not in _POOL:
        _POOL["pool"] = build_pool(root, cfg)
    cell = Cell(**json.loads(cell_json))
    return evaluate_cell(cell, cfg, root, _POOL["pool"], misspec)


def cells_of(cfg: Mapping[str, Any]) -> list[tuple[Cell, bool]]:
    """Cells of the sweep: the product of each truth's axes (``axes`` of the truth, else of the config)."""
    out: list[tuple[Cell, bool]] = []
    for key, mis in (("truths", False), ("misspecified", True)):
        for t in cfg.get(key, []):
            ax = t.get("axes", cfg["misspec_axes"] if mis and "misspec_axes" in cfg else cfg["axes"])
            out += [(Cell(t["name"], n, a, s), mis) for n, a, s in product(ax["n_targets"], ax["noise_a"], ax["seeds"])]
    return out


def run(root: Path, cfg_path: Path, out: Path, workers: int = 2) -> Path:
    from occamworm.analysis.provenance import file_sha256, git_revision, write_run_manifest

    cfg = json.loads(cfg_path.read_text())
    write_run_manifest(
        out,
        {
            "kind": "recovery_sweep",
            "config_sha256": file_sha256(cfg_path),
            "grammar_sha256": file_sha256(root / cfg["enumerate"]),
            "rules_sha256": {r: file_sha256(root / r) for r in cfg.get("extra_rules", [])},
            "code": git_revision(root),
        },
    )
    log = out / "cells.jsonl"
    done = (
        {json.loads(line)["cell"] for line in log.read_text().splitlines() if line.strip()} if log.exists() else set()
    )
    todo = [(c, m) for c, m in cells_of(cfg) if c.cell_id not in done]
    print(f"{len(todo)} cells to run, {len(done)} done", file=sys.stderr, flush=True)
    jobs = [(str(root), json.dumps(asdict(c)), dict(cfg), m) for c, m in todo]

    def record(res: dict[str, Any]) -> None:
        with log.open("a") as fh:
            fh.write(json.dumps(res, sort_keys=True) + "\n")
        print(
            f"{res['cell']}: winner {res['winner']['name']} truth rank {res.get('truth_rank')} {res['seconds']:.0f}s",
            file=sys.stderr,
            flush=True,
        )

    if workers <= 1:
        for j in jobs:
            record(_job(j))
    else:
        import multiprocessing as mp

        with ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn"), initializer=_init_worker) as ex:
            for fut in as_completed([ex.submit(_job, j) for j in jobs]):
                record(fut.result())
    return log


# ---------------------------------------------------------------------------------------------------------------
# Summary


def _mean(xs: Sequence[float]) -> float:
    return float(np.mean(xs)) if len(xs) else math.nan


def load_cells(out: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in (out / "cells.jsonl").read_text().splitlines() if line.strip()]


def summarize(out: Path) -> str:
    """Markdown tables: recovery by truth, graph size and noise; misspecification; strategy curves."""
    run_json = json.loads((out / "run.json").read_text())
    cells = load_cells(out)
    cfg_name = out.name
    lines = [
        f"# Recovery sweep: {cfg_name}",
        "",
        f"Run `{run_json['run_id'][:12]}`, code `{str(run_json['inputs']['code']['commit'])[:12]}`.",
        "",
    ]
    good = [c for c in cells if not c["misspecified"]]
    for truth in sorted({c["truth"] for c in good}):
        rows = [c for c in good if c["truth"] == truth]
        pool = rows[0]["pool"]
        lines += [
            f"## Planted `{truth}` (pool of {pool} programs)",
            "",
            "| neurons | noise a | cells | winner=true | equivalent | on Pareto | gated | med. rank | med. par. err |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
        for n, a in sorted({(c["neurons"], c["noise_a"]) for c in rows}):
            g = [c for c in rows if c["neurons"] == n and c["noise_a"] == a]
            live = [c for c in g if c["truth_in_pool"]]

            def frac(key: str, cs: list[dict[str, Any]] = live, total: int = len(g)) -> str:
                return f"{sum(bool(c.get(key)) for c in cs)}/{total}"

            ranks = [c["truth_rank"] for c in live]
            errs = [max(c["param_rel_err"].values()) for c in live if c.get("param_rel_err")]
            lines.append(
                f"| {n} | {a:g} | {len(g)} | {frac('winner_is_truth')} | {frac('equivalent')} "
                f"| {frac('truth_on_pareto')} | {len(g) - len(live)} "
                f"| {np.median(ranks) if ranks else math.nan:.0f} | {np.median(errs) if errs else math.nan:.3f} |"
            )
        lines.append("")
    bad = [c for c in cells if c["misspecified"]]
    if bad:
        lines += [
            "## Misspecified truths",
            "",
            "| planted | neurons | noise a | winner | L_total (bits) | B0 NLL | B3 NLL | winner NLL | gap closed |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
        for c in sorted(bad, key=lambda c: (c["truth"], c["neurons"], c["noise_a"], c["seed"])):
            r = c["references"]
            lines.append(
                f"| {c['truth']} s{c['seed']} | {c['neurons']} | {c['noise_a']:g} | `{c['winner']['name']}` "
                f"| {c['winner']['l_total_bits']:.0f} | {r['B0']['inner_nll']:,.0f} | {r['B3']['inner_nll']:,.0f} "
                f"| {c['best_nll']:,.0f} | {c.get('gap_closed', math.nan):.2f} |"
            )
        lines.append("")
    strat = [c for c in good if "strategies" in c]
    if strat:
        keyed: dict[tuple[str, int, float], list[dict[str, Any]]] = {}
        for c in strat:
            keyed.setdefault((c["truth"], c["neurons"], c["noise_a"]), []).append(c)
        for (truth, n, a), cs in sorted(keyed.items()):
            ks = cs[0]["strategies"]["k"]
            lines += [
                f"## Strategies, planted `{truth}`, {n} neurons, noise a={a:g} ({len(cs)} data seeds)",
                "",
                "P(truth is the selected winner after k evaluations) / mean excess best NLL over the truth's (nats):",
                "",
                "| k | enumeration | random | S1 hill-climb |",
                "| --- | --- | --- | --- |",
            ]
            for i, k in enumerate(ks):
                cell = []
                for name in ("enumeration", "random", "s1"):
                    p = _mean([c["strategies"][name]["p_truth_selected"][i] for c in cs])
                    e = _mean([c["strategies"][name]["mean_excess_nll"][i] for c in cs])
                    cell.append(f"{p:.2f} / {e:.1f}")
                lines.append(f"| {k} | " + " | ".join(cell) + " |")
            lines.append("")
    return "\n".join(lines) + "\n"
