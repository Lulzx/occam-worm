#!/usr/bin/env python3
"""Generate the WRL conformance cases (docs/language/WRL_SYNTAX.md, section 7).

Every expected value below is derived independently of the C++ interpreter: from closed-form analytic solutions
(`math.exp`, `math.tanh`), from exact rational arithmetic (`fractions.Fraction`) applied to the formulas of
section 6, or from small hand-written recurrences over exactly representable (dyadic) numbers. Nothing here is
ever copied from interpreter output.

Run `python3 tests/conformance/gen_cases.py`; it rewrites `tests/conformance/NN-name.json` deterministically.
"""

import json
import math
from fractions import Fraction
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------------------------------------------
# JSON emission: numeric arrays and flat records stay on one line, everything else is indented by two spaces.
# ---------------------------------------------------------------------------------------------------------------


def _is_scalar(x):
    return x is None or isinstance(x, bool | int | float | str)


def _is_numlist(x):
    return isinstance(x, list) and all(_is_scalar(e) and not isinstance(e, str) for e in x)


def _is_inline_dict(x):
    return isinstance(x, dict) and all(_is_scalar(v) or _is_numlist(v) for v in x.values())


def _scalar(x):
    if x is None:
        return "null"
    if isinstance(x, bool):
        return "true" if x else "false"
    if isinstance(x, int):
        return str(x)
    if isinstance(x, float):
        if not math.isfinite(x):
            raise ValueError("non-finite number in case file")
        return repr(x + 0.0)  # the +0.0 turns -0.0 into 0.0
    return json.dumps(x)


def dump(x, level=0):
    pad = "  " * (level + 1)
    end = "  " * level
    if _is_scalar(x):
        return _scalar(x)
    if _is_numlist(x):
        return "[" + ", ".join(_scalar(e) for e in x) + "]"
    if _is_inline_dict(x):
        return "{" + ", ".join(f"{json.dumps(k)}: {dump(v)}" for k, v in x.items()) + "}"
    if isinstance(x, list):
        return "[\n" + ",\n".join(pad + dump(e, level + 1) for e in x) + "\n" + end + "]"
    if isinstance(x, dict):
        body = ",\n".join(f"{pad}{json.dumps(k)}: {dump(v, level + 1)}" for k, v in x.items())
        return "{\n" + body + "\n" + end + "}"
    raise TypeError(type(x))


# ---------------------------------------------------------------------------------------------------------------
# Case construction helpers
# ---------------------------------------------------------------------------------------------------------------

CASES = []


def add_case(name, description, derivation, source, inp, check):
    CASES.append(
        {
            "name": name,
            "description": description,
            "derivation": derivation,
            "program": {"source": source},
            "input": inp,
            "check": check,
        }
    )


def neurons(*ids, types=None):
    out = []
    for i, nid in enumerate(ids):
        rec = {"id": nid}
        if types is not None and types[i] is not None:
            rec["type"] = types[i]
        out.append(rec)
    return out


def chem(pre, post, weight=1.0, sign=1, delay=0):
    return {"pre": pre, "post": post, "weight": float(weight), "sign": sign, "delay": delay}


def gapj(a, b, g):
    return {"a": a, "b": b, "g": float(g)}


def stim(neuron, start, end, amplitude=1.0):
    return {"neuron": neuron, "start": start, "end": end, "amplitude": float(amplitude)}


def sim_input(dt, n_steps, ids, chemical=(), gaps=(), stimulus=(), types=None, **extra):
    inp = {"dt": dt, "n_steps": n_steps}
    if "sample_ticks" in extra:
        inp["sample_ticks"] = extra.pop("sample_ticks")
    inp["graph"] = {"neurons": neurons(*ids, types=types), "chemical": list(chemical), "gap": list(gaps)}
    inp["stimulus"] = list(stimulus)
    inp.update(extra)
    return inp


def traces(registers=None, observation=None, abs_tol=1e-12):
    expected = {}
    if registers is not None:
        expected["registers"] = registers
    if observation is not None:
        expected["observation"] = observation
    return {"type": "traces", "tolerance": {"abs": abs_tol, "rel": 0}, "expected": expected}


def rows(series, ids, ticks=None):
    """series[id][tick] -> [tick][neuron] (optionally restricted to the sampled ticks)."""
    n_ticks = len(series[ids[0]])
    ticks = range(n_ticks) if ticks is None else ticks
    return [[float(series[i][t]) for i in ids] for t in ticks]


def compile_error(name, description, source, code):
    add_case(
        name,
        description,
        f"The program is rejected at compile time; section 2 of the language document assigns this defect to {code}.",
        source,
        {"dt": 0.1, "n_steps": 1, "graph": {"neurons": neurons("A")}},
        {"type": "compile_error", "code": code},
    )


# ---------------------------------------------------------------------------------------------------------------
# Independent mini-models used to derive expected values
# ---------------------------------------------------------------------------------------------------------------


def hist(seq, s):
    """History of section 6.5: constant extension of the initial state for negative ticks."""
    return seq[s] if s >= 0 else seq[0]


def stim_at(events, nid, t):
    acc = 0.0
    for e in events:
        if e["neuron"] == nid and e["start"] <= t < e["end"]:
            acc += e["amplitude"]
    return acc


def assign_net(ids, edges, events, n_steps, init=None, k_exc=1.0, k_inh=0.0):
    """Networks whose one-step update is a plain assignment (euler_leak with dt = tau = 1).

    x_i(t+1) = u_i(t) + k_exc * sum_exc w * x_j(t-d) + k_inh * sum_inh w * x_j(t-d),  with section 6.5 history.
    """
    init = init or {}
    series = {i: [float(init.get(i, 0.0))] for i in ids}
    for t in range(n_steps):
        new = {}
        for i in ids:
            exc = 0.0
            inh = 0.0
            for e in edges:
                if e["post"] != i:
                    continue
                term = e["weight"] * hist(series[e["pre"]], t - e["delay"])
                if e["sign"] > 0:
                    exc += term
                else:
                    inh += term
            new[i] = stim_at(events, i, t) + k_exc * exc + k_inh * inh
        for i in ids:
            series[i].append(new[i])
    return series


def gap_step(x, w, junctions, dt, scale=Fraction(1)):
    """Exact section 6.6 update with rational arithmetic: (w_i + dt S_i) / (1 + dt G_i) from the OLD values x."""
    n = len(x)
    big_g = [Fraction(0)] * n
    big_s = [Fraction(0)] * n
    for i, j, g in junctions:
        c = scale * g
        big_g[i] += c
        big_g[j] += c
        big_s[i] += c * x[j]
        big_s[j] += c * x[i]
    return [(w[i] + dt * big_s[i]) / (1 + dt * big_g[i]) for i in range(n)]


def gap_hold_series(x0, junctions, dt, n_steps, scale=Fraction(1)):
    """Identity local dynamics (w = x): returns [tick][neuron] as Fractions."""
    xs = [list(x0)]
    for _ in range(n_steps):
        xs.append(gap_step(xs[-1], xs[-1], junctions, dt, scale))
    return xs


def fr_rows(xs):
    return [[float(v) for v in row] for row in xs]


def lut_py(x, table):
    f = math.floor(x)
    idx = 0 if f <= 0 else (len(table) - 1 if f >= len(table) - 1 else f)
    return table[idx]


# ---------------------------------------------------------------------------------------------------------------
# Shared program text
# ---------------------------------------------------------------------------------------------------------------

# dt = tau = 1: euler_leak(x, target, 1) has rate dt/tau = 1, so one step is exactly `x := target`.
ASSIGN_HEAD = ["wrl 0.1", "tier G1", "dt_max 1"]
ASSIGN_TAIL = ["param tau : s = 1 fixed", "input u = stimulus"]
ASSIGN_PREAMBLE = [*ASSIGN_HEAD, "state v : 1 = 0", *ASSIGN_TAIL]
ASSIGN_EXC = [
    *ASSIGN_PREAMBLE,
    "input e = sum_in(v, exc)",
    "next v = euler_leak(v, u + e, tau)",
    "observe identity_v1(v)",
]
ASSIGN_EXC_INH = [
    *ASSIGN_PREAMBLE,
    "input e = sum_in(v, exc)",
    "input h = sum_in(v, inh)",
    "next v = euler_leak(v, u + e - h, tau)",
    "observe identity_v1(v)",
]
GAP_HOLD = ["wrl 0.1", "tier G1", "state v : 1 = 0", "gap v", "observe identity_v1(v)"]
RULE90 = ["wrl 0.1", "tier G0", "state s : 1 = 0", "next s = lut(count_in(s, 1), [0, 1, 0])", "observe identity_v1(s)"]

# ---------------------------------------------------------------------------------------------------------------
# 1-3: scalar integrators against closed forms
# ---------------------------------------------------------------------------------------------------------------


def case_leaky_const_drive():
    dt, tau, n = 0.1, 0.4, 20
    v = [1 - math.exp(-k * dt / tau) for k in range(n + 1)]
    add_case(
        "leaky-const-drive",
        "One neuron, leaky_integrate towards a constant stimulus of 1.",
        "Each step is the exact exponential step with the target held at 1, so v_n = 1 - exp(-n dt / tau) with "
        "dt = 0.1, tau = 0.4. The identity observation equals the register.",
        [
            "wrl 0.1",
            "tier G1",
            "state v : 1 = 0",
            "param tau : s = 0.4 fixed",
            "next v = leaky_integrate(v, stimulus, tau)",
            "observe identity_v1(v)",
        ],
        sim_input(dt, n, ["A"], stimulus=[stim("A", 0, n)]),
        traces({"v": [[x] for x in v]}, [[x] for x in v]),
    )


def case_euler_const_drive():
    dt, tau, n = 0.1, 0.4, 20
    r = dt / tau  # exactly 0.25
    v = [1 - (1 - r) ** k for k in range(n + 1)]
    add_case(
        "euler-const-drive",
        "One neuron, euler_leak towards a constant stimulus of 1 (rate dt/tau = 0.25).",
        "x + 0.25 (1 - x) has fixed point 1 and error multiplier 0.75, so v_n = 1 - 0.75^n.",
        [
            "wrl 0.1",
            "tier G1",
            "dt_max 0.1",
            "state v : 1 = 0",
            "param tau : s = 0.4 fixed",
            "next v = euler_leak(v, stimulus, tau)",
            "observe identity_v1(v)",
        ],
        sim_input(dt, n, ["A"], stimulus=[stim("A", 0, n)]),
        traces({"v": [[x] for x in v]}),
    )


def case_decay_zero_stimulus():
    dt, tau, n = 0.1, 0.5, 20
    v = [0.5 * math.exp(-k * dt / tau) for k in range(n + 1)]
    add_case(
        "decay-zero-stimulus",
        "decay(v, tau) from an initial value of 0.5 with no stimulus.",
        "decay is leaky_integrate towards 0, the exact solution of dx/dt = -x/tau: v_n = 0.5 exp(-n dt / tau).",
        [
            "wrl 0.1",
            "tier G1",
            "state v : 1 = 0.5",
            "param tau : s = 0.5 fixed",
            "next v = decay(v, tau)",
            "observe identity_v1(v)",
        ],
        sim_input(dt, n, ["A"]),
        traces({"v": [[x] for x in v]}),
    )


# ---------------------------------------------------------------------------------------------------------------
# 4-6: chemical synapses (dt = tau = 1, exact dyadic arithmetic)
# ---------------------------------------------------------------------------------------------------------------


def case_chemical_excitation():
    ids = ["A", "B", "C", "D"]
    edges = [chem("A", "B", 0.5), chem("B", "C", 2.0), chem("A", "D", 2.0)]
    events = [stim("A", 0, 2)]
    n = 7
    s = assign_net(ids, edges, events, n)
    add_case(
        "chemical-excitation",
        "A->B (0.5), B->C (2) and A->D (2) excitatory edges; A driven for ticks 0 and 1.",
        "With dt = tau = 1 a step is x := target. A(t+1) = u(t), B(t+1) = 0.5 A(t), C(t+1) = 2 B(t), "
        "D(t+1) = 2 A(t): A = 0,1,1,0,..; B = 0,0,.5,.5,0,..; C = 0,0,0,1,1,0,..; D = 0,0,2,2,0,.. (2-tick path).",
        ASSIGN_EXC,
        sim_input(1.0, n, ids, edges, stimulus=events),
        traces({"v": rows(s, ids)}, abs_tol=0),
    )


def case_chemical_inhibition():
    ids = ["A", "I", "B", "C"]
    edges = [chem("A", "B", 0.5), chem("I", "B", 0.25, sign=-1), chem("I", "C", 1.0, sign=-1)]
    events = [stim("A", 0, 6), stim("I", 0, 2, 2.0), stim("B", 0, 8, 3.0)]
    n = 8
    s = assign_net(ids, edges, events, n, k_exc=1.0, k_inh=-1.0)
    add_case(
        "chemical-inhibition",
        "Inhibitory edges subtracted from a constant drive using separate exc and inh sums.",
        "dt = tau = 1 so v(t+1) = u(t) + sum_in(exc) - sum_in(inh) with inhibitory magnitudes reported as "
        "non-negative: B(t+1) = 3 + 0.5 A(t) - 0.25 I(t), C(t+1) = -I(t) (no excitatory input), with A = "
        "1 for ticks 1..6 and I = 2 for ticks 1..2 (stimulus 2 on ticks 0..1). All values are dyadic.",
        ASSIGN_EXC_INH,
        sim_input(1.0, n, ids, edges, stimulus=events),
        traces({"v": rows(s, ids)}, abs_tol=0),
    )


def case_chemical_signed_sum():
    ids = ["P", "Q", "R", "S"]
    edges = [
        chem("P", "S", 1.0),
        chem("Q", "S", 2.0, sign=-1),
        chem("R", "S", 0.5),
        chem("Q", "R", 1.0, sign=-1, delay=1),
    ]
    events = [stim("P", 0, 5), stim("Q", 1, 4, 0.5), stim("R", 2, 3, 2.0)]
    n = 9
    s = assign_net(ids, edges, events, n, k_exc=2.0, k_inh=-2.0)
    add_case(
        "chemical-signed-sum",
        "sum_in(v, all) is the signed sum; the program doubles it.",
        "all = (sum of excitatory terms) - (sum of inhibitory magnitudes), then target = u + 2 all with "
        "dt = tau = 1, so v(t+1) = u(t) + 2 (sum_exc - sum_inh) evaluated on tick-t history (Q->R has delay 1). "
        "All numbers are small dyadic rationals, so the result is exact.",
        [
            *ASSIGN_PREAMBLE,
            "input a = sum_in(v, all)",
            "next v = euler_leak(v, u + 2 * a, tau)",
            "observe identity_v1(v)",
        ],
        sim_input(1.0, n, ids, edges, stimulus=events),
        traces({"v": rows(s, ids)}, abs_tol=0),
    )


# ---------------------------------------------------------------------------------------------------------------
# 7-13: gap junctions
# ---------------------------------------------------------------------------------------------------------------


def case_gap_diffusion():
    dt, g, n = 0.1, 2.0, 4
    xs = gap_hold_series([Fraction(1), Fraction(0)], [(0, 1, Fraction(g))], Fraction(dt), n)
    add_case(
        "gap-diffusion",
        "Two neurons with one gap junction, identity local dynamics, v = (1, 0), g = 2, dt = 0.1.",
        "Section 6.6 with w = x: v_i' = (v_i + dt g v_j) / (1 + dt g), evaluated exactly with rational arithmetic "
        "from the old values for four steps (the sum is conserved for two nodes, the difference shrinks by "
        "(1 - dt g)/(1 + dt g) per step).",
        GAP_HOLD,
        sim_input(dt, n, ["A", "B"], gaps=[gapj("A", "B", g)], initial_state={"v": {"A": 1.0}}),
        traces({"v": fr_rows(xs)}),
    )


def case_gap_dyadic_exact():
    n = 6
    xs = gap_hold_series([Fraction(1), Fraction(0)], [(0, 1, Fraction(3))], Fraction(1), n)
    add_case(
        "gap-dyadic-exact",
        "Gap coupling with dt = 1 and g = 3 so that every intermediate value is dyadic (zero tolerance).",
        "v_i' = (v_i + 3 v_j) / 4 from the old values: (1, 0) -> (1/4, 3/4) -> (5/8, 3/8) -> ... All divisions are "
        "by 4, so binary floating point reproduces the rational values exactly.",
        GAP_HOLD,
        sim_input(1.0, n, ["A", "B"], gaps=[gapj("A", "B", 3.0)], initial_state={"v": {"A": 1.0}}),
        traces({"v": fr_rows(xs)}, abs_tol=0),
    )


def case_gap_constant_equilibrium():
    ids = ["A", "B", "C", "D"]
    n = 20
    gaps = [gapj("A", "B", 0.3), gapj("B", "C", 1.7), gapj("B", "C", 0.4), gapj("C", "D", 2.2)]
    add_case(
        "gap-constant-equilibrium",
        "Uneven gap graph (with a parallel junction) at a uniform value 0.7 stays at 0.7.",
        "If every neuron and neighbour holds c, (c + dt c G) / (1 + dt G) = c for any conductances (property (a) "
        "of section 6.6), so every sampled value is 0.7 up to rounding.",
        ["wrl 0.1", "tier G1", "state v : 1 = 0.7", "gap v", "observe identity_v1(v)"],
        sim_input(0.1, n, ids, gaps=gaps),
        traces({"v": [[0.7] * 4 for _ in range(n + 1)]}, abs_tol=1e-14),
    )


def case_gap_equilibrium_scale():
    ids = ["A", "B", "C", "D"]
    n = 20
    gaps = [gapj("A", "B", 0.3), gapj("B", "C", 1.7), gapj("C", "D", 2.2), gapj("A", "D", 0.9)]
    add_case(
        "gap-equilibrium-scale",
        "Uniform 0.7 on a gap ring with a scale parameter at its default (lo + hi) / 2 = 2.5.",
        "The scale multiplies every conductance but equilibrium preservation holds for any non-negative "
        "conductances, so all values stay at 0.7 up to rounding.",
        [
            "wrl 0.1",
            "tier G1",
            "state v : 1 = 0.7",
            "param k : 1 in [0, 5] trainable bits 8",
            "gap v scale k",
            "observe identity_v1(v)",
        ],
        sim_input(0.1, n, ids, gaps=gaps),
        traces({"v": [[0.7] * 4 for _ in range(n + 1)]}, abs_tol=1e-14),
    )


def case_gap_large_step_bounded():
    dt, n = Fraction(1), 5
    junctions = [(0, 1, Fraction(1000)), (1, 2, Fraction(10))]
    xs = gap_hold_series([Fraction(1), Fraction(0), Fraction(-1)], junctions, dt, n)
    assert all(-1 <= v <= 1 for row in xs for v in row)  # convex-combination property (b)
    add_case(
        "gap-large-step-bounded",
        "dt * g = 1000 (stiff) on a 3-node chain stays inside [-1, 1] and matches the exact rational formula.",
        "Section 6.6 evaluated with Fractions: (v_i + dt S_i) / (1 + dt G_i) with G = (1000, 1010, 10). The script "
        "asserts that every value stays in the initial range [-1, 1] (property (b)).",
        GAP_HOLD,
        sim_input(
            1.0,
            n,
            ["A", "B", "C"],
            gaps=[gapj("A", "B", 1000.0), gapj("B", "C", 10.0)],
            initial_state={"v": {"A": 1.0, "C": -1.0}},
        ),
        traces({"v": fr_rows(xs)}),
    )


def case_gap_local_dynamics():
    dt, n = Fraction(1), 6
    junctions = [(0, 1, Fraction(1)), (1, 2, Fraction(3))]
    scale = Fraction(1, 2)  # default (0 + 1) / 2
    x = [Fraction(0)] * 3
    xs = [list(x)]
    for _ in range(n):
        w = [Fraction(1), Fraction(0), Fraction(0)]  # euler_leak with rate 1 gives w = stimulus
        x = gap_step(x, w, junctions, dt, scale)
        xs.append(x)
    add_case(
        "gap-local-dynamics",
        "Gap stage applied to the provisional value of a non-trivial local update with a default scale parameter.",
        "w = u (euler_leak with dt = tau = 1), then x' = (w_i + dt S_i)/(1 + dt G_i) with S from the OLD x and "
        "c = 0.5 * g (default scale (0+1)/2): exact rational recurrence with u = (1, 0, 0).",
        [
            "wrl 0.1",
            "tier G1",
            "dt_max 1",
            "state v : 1 = 0",
            "param tau : s = 1 fixed",
            "param k : 1 in [0, 1] trainable bits 8",
            "next v = euler_leak(v, stimulus, tau)",
            "gap v scale k",
            "observe identity_v1(v)",
        ],
        sim_input(1.0, n, ["A", "B", "C"], gaps=[gapj("A", "B", 1.0), gapj("B", "C", 3.0)], stimulus=[stim("A", 0, n)]),
        traces({"v": fr_rows(xs)}),
    )


def case_gap_multigraph():
    dt, n = Fraction(1, 2), 4
    # (A,B,1) and (B,A,0.5) and (A,B,0.25) are three parallel junctions of total conductance 1.75.
    junctions = [(0, 1, Fraction(1)), (1, 0, Fraction(1, 2)), (0, 1, Fraction(1, 4)), (1, 2, Fraction(2))]
    xs = gap_hold_series([Fraction(1), Fraction(0), Fraction(-1)], junctions, dt, n)
    add_case(
        "gap-multigraph",
        "Parallel gap junctions, including one listed as (b, a), are separate junctions.",
        "Three junctions between A and B (1, 0.5, 0.25) add conductances to G and S of both endpoints; exact "
        "rational recurrence of section 6.6 with dt = 1/2.",
        GAP_HOLD,
        sim_input(
            0.5,
            n,
            ["A", "B", "C"],
            gaps=[gapj("A", "B", 1.0), gapj("B", "A", 0.5), gapj("A", "B", 0.25), gapj("B", "C", 2.0)],
            initial_state={"v": {"A": 1.0, "C": -1.0}},
        ),
        traces({"v": fr_rows(xs)}),
    )


# ---------------------------------------------------------------------------------------------------------------
# 14-16: delays and history
# ---------------------------------------------------------------------------------------------------------------


def case_delay_ring_wraparound():
    ids = ["A", "B", "C", "D"]
    edges = [chem("A", "B", 1.0, delay=2), chem("A", "C", 1.0, delay=0), chem("A", "D", 1.0, delay=1)]
    events = [stim("A", 0, 1), stim("A", 5, 6)]
    n = 14
    s = assign_net(ids, edges, events, n)
    add_case(
        "delay-ring-wraparound",
        "Pulses at ticks 0 and 5 through delays 2, 0 and 1 (ring of 3 slots wraps several times); every tick sampled.",
        "A(k) = u(k-1), so A is 1 at ticks 1 and 6. B(t+1) = A(t-2), C(t+1) = A(t), D(t+1) = A(t-1) with A(s<0) = "
        "A(0) = 0: B pulses at ticks 4 and 9, C at 2 and 7, D at 3 and 8.",
        ASSIGN_EXC,
        sim_input(1.0, n, ids, edges, stimulus=events),
        traces({"v": rows(s, ids)}, abs_tol=0),
    )


def case_history_init():
    ids = ["A", "B", "C"]
    edges = [chem("A", "B", 0.5, delay=3), chem("B", "C", 1.0, delay=2)]
    init = {"A": 1.0, "B": 0.25, "C": 0.0}
    n = 10
    s = assign_net(ids, edges, [], n, init=init)
    add_case(
        "history-init",
        "Initial state (dense list form) is read for negative ticks before the history window fills.",
        "A(t+1) = 0 (no input), so A = 1 at tick 0 and 0 afterwards. B(t+1) = 0.5 A(t-3): ticks 1..3 read "
        "A(-2..0) = 1, giving 0.5, then 0. C(t+1) = B(t-2): ticks 1..2 read the initial B = 0.25, tick 3 reads B(0) = "
        "0.25, ticks 4..6 read B(1..3) = 0.5, then B(4..) = 0.",
        ASSIGN_EXC,
        sim_input(1.0, n, ids, edges, initial_state={"v": [1.0, 0.25, 0.0]}),
        traces({"v": rows(s, ids)}, abs_tol=0),
    )


def case_delay_op_own_register():
    ids = ["A", "B"]
    events = [stim("A", 0, 1), stim("A", 2, 3)]
    n = 12
    series = {"A": [0.5], "B": [0.5]}
    for t in range(n):
        series["A"].append(stim_at(events, "A", t) + hist(series["A"], t - 2))
        series["B"].append(stim_at(events, "B", t) + hist(series["B"], t - 2))
    add_case(
        "delay-op-own-register",
        "delay(v, 2) reads the neuron's own register two ticks back; initial value 0.5.",
        "dt = tau = 1 so v(t+1) = u(t) + v(t-2) with v(s<0) = v(0) = 0.5. B (no stimulus) stays 0.5; A gets pulses "
        "at ticks 0 and 2 that recirculate with period 3.",
        [
            "wrl 0.1",
            "tier G1",
            "dt_max 1",
            "state v : 1 = 0.5",
            "param tau : s = 1 fixed",
            "next v = euler_leak(v, stimulus + delay(v, 2), tau)",
            "observe identity_v1(v)",
        ],
        sim_input(1.0, n, ids, stimulus=events),
        traces({"v": rows(series, ids)}, abs_tol=0),
    )


# ---------------------------------------------------------------------------------------------------------------
# 17-18: adaptation registers
# ---------------------------------------------------------------------------------------------------------------


def case_adaptation_register():
    n = 12
    a, u = 1.5, 1.0
    v, h = [0.0], [0.0]
    for _ in range(n):
        v_old, h_old = v[-1], h[-1]
        v.append(u - a * h_old)
        h.append(max(v_old, 0.0))
    add_case(
        "adaptation-register",
        "Two registers: v' = u - 1.5 h and h' = relu(v) with simultaneous (old-value) updates.",
        "dt = tau = 1: v(t+1) = 1 - 1.5 h(t) and h(t+1) = relu(v(t)) use only tick-t values. Hand iteration from "
        "(0,0): v = 0, 1, 1, -.5, -.5, 1, 1, ..., h = 0, 0, 1, 1, 0, 0, 1, ... (relu clips the negative v).",
        [
            "wrl 0.1",
            "tier G1",
            "dt_max 1",
            "state v : 1 = 0",
            "state h : 1 = 0",
            "param tau : s = 1 fixed",
            "param adapt : 1 = 1.5 fixed",
            "input u = stimulus",
            "next v = euler_leak(v, u - adapt * h, tau)",
            "next h = euler_leak(h, relu(v), tau)",
            "observe identity_v1(v)",
        ],
        sim_input(1.0, n, ["A"], stimulus=[stim("A", 0, n)]),
        traces({"v": [[x] for x in v], "h": [[x] for x in h]}, [[x] for x in v], abs_tol=0),
    )


def case_adaptation_exponential():
    dt, tau_h, n = 0.05, 0.3, 24
    h = [1 - math.exp(-k * dt / tau_h) for k in range(n + 1)]
    add_case(
        "adaptation-exponential",
        "v is held at 1 (no next); h follows v with leaky_integrate and is observed.",
        "h is the exact exponential relaxation to the constant 1: h_n = 1 - exp(-n dt / tau_h) with dt = 0.05, "
        "tau_h = 0.3. The holding register stays 1.",
        [
            "wrl 0.1",
            "tier G1",
            "state v : 1 = 1",
            "state h : 1 = 0",
            "param tau_h : s = 0.3 fixed",
            "next h = leaky_integrate(h, v, tau_h)",
            "observe identity_v1(h)",
        ],
        sim_input(dt, n, ["A"]),
        traces({"v": [[1.0]] * (n + 1), "h": [[x] for x in h]}, [[x] for x in h]),
    )


# ---------------------------------------------------------------------------------------------------------------
# 19-20: calcium observation
# ---------------------------------------------------------------------------------------------------------------


def case_calcium_impulse():
    dt, tau = 0.2, 0.8
    a = math.exp(-dt / tau)
    ticks = [0, 1, 2, 5, 10]
    c = [0.0 if k == 0 else (1 - a) * a ** (k - 1) for k in ticks]
    x = [1.0 if k == 1 else 0.0 for k in ticks]
    add_case(
        "calcium-impulse",
        "x(t+1) = stimulus with an impulse at tick 0; calcium_linear_v1 observed at sparse sample ticks.",
        "x = 1 only at tick 1. c(0) = x(0) = 0, c(1) = 1 + (0 - 1) a = 1 - a, then c decays by a each step: "
        "c(k) = (1 - a) a^(k-1) with a = exp(-dt/tau), dt = 0.2, tau = 0.8.",
        [
            "wrl 0.1",
            "tier G1",
            "state x : 1 = 0",
            "param tau : s = 0.8 fixed",
            "next x = stimulus",
            "observe calcium_linear_v1(x, tau)",
        ],
        sim_input(dt, 10, ["A"], stimulus=[stim("A", 0, 1)], sample_ticks=ticks),
        traces({"x": [[v] for v in x]}, [[v] for v in c]),
    )


def case_calcium_step():
    dt, tau, n = 0.25, 0.5, 8
    a = math.exp(-dt / tau)
    x0 = {"A": 0.0, "B": 0.6, "C": 0.6}
    # A: x = 1 from tick 1 (stimulus), B: x = 0.6 then 0 (no stimulus), C: x = 0.6 then 1 (stimulus).
    obs = {"A": [0.0], "B": [0.6], "C": [0.6]}
    reg = {"A": [0.0], "B": [0.6], "C": [0.6]}
    for k in range(1, n + 1):
        obs["A"].append(1 - a**k)
        obs["B"].append(0.6 * a**k)
        obs["C"].append(1 - 0.4 * a**k)
        reg["A"].append(1.0)
        reg["B"].append(0.0)
        reg["C"].append(1.0)
    ids = ["A", "B", "C"]
    add_case(
        "calcium-step",
        "Calcium filter of a step input and of non-zero initial registers (c(0) = x(0)).",
        "c(0) = x(0). With x(k) = X for k >= 1: c(k) = X + (x(0) - X) a^k, a = exp(-dt/tau), dt = 0.25, tau = 0.5: "
        "A (0 -> 1): 1 - a^k; B (0.6 -> 0): 0.6 a^k; C (0.6 -> 1): 1 - 0.4 a^k.",
        [
            "wrl 0.1",
            "tier G1",
            "state x : 1 = 0",
            "next x = stimulus",
            "observe calcium_linear_v1(x, 0.5[s])",
        ],
        sim_input(
            dt,
            n,
            ids,
            stimulus=[stim("A", 0, n), stim("C", 0, n)],
            initial_state={"x": {k: x0[k] for k in ("B", "C")}},
        ),
        traces({"x": rows(reg, ids)}, rows(obs, ids)),
    )


# ---------------------------------------------------------------------------------------------------------------
# 21-23: stimulus variants and edge edits
# ---------------------------------------------------------------------------------------------------------------


def case_stimulus_variants():
    ids = ["A", "B", "C"]
    edges = [chem("A", "B", 1.0), chem("B", "C", 1.0)]
    events = [stim("A", 0, 4, 1.0), stim("A", 2, 6, 0.5), stim("B", 3, 4, 2.0), stim("C", 5, 8, -1.0)]
    n = 10
    s = assign_net(ids, edges, events, n)
    add_case(
        "stimulus-variants",
        "Constant, overlapping (accumulating), impulse and negative stimulus events on a 3-neuron chain.",
        "A: u = 1,1,1.5,1.5,.5,.5,0 (events [0,4) amp 1 and [2,6) amp .5 add); B: impulse 2 at tick 3; C: -1 on "
        "[5,8). With dt = tau = 1: A(t+1) = u_A(t), B(t+1) = u_B(t) + A(t), C(t+1) = u_C(t) + B(t).",
        ASSIGN_EXC,
        sim_input(1.0, n, ids, edges, stimulus=events),
        traces({"v": rows(s, ids)}, abs_tol=0),
    )


def case_edge_deletion():
    ids = ["A", "B", "C", "D"]
    edges = [chem("A", "B", 0.5), chem("B", "C", 2.0), chem("A", "D", 2.0)]
    kept = [e for e in edges if (e["pre"], e["post"]) != ("A", "B")]
    events = [stim("A", 0, 2)]
    n = 7
    s = assign_net(ids, kept, events, n)
    add_case(
        "edge-deletion",
        "The graph of chemical-excitation with edits.delete_chemical removing A->B.",
        "Without the A->B edge, B receives nothing and so neither does C: both stay 0. D(t+1) = 2 A(t) is "
        "untouched: D = 0,0,2,2,0,...",
        ASSIGN_EXC,
        sim_input(1.0, n, ids, edges, stimulus=events, edits={"delete_chemical": [["A", "B"]]}),
        traces({"v": rows(s, ids)}, abs_tol=0),
    )


def case_gap_edge_deletion():
    n = 5
    xs = gap_hold_series(
        [Fraction(1), Fraction(0), Fraction(1, 2)], [(0, 1, Fraction(3))], Fraction(1), n
    )  # B-C deleted: C isolated
    add_case(
        "gap-edge-deletion",
        "edits.delete_gap removes the B-C junction, leaving C isolated at its initial value.",
        "Only the A-B junction (g = 3) remains: the dyadic recurrence (v_i + 3 v_j)/4 on (A, B) from (1, 0); the "
        "isolated C keeps 0.5 (G = S = 0 leaves w unchanged).",
        GAP_HOLD,
        sim_input(
            1.0,
            n,
            ["A", "B", "C"],
            gaps=[gapj("A", "B", 3.0), gapj("B", "C", 1.0)],
            initial_state={"v": {"A": 1.0, "C": 0.5}},
            edits={"delete_gap": [["B", "C"]]},
        ),
        traces({"v": fr_rows(xs)}, abs_tol=0),
    )


# ---------------------------------------------------------------------------------------------------------------
# 24-27: G0 cellular automata
# ---------------------------------------------------------------------------------------------------------------


def case_g0_rule90():
    m, n = 8, 6
    ids = [f"N{i}" for i in range(m)]
    edges = [chem(ids[i], ids[(i + d) % m]) for i in range(m) for d in (-1, 1)]
    cells = [0] * m
    cells[3] = 1
    gens = [list(cells)]
    for _ in range(n):
        cells = [cells[(i - 1) % m] ^ cells[(i + 1) % m] for i in range(m)]
        gens.append(list(cells))
    add_case(
        "g0-rule90",
        "Ring of 8 cells with in-edges from both ring neighbours; one live cell; elementary rule 90.",
        "count_in(s, 1) is the number of live neighbours; lut [0, 1, 0] maps it to the XOR of the two neighbours: "
        "new[i] = old[i-1] xor old[i+1], iterated 6 generations by hand from a single live cell.",
        RULE90,
        sim_input(1.0, n, ids, edges, initial_state={"s": {"N3": 1}}),
        traces({"s": [[float(c) for c in g] for g in gens]}, abs_tol=0),
    )


def case_g0_own_state_table():
    m, n = 5, 8
    table = [0, 0, 1, 0, 1, 1]
    ids = [f"C{i}" for i in range(m)]
    edges = [chem(ids[i], ids[i + d]) for i in range(m) for d in (-1, 1) if 0 <= i + d < m]
    cells = [0, 1, 0, 0, 1]
    gens = [list(cells)]
    for _ in range(n):
        new = []
        for i in range(m):
            count = sum(1 for j in (i - 1, i + 1) if 0 <= j < m and cells[j] == 1)
            new.append(table[min(3 * cells[i] + count, len(table) - 1)])
        cells = new
        gens.append(list(cells))
    add_case(
        "g0-own-state-table",
        "Path of 5 cells; the next state is a table lookup on 3 * own state + number of live in-neighbours.",
        "idx = 3 s + count is looked up in [0,0,1,0,1,1]: (s,count) = (0,0)->0, (0,1)->0, (0,2)->1, (1,0)->0, "
        "(1,1)->1, (1,2)->1. End cells have one neighbour. Eight generations are iterated by hand from 0 1 0 0 1.",
        [
            "wrl 0.1",
            "tier G0",
            "state s : 1 = 0",
            "next s = lut(add(mul(s, 3), count_in(s, 1)), [0, 0, 1, 0, 1, 1])",
            "observe identity_v1(s)",
        ],
        sim_input(1.0, n, ids, edges, initial_state={"s": {"C1": 1, "C4": 1}}),
        traces({"s": [[float(c) for c in g] for g in gens]}, abs_tol=0),
    )


def case_g0_lut_clamp():
    ids = ["A", "B", "C", "D", "E"]
    table = [0, 2, 1, 3]
    events = [
        stim("A", 0, 6, 1.0),
        stim("B", 0, 1, -5.0),
        stim("C", 0, 1, 1.0),
        stim("D", 0, 6, 1.5),
        stim("E", 0, 6, -0.5),
    ]
    n = 6
    s = {i: [0.0] for i in ids}
    for t in range(n):
        for i in ids:
            s[i].append(float(lut_py(s[i][t] + stim_at(events, i, t), table)))
    add_case(
        "g0-lut-clamp",
        "lut index = floor(x) clamped to [0, n-1], with negative, fractional and oversized arguments.",
        "next s = lut(s + u, [0,2,1,3]). A (u = 1): 0 -> lut(1) = 2 -> lut(3) = 3 -> lut(4) clamps to 3. B (u = -5 "
        "once): lut(-5) -> 0. C (u = 1 once): 2 -> lut(2) = 1 -> lut(1) = 2 -> oscillates. D (u = 1.5): floor "
        "keeps index 1, then 3.5 -> 3. E (u = -0.5): floor = -1 -> index 0.",
        [
            "wrl 0.1",
            "tier G0",
            "state s : 1 = 0",
            "next s = lut(add(s, stimulus), [0, 2, 1, 3])",
            "observe identity_v1(s)",
        ],
        sim_input(1.0, n, ids, stimulus=events),
        traces({"s": rows(s, ids)}, abs_tol=0),
    )


def case_g0_count_delay():
    ids = ["A", "B", "C"]
    edges = [chem("A", "B", 1.0, delay=2), chem("A", "C", 0.5, delay=0), chem("B", "C", 3.0, sign=-1, delay=1)]
    events = [stim("A", 0, 1), stim("A", 4, 5)]
    init = {"A": 0, "B": 1, "C": 0}
    table = [0, 1]
    n = 12
    s = {i: [init[i]] for i in ids}
    for t in range(n):
        new = {}
        for i in ids:
            count = 0
            for e in edges:
                if e["post"] == i and hist(s[e["pre"]], t - e["delay"]) == 1:
                    count += 1
            new[i] = lut_py(count + stim_at(events, i, t), table)
        for i in ids:
            s[i].append(new[i])
    add_case(
        "g0-count-delay",
        "count_in reads delayed history of any sign and weight; B starts live so the initial state is read for t < 0.",
        "next s = lut(count_in(s,1) + u, [0,1]) (clamped index, so any positive count gives 1). B(t+1) = [A(t-2) == "
        "1]; C(t+1) = [[A(t)==1] + [B(t-1)==1] + u >= 1]. B = 1 initially, so C(1) = 1 via B(-1) = B(0) = 1. "
        "Weights and the inhibitory sign do not affect count_in.",
        [
            "wrl 0.1",
            "tier G0",
            "state s : 1 = 0",
            "next s = lut(add(count_in(s, 1), stimulus), [0, 1])",
            "observe identity_v1(s)",
        ],
        sim_input(1.0, n, ids, edges, stimulus=events, initial_state={"s": {"B": 1}}),
        traces({"s": rows(s, ids)}, abs_tol=0),
    )


# ---------------------------------------------------------------------------------------------------------------
# 28-30: type masks, operators, transcendental functions
# ---------------------------------------------------------------------------------------------------------------


def case_type_mask_mixed():
    ids = ["n0", "n1", "n2", "n3", "n4", "n5"]
    types = ["a", "b", "a", "b", None, "A"]  # None -> default type "generic"; "A" != "a" (exact match)
    gain = {"n0": 2.0, "n1": 3.0, "n2": 2.0, "n3": 3.0, "n4": 0.0, "n5": 0.0}
    events = [stim(i, 1, 3) for i in ids]
    n = 5
    series = {i: [0.0] for i in ids}
    for t in range(n):
        for i in ids:
            series[i].append(gain[i] * stim_at(events, i, t))
    observed = ["n1", "n2", "n3", "n5"]
    add_case(
        "type-mask-mixed-identities",
        "Neurons of types a, b, generic and A; the program branches on type_mask; only some neurons are reported.",
        "target = select(type_mask(a), 2 u, 3 type_mask(b) u) gives gain 2 for type a, 3 for b and 0 for generic and "
        "for 'A' (exact, case-sensitive match). With u = 1 on ticks 1..2 and dt = tau = 1: v(t+1) = gain u(t).",
        [
            *ASSIGN_PREAMBLE,
            "next v = euler_leak(v, select(type_mask(a), 2 * u, 3 * type_mask(b) * u), tau)",
            "observe identity_v1(v)",
        ],
        sim_input(1.0, n, ids, stimulus=events, types=types, observed=observed),
        traces({"v": rows(series, observed)}, abs_tol=0),
    )


def case_threshold_select_clamp():
    ticks = 8
    events = [stim("A", 0, ticks, -2.5)] + [stim("A", k, ticks, 0.5) for k in range(ticks)]
    u = [stim_at(events, "A", t) for t in range(ticks)]  # -2, -1.5, ..., 1.5
    fns = {
        "thr": lambda x: 1.0 if x >= 0.5 else 0.0,
        "sel": lambda x: x if x >= 0.0 else -x,
        "clp": lambda x: min(max(x, -1.0), 1.0),
        "mn": lambda x: min(x, 0.5),
        "mx": lambda x: max(x, -0.5),
        "ab": abs,
        "rl": lambda x: max(x, 0.0),
    }
    reg = {k: [0.0] + [f(x) for x in u] for k, f in fns.items()}
    reg["out"] = [0.0] + [sum(reg[k][t] for k in fns) for t in range(ticks)]
    program = [
        *ASSIGN_HEAD,
        *ASSIGN_TAIL,
        "state thr : 1 = 0",
        "state sel : 1 = 0",
        "state clp : 1 = 0",
        "state mn : 1 = 0",
        "state mx : 1 = 0",
        "state ab : 1 = 0",
        "state rl : 1 = 0",
        "state out : 1 = 0",
        "next thr = euler_leak(thr, threshold(u, 0.5), tau)",
        "next sel = euler_leak(sel, select(threshold(u, 0), u, -u), tau)",
        "next clp = euler_leak(clp, clamp(u, -1, 1), tau)",
        "next mn = euler_leak(mn, min(u, 0.5), tau)",
        "next mx = euler_leak(mx, max(u, -0.5), tau)",
        "next ab = euler_leak(ab, abs(u), tau)",
        "next rl = euler_leak(rl, relu(u), tau)",
        "next out = euler_leak(out, thr + sel + clp + mn + mx + ab + rl, tau)",
        "observe identity_v1(out)",
    ]
    add_case(
        "threshold-select-clamp",
        "threshold, select, clamp, min, max, abs and relu on a stimulus ramp u = -2, -1.5, ..., 1.5.",
        "dt = tau = 1, so each register is its target one tick later: thr = [u >= 0.5], sel = |u| (select on "
        "[u >= 0], including the boundary u = 0), clp = clamp(u,-1,1), mn = min(u,0.5), mx = max(u,-0.5), ab = |u|, "
        "rl = max(u,0); out(t+1) is the sum of the others at tick t. All values are dyadic (exact).",
        program,
        sim_input(1.0, ticks, ["A"], stimulus=events),
        traces({k: [[x] for x in v] for k, v in reg.items()}, abs_tol=0),
    )


def case_tanh_sigmoid():
    ids = ["T0", "T1", "T2", "T3", "T4"]
    amps = [-3.0, 0.0, 0.7, 2.5]
    events = [stim(i, 0, 6, a) for i, a in zip(ids, amps, strict=False)]
    events += [stim("T4", 0, 2, 1.5), stim("T4", 2, 4, -1.5)]
    n = 6
    ser = {name: {i: [0.0] for i in ids} for name in ("y", "w", "out")}
    for t in range(n):
        for i in ids:
            x = stim_at(events, i, t)
            ser["y"][i].append(math.tanh(x))
            ser["w"][i].append(1.0 / (1.0 + math.exp(-x)))
            ser["out"][i].append(ser["y"][i][t] + ser["w"][i][t])
    add_case(
        "tanh-sigmoid",
        "tanh and sigmoid of the stimulus; their sum is read one tick later.",
        "dt = tau = 1 so y(t+1) = tanh(u(t)), w(t+1) = 1/(1 + exp(-u(t))) and out(t+1) = y(t) + w(t); tolerance "
        "1e-12 covers libm differences. Includes u = 0 (tanh 0 = 0, sigmoid 0 = 0.5) and a sign change.",
        [
            *ASSIGN_HEAD,
            *ASSIGN_TAIL,
            "state y : 1 = 0",
            "state w : 1 = 0",
            "state out : 1 = 0",
            "next y = euler_leak(y, tanh(u), tau)",
            "next w = euler_leak(w, sigmoid(u), tau)",
            "next out = euler_leak(out, y + w, tau)",
            "observe identity_v1(out)",
        ],
        sim_input(1.0, n, ids, stimulus=events),
        traces({name: rows(ser[name], ids) for name in ("y", "w", "out")}),
    )


# ---------------------------------------------------------------------------------------------------------------
# 31-33: parameters, dead code, zero steps
# ---------------------------------------------------------------------------------------------------------------


def case_param_override():
    dt, n = 0.1, 15
    tau, gain = 0.6, 2.0  # tau overridden by the input, gain = (0 + 4) / 2 default
    v = [gain * (1 - math.exp(-k * dt / tau)) for k in range(n + 1)]
    add_case(
        "param-override",
        "A trainable parameter overridden by the simulation input and one left at the default (lo + hi) / 2.",
        "tau is overridden to 0.6 (the program default is 0.2); gain has no declared value so it is (0 + 4)/2 = 2. "
        "Constant stimulus 1 gives v_n = 2 (1 - exp(-n dt / 0.6)).",
        [
            "wrl 0.1",
            "tier G1",
            "state v : 1 = 0",
            "param tau : s = 0.2 in [0.05, 1] trainable bits 8",
            "param gain : 1 in [0, 4] trainable bits 6",
            "next v = leaky_integrate(v, gain * stimulus, tau)",
            "observe identity_v1(v)",
        ],
        sim_input(dt, n, ["A"], stimulus=[stim("A", 0, n)], params={"tau": tau}),
        traces({"v": [[x] for x in v]}),
    )


def case_dead_code_names_ignored():
    dt, tau, n = 0.1, 0.3, 10
    v = [1 - math.exp(-k * dt / tau) for k in range(n + 1)]
    add_case(
        "dead-code-names-ignored",
        "Registers and parameters removed as dead code may still be named in the input and are ignored.",
        "Only v reaches the observation; d and `unused` are eliminated. Their input overrides are accepted and "
        "have no effect, so v_n = 1 - exp(-n dt / tau) with tau = 0.3.",
        [
            "wrl 0.1",
            "tier G1",
            "state v : 1 = 0",
            "state d : 1 = 0.5",
            "param tau : s = 0.3 fixed",
            "param unused : 1 = 1 in [0, 2] trainable bits 4",
            "next v = leaky_integrate(v, stimulus, tau)",
            "next d = d + stimulus * unused",
            "observe identity_v1(v)",
        ],
        sim_input(
            dt,
            n,
            ["A"],
            stimulus=[stim("A", 0, n)],
            params={"unused": 1.5},
            initial_state={"d": {"A": 2.0}},
        ),
        traces({"v": [[x] for x in v]}),
    )


def case_n_steps_zero():
    add_case(
        "n-steps-zero",
        "n_steps = 0 samples only the initial state; the calcium observation starts at the register (steady state).",
        "No step is taken: tick 0 reports the initial registers (0.75, with B overridden to 0.25) and c(0) = x(0).",
        [
            "wrl 0.1",
            "tier G1",
            "state v : 1 = 0.75",
            "param tau : s = 0.5 fixed",
            "next v = decay(v, tau)",
            "observe calcium_linear_v1(v, 0.5[s])",
        ],
        sim_input(0.1, 0, ["A", "B"], initial_state={"v": {"B": 0.25}}),
        traces({"v": [[0.75, 0.25]]}, [[0.75, 0.25]], abs_tol=0),
    )


# ---------------------------------------------------------------------------------------------------------------
# 34-35: permutation equivariance
# ---------------------------------------------------------------------------------------------------------------


def case_permutation_g1():
    ids = [f"N{i}" for i in range(6)]
    types = ["x", "y", "x", "z", "y", "x"]
    edges = [
        chem("N0", "N1", 0.8),
        chem("N0", "N2", 0.3, delay=2),
        chem("N1", "N3", 1.2, sign=-1),
        chem("N2", "N3", 0.6, delay=1),
        chem("N3", "N4", 0.9),
        chem("N3", "N4", 0.4, sign=-1, delay=3),
        chem("N4", "N5", 1.1, delay=1),
        chem("N5", "N0", 0.5, sign=-1),
        chem("N5", "N2", 0.7, delay=2),
    ]
    gaps = [gapj("N0", "N3", 1.5), gapj("N1", "N4", 0.4), gapj("N2", "N5", 2.5), gapj("N4", "N5", 0.8)]
    events = [stim("N0", 0, 5, 1.2), stim("N3", 2, 4, -0.8), stim("N5", 6, 9, 0.9)]
    add_case(
        "permutation-equivariance",
        "Mixed-sign, delayed chemical edges, gap junctions, types and a calcium observation; neurons reordered.",
        "Relabelling the neuron order must not change any per-neuron output beyond summation-order rounding "
        "(section 7, check type permutation). No numeric expectation is needed: the invariance is the property.",
        [
            "wrl 0.1",
            "tier G1",
            "state v : 1 = 0",
            "param tau : s = 0.3 in [0.1, 2] trainable bits 8",
            "param k : 1 = 1.5 in [0, 4] trainable bits 8",
            "input u = stimulus",
            "input e = sum_in(v, exc)",
            "input h = sum_in(v, inh)",
            "input m = type_mask(x)",
            "next v = leaky_integrate(v, tanh(u + e - h + m + 0.25 * delay(v, 2)), tau)",
            "gap v scale k",
            "observe calcium_linear_v1(v, tau)",
        ],
        sim_input(
            0.1,
            12,
            ids,
            edges,
            gaps,
            events,
            types=types,
            initial_state={"v": {"N1": 0.3, "N4": -0.2}},
        ),
        {"type": "permutation", "permutation": [3, 0, 5, 1, 4, 2], "tolerance": {"abs": 1e-12, "rel": 0}},
    )


def case_permutation_g0():
    m = 8
    ids = [f"N{i}" for i in range(m)]
    edges = [chem(ids[i], ids[(i + d) % m]) for i in range(m) for d in (-1, 1)]
    add_case(
        "permutation-g0-ring",
        "Rule 90 on a ring with neurons listed in a scrambled order.",
        "Integer dynamics are order independent: every output of the reordered neuron equals the original's "
        "exactly (zero tolerance).",
        RULE90,
        sim_input(1.0, 8, ids, edges, initial_state={"s": {"N3": 1, "N4": 1}}),
        {"type": "permutation", "permutation": [5, 2, 7, 0, 3, 6, 1, 4], "tolerance": {"abs": 0, "rel": 0}},
    )


# ---------------------------------------------------------------------------------------------------------------
# 36-37: convergence
# ---------------------------------------------------------------------------------------------------------------


def case_convergence_euler():
    tau, t_end, dts = 0.5, 1.0, [0.1, 0.05, 0.025]
    exact = 1 - math.exp(-t_end / tau)
    errs = []
    for dt in dts:
        steps = round(t_end / dt)
        errs.append(abs((1 - (1 - dt / tau) ** steps) - exact))
    ratio = max(errs[k + 1] / errs[k] for k in range(len(errs) - 1))
    assert ratio <= 0.6 and errs[-1] <= 0.01, (ratio, errs)
    add_case(
        "convergence-euler",
        "Forward Euler for dx/dt = (1 - x)/tau is first order: halving dt roughly halves the error.",
        "The iterate is 1 - (1 - dt/tau)^n; the analytic value at t = 1 is 1 - exp(-t/tau) with tau = 0.5. The "
        "script verified the error ratios (about 0.5) against max_ratio 0.6 and the finest error (about 0.007).",
        [
            "wrl 0.1",
            "tier G1",
            "dt_max 0.1",
            "state v : 1 = 0",
            "param tau : s = 0.5 fixed",
            "next v = euler_leak(v, stimulus, tau)",
            "observe identity_v1(v)",
        ],
        sim_input(0.1, 10, ["A"], stimulus=[stim("A", 0, 1000)]),
        {
            "type": "convergence",
            "dts": dts,
            "time": t_end,
            "register": "v",
            "neuron": "A",
            "expected": exact,
            "max_ratio": 0.6,
            "max_error_finest": 0.01,
        },
    )


def case_convergence_gap():
    g, t_end, dts = 2.0, 1.0, [0.1, 0.05, 0.025]
    exact = 0.5 * (1 + math.exp(-2 * g * t_end))
    errs = []
    for dt in dts:
        steps = round(t_end / dt)
        r = dt * g
        d = ((1 - r) / (1 + r)) ** steps  # a - b contracts by (1 - r)/(1 + r) per step; a + b is conserved
        errs.append(abs(0.5 * (1 + d) - exact))
    ratio = max(errs[k + 1] / errs[k] for k in range(len(errs) - 1))
    assert ratio <= 0.35 and errs[-1] <= 0.001, (ratio, errs)
    add_case(
        "convergence-gap",
        "Two-neuron gap diffusion converges to v_A(t) = 0.5 (1 + exp(-2 g t)) as dt shrinks.",
        "With identity local dynamics the scheme keeps a + b = 1 and multiplies a - b by (1 - dt g)/(1 + dt g) per "
        "step, so a_n = 0.5 (1 + ((1-r)/(1+r))^n). The script computed the errors for dts 0.1, 0.05, 0.025 (ratios "
        "about 0.25, the scheme is better than first order here) and checked max_ratio and max_error_finest.",
        GAP_HOLD,
        sim_input(0.1, 10, ["A", "B"], gaps=[gapj("A", "B", g)], initial_state={"v": {"A": 1.0}}),
        {
            "type": "convergence",
            "dts": dts,
            "time": t_end,
            "register": "v",
            "neuron": "A",
            "expected": exact,
            "max_ratio": 0.35,
            "max_error_finest": 0.001,
        },
    )


# ---------------------------------------------------------------------------------------------------------------
# 38-40: graph corner cases
# ---------------------------------------------------------------------------------------------------------------


def case_multigraph_chemical():
    ids = ["A", "B"]
    edges = [
        chem("A", "B", 0.125, sign=-1, delay=2),
        chem("A", "B", 0.5),
        chem("A", "B", 1.0, sign=-1, delay=3),
        chem("A", "B", 0.5),
        chem("A", "B", 0.25, delay=1),
        chem("A", "B", 0.125, sign=-1),
    ]
    events = [stim("A", 0, 1, 2.0), stim("A", 3, 4, 1.0)]
    n = 10
    s = assign_net(ids, edges, events, n, k_exc=1.0, k_inh=-1.0)
    add_case(
        "multigraph-chemical",
        "Six parallel A->B edges (identical duplicates, different delays and signs), listed unsorted.",
        "Parallel edges are independent terms. B(t+1) = 1.0 A(t) + 0.25 A(t-1) - 0.125 A(t) - 0.125 A(t-2) - A(t-3) "
        "(with the duplicated 0.5 edge counted twice), where A(t+1) = u_A(t); all dyadic, exact.",
        ASSIGN_EXC_INH,
        sim_input(1.0, n, ids, edges, stimulus=events),
        traces({"v": rows(s, ids)}, abs_tol=0),
    )


def case_self_loop():
    ids = ["A", "B"]
    edges = [chem("A", "A", 0.5, delay=0), chem("B", "B", 1.0, delay=2)]
    events = [stim("A", 0, 1), stim("B", 0, 1)]
    n = 10
    s = assign_net(ids, edges, events, n)
    add_case(
        "self-loop",
        "Chemical self-edges with delay 0 and delay 2.",
        "A(t+1) = u(t) + 0.5 A(t): an impulse decays as 1, .5, .25, ... B(t+1) = u(t) + B(t-2): the impulse "
        "recirculates with period 3 (1 at ticks 1, 4, 7, 10).",
        ASSIGN_EXC,
        sim_input(1.0, n, ids, edges, stimulus=events),
        traces({"v": rows(s, ids)}, abs_tol=0),
    )


def case_no_edges():
    ids = ["A", "B", "C"]
    events = [stim("A", 0, 5, 1.0)]
    n = 6
    s = assign_net(ids, [], events, n)
    add_case(
        "no-edges",
        "No chemical or gap edges at all: nothing propagates and the gap stage leaves values unchanged.",
        "Without edges sum_in is 0 and G = S = 0, so a neuron only sees its own stimulus: A(t+1) = u_A(t) and "
        "B = C = 0 forever.",
        [*ASSIGN_EXC[:-2], "next v = euler_leak(v, u + e, tau)", "gap v", "observe identity_v1(v)"],
        sim_input(1.0, n, ids, stimulus=events),
        traces({"v": rows(s, ids)}, abs_tol=0),
    )


# ---------------------------------------------------------------------------------------------------------------
# 41-: compile errors
# ---------------------------------------------------------------------------------------------------------------


def compile_error_cases():
    head = ["wrl 0.1", "tier G1", "state v : 1 = 0"]
    tail = ["observe identity_v1(v)"]
    compile_error(
        "error-unit-add",
        "Adding a time constant (seconds) to a dimensionless register.",
        [*head, "param tau : s = 1 fixed", "next v = v + tau", *tail],
        "E_UNIT",
    )
    compile_error(
        "error-unit-next",
        "A register update whose unit differs from the register unit.",
        [*head, "param tau : s = 1 fixed", "next v = tau", *tail],
        "E_UNIT",
    )
    compile_error(
        "error-stability-bound",
        "euler_leak with dt_max / tau = 2 > 1.",
        ["wrl 0.1", "tier G1", "dt_max 1", "state v : 1 = 0", "param tau : s = 0.5 fixed"]
        + ["next v = euler_leak(v, stimulus, tau)", *tail],
        "E_STABILITY",
    )
    compile_error(
        "error-stability-no-dt-max",
        "euler_leak without any dt_max declaration.",
        [*head, "param tau : s = 0.5 fixed", "next v = euler_leak(v, stimulus, tau)", *tail],
        "E_STABILITY",
    )
    compile_error(
        "error-syntax-division",
        "Division is not part of the language.",
        [*head, "next v = v / 2", *tail],
        "E_SYNTAX",
    )
    compile_error(
        "error-tier-g0-relu",
        "relu is a G1 operator and not allowed in a G0 program.",
        ["wrl 0.1", "tier G0", "state s : 1 = 0", "next s = relu(s)", "observe identity_v1(s)"],
        "E_TIER",
    )
    compile_error(
        "error-tier-g1-lut",
        "lut is G0 only.",
        [*head, "next v = lut(v, [0, 1])", *tail],
        "E_TIER",
    )
    compile_error(
        "error-tier-g0-gap",
        "gap is not available in G0.",
        ["wrl 0.1", "tier G0", "state s : 1 = 0", "gap s", "observe identity_v1(s)"],
        "E_TIER",
    )
    compile_error(
        "error-name-unknown",
        "Use of an undefined name.",
        [*head, "next v = w + 1", *tail],
        "E_NAME",
    )
    compile_error(
        "error-name-missing-observe",
        "Exactly one observe statement is required.",
        [*head, "next v = v"],
        "E_NAME",
    )
    compile_error(
        "error-type-arity",
        "abs takes exactly one argument.",
        [*head, "next v = abs(v, v)", *tail],
        "E_TYPE",
    )
    compile_error(
        "error-limit-identifier",
        "Identifiers are limited to 64 characters.",
        ["wrl 0.1", "tier G1", "state " + "x" * 65 + " : 1 = 0", "observe identity_v1(" + "x" * 65 + ")"],
        "E_LIMIT",
    )


# ---------------------------------------------------------------------------------------------------------------
# 53: modulatory edges
# ---------------------------------------------------------------------------------------------------------------


def modl(pre, post, weight=1.0):
    return {"pre": pre, "post": post, "weight": float(weight)}


def case_modulatory_sum():
    ids = ["A", "B", "C", "D"]
    edges = [chem("A", "B", 1.0), chem("B", "C", 0.5, sign=-1, delay=1)]
    mods = [modl("A", "C", 0.5), modl("A", "C", 0.25), modl("D", "B", 1.0), modl("C", "C", 0.5), modl("B", "D", 2.0)]
    events = [stim("A", 0, 3), stim("D", 1, 2, 0.5)]
    n = 8
    series = {i: [0.0] for i in ids}
    for t in range(n):
        new = {}
        for i in ids:
            chem_all = 0.0
            for e in edges:
                if e["post"] == i:
                    term = e["weight"] * hist(series[e["pre"]], t - e["delay"])
                    chem_all += term if e["sign"] > 0 else -term
            mod = sum(m["weight"] * series[m["pre"]][t] for m in mods if m["post"] == i)
            new[i] = stim_at(events, i, t) + chem_all - 0.5 * mod
        for i in ids:
            series[i].append(new[i])
    inp = sim_input(1.0, n, ids, edges, stimulus=events)
    inp["graph"]["modulatory"] = mods
    add_case(
        "modulatory-sum",
        "sum_in(v, mod) sums the unsigned, undelayed modulatory edges, separately from the chemical edges.",
        "v(t+1) = u(t) + (signed chemical sum on tick-t history, B->C has delay 1) - 0.5 (sum of modulatory "
        "weight * v_pre(t)). Modulatory edges carry no sign or delay, may repeat (A->C twice) and may be self-loops "
        "(C->C); sum_in(v, all) never reads them. All numbers are dyadic, so the result is exact.",
        [
            *ASSIGN_PREAMBLE,
            "input a = sum_in(v, all)",
            "input m = sum_in(v, mod)",
            "next v = euler_leak(v, u + a - 0.5 * m, tau)",
            "observe identity_v1(v)",
        ],
        inp,
        traces({"v": rows(series, ids)}, abs_tol=0),
    )


# ---------------------------------------------------------------------------------------------------------------


def build_all():
    case_leaky_const_drive()
    case_euler_const_drive()
    case_decay_zero_stimulus()
    case_chemical_excitation()
    case_chemical_inhibition()
    case_chemical_signed_sum()
    case_gap_diffusion()
    case_gap_dyadic_exact()
    case_gap_constant_equilibrium()
    case_gap_equilibrium_scale()
    case_gap_large_step_bounded()
    case_gap_local_dynamics()
    case_gap_multigraph()
    case_delay_ring_wraparound()
    case_history_init()
    case_delay_op_own_register()
    case_adaptation_register()
    case_adaptation_exponential()
    case_calcium_impulse()
    case_calcium_step()
    case_stimulus_variants()
    case_edge_deletion()
    case_gap_edge_deletion()
    case_g0_rule90()
    case_g0_own_state_table()
    case_g0_lut_clamp()
    case_g0_count_delay()
    case_type_mask_mixed()
    case_threshold_select_clamp()
    case_tanh_sigmoid()
    case_param_override()
    case_dead_code_names_ignored()
    case_n_steps_zero()
    case_permutation_g1()
    case_permutation_g0()
    case_convergence_euler()
    case_convergence_gap()
    case_multigraph_chemical()
    case_self_loop()
    case_no_edges()
    compile_error_cases()
    case_modulatory_sum()


def main():
    build_all()
    names = [c["name"] for c in CASES]
    assert len(set(names)) == len(names)
    assert len(CASES) < 100
    wanted = {f"{i:02d}-{c['name']}.json" for i, c in enumerate(CASES, start=1)}
    for old in OUT_DIR.glob("[0-9][0-9]-*.json"):
        if old.name not in wanted:
            old.unlink()
    for i, case in enumerate(CASES, start=1):
        (OUT_DIR / f"{i:02d}-{case['name']}.json").write_text(dump(case) + "\n", encoding="utf-8")
    print(f"wrote {len(CASES)} cases to {OUT_DIR}")


if __name__ == "__main__":
    main()
