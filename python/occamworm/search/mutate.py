"""S1 local structured mutation over an enumerated program set (§7.2; OW-012).

A program of the canonical enumeration text (one register, ``let nK = ...`` lines) is parsed to an expression tree.
One mutation is one local edit: replace a leaf, swap an operator for one of the same arity, insert an operator
above a subtree, or remove one. Leaves and operators come only from the vocabulary of the declared search config,
so a proposal is always an expression of the declared ops. A proposal is kept only if the compiler accepts it and
its canonical hash is a member of the enumerated set, which is exactly the declared grammar and AST budget (the
enumeration applied the node, depth, parameter and stability limits). Canonical dedup is by program hash: a
``Mutator`` never proposes a hash in its ``seen`` set.

Only single-register programs are supported (``max_registers == 1``).
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from occamworm.sim.ir import CompilerError, compile_source

Tree = str | tuple[str, tuple["Tree", ...]]  # a leaf is its canonical text; an operator is (name, children)

ARITY = {"neg": 1, "relu": 1, "tanh": 1, "add": 2, "mul": 2, "leaky_integrate": 3, "euler_leak": 3}
SAME_ARITY = (("neg", "relu", "tanh"), ("add", "mul"), ("leaky_integrate", "euler_leak"))
TAU_SLOT = {"leaky_integrate": 2, "euler_leak": 2}
LEAF_OPS = ("state", "stimulus", "sum_in", "param")
_CALL = re.compile(r"^(\w+)\((.*)\)$")
_PARAM_LEAF = re.compile(r"^param\((.*)\)$")


def _split_args(text: str) -> list[str]:
    out, depth, cur = [], 0, ""
    for ch in text:
        if ch == "," and depth == 0:
            out.append(cur.strip())
            cur = ""
            continue
        depth += ch in "(["
        depth -= ch in ")]"
        cur += ch
    if cur.strip():
        out.append(cur.strip())
    return out


def parse(source: str) -> Tree:
    """Expression tree of a canonical enumeration source. Parameter leaves keep their declaration tail."""
    params: dict[str, str] = {}
    lets: dict[str, str] = {}
    root = ""
    for line in source.splitlines():
        if line.startswith("param "):
            name, tail = line[len("param ") :].split(" : ", 1)
            params[name] = tail
        elif line.startswith("let "):
            name, expr = line[len("let ") :].split(" = ", 1)
            lets[name] = expr
        elif line.startswith("next "):
            root = line.split(" = ", 1)[1]
        elif line.startswith("state ") and not line.startswith("state s0 :"):
            raise NotImplementedError("mutation supports one register only")

    def build(name: str) -> Tree:
        expr = lets[name]
        m = _CALL.match(expr)
        if m is None:
            return expr  # a constant such as ``1e0[1]``
        op, args = m.group(1), _split_args(m.group(2))
        if op == "param":
            return f"param({params[args[0]]})"
        if op in LEAF_OPS:
            return expr
        nary = op in ("add", "mul") and len(args) >= 2  # the canonical form flattens nested sums and products
        if op not in ARITY or (len(args) != ARITY[op] and not nary):
            raise NotImplementedError(f"operator {op!r} is outside the mutation vocabulary")
        return (op, tuple(build(a) for a in args))

    return build(root)


def render(tree: Tree, header: str) -> str:
    """WRL source for a tree; ``header`` is the program preamble up to and including the ``state`` line."""
    names: dict[str, str] = {}

    def expr(t: Tree) -> str:
        if isinstance(t, str):
            m = _PARAM_LEAF.match(t)
            if m:
                return f"param({names.setdefault(m.group(1), f'p{len(names)}')})"
            return t
        return f"{t[0]}({', '.join(expr(c) for c in t[1])})"

    body = expr(tree)
    decls = "".join(f"param {n} : {spec}\n" for spec, n in names.items())
    return f"{header}{decls}let n0 = {body}\nnext s0 = n0\nobserve identity_v1(s0)\n"


def header_of(source: str) -> str:
    lines = source.splitlines(keepends=True)
    cut = next(i for i, ln in enumerate(lines) if ln.startswith("state "))
    return "".join(lines[: cut + 1])


def _paths(tree: Tree, here: tuple[int, ...] = ()) -> list[tuple[int, ...]]:
    out = [here]
    if not isinstance(tree, str):
        for i, c in enumerate(tree[1]):
            out += _paths(c, (*here, i))
    return out


def _get(tree: Tree, path: tuple[int, ...]) -> Tree:
    for i in path:
        assert not isinstance(tree, str)
        tree = tree[1][i]
    return tree


def _put(tree: Tree, path: tuple[int, ...], new: Tree) -> Tree:
    if not path:
        return new
    assert not isinstance(tree, str)
    kids = list(tree[1])
    kids[path[0]] = _put(kids[path[0]], path[1:], new)
    return (tree[0], tuple(kids))


def _is_tau(path: tuple[int, ...], tree: Tree) -> bool:
    if not path:
        return False
    parent = _get(tree, path[:-1])
    return not isinstance(parent, str) and TAU_SLOT.get(parent[0]) == path[-1]


class Mutator:
    """Proposes in-grammar neighbours of enumerated programs; ``programs`` are ``ow search enumerate`` rows."""

    def __init__(self, config: Mapping[str, Any], programs: Sequence[Mapping[str, Any]]) -> None:
        if int(config["max_registers"]) != 1:
            raise NotImplementedError("mutation supports max_registers == 1 only")
        self.ops = [o for o in config["ops"] if o in ARITY]
        self.index = {p["hash"]: p for p in programs}
        self.header = header_of(programs[0]["source"])
        self.value_leaves: set[str] = set()
        self.tau_leaves: set[str] = set()
        for p in programs:
            self._collect(parse(p["source"]), p["source"], is_tau=False)
        self.value_leaves_l = sorted(self.value_leaves)
        self.tau_leaves_l = sorted(self.tau_leaves)
        self._compiled: dict[str, str | None] = {}

    def _collect(self, t: Tree, source: str, is_tau: bool) -> None:
        if isinstance(t, str):
            (self.tau_leaves if is_tau else self.value_leaves).add(t)
            return
        for i, c in enumerate(t[1]):
            self._collect(c, source, TAU_SLOT.get(t[0]) == i)

    def hash_of(self, tree: Tree) -> str | None:
        """Canonical program hash of a tree, or None if the compiler rejects it."""
        src = render(tree, self.header)
        if src not in self._compiled:
            try:
                self._compiled[src] = compile_source(src).program_hash
            except CompilerError:
                self._compiled[src] = None
        return self._compiled[src]

    def mutate(self, tree: Tree, rng: np.random.Generator) -> Tree | None:
        """One random local edit, or None if the drawn edit does not apply."""
        paths = _paths(tree)
        path = paths[int(rng.integers(len(paths)))]
        node = _get(tree, path)
        kind = ("leaf", "op", "insert", "remove")[int(rng.integers(4))]
        tau = _is_tau(path, tree)
        if kind == "leaf" and isinstance(node, str):
            pool = self.tau_leaves_l if tau else self.value_leaves_l
            return _put(tree, path, pool[int(rng.integers(len(pool)))])
        if kind == "op" and not isinstance(node, str):
            group = next(g for g in SAME_ARITY if node[0] in g)
            options = [o for o in group if o != node[0] and o in self.ops]
            if options:
                return _put(tree, path, (options[int(rng.integers(len(options)))], node[1]))
        if kind == "insert" and not tau:
            op = self.ops[int(rng.integers(len(self.ops)))]
            if ARITY[op] == 1:
                return _put(tree, path, (op, (node,)))
            if ARITY[op] == 2:
                leaf = self.value_leaves_l[int(rng.integers(len(self.value_leaves_l)))]
                return _put(tree, path, (op, (node, leaf) if rng.random() < 0.5 else (leaf, node)))
        if kind == "remove" and not isinstance(node, str) and not tau:
            keep = [c for i, c in enumerate(node[1]) if TAU_SLOT.get(node[0]) != i]
            return _put(tree, path, keep[int(rng.integers(len(keep)))])
        return None

    def propose(
        self,
        parent: str,
        rng: np.random.Generator,
        seen: set[str],
        attempts: int = 50,
        allowed: set[str] | None = None,
    ) -> Mapping[str, Any] | None:
        """An unseen enumerated neighbour of ``parent`` (a program hash), or None after ``attempts`` tries.
        ``allowed`` further restricts proposals to a subset of the enumerated set (e.g. the gated programs)."""
        tree = parse(self.index[parent]["source"])
        for _ in range(attempts):
            child = self.mutate(tree, rng)
            if child is None:
                continue
            h = self.hash_of(child)
            if h is not None and h in self.index and h not in seen and (allowed is None or h in allowed):
                return self.index[h]
        return None
