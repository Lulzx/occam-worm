"""``python -m occamworm.baselines {benchmark,summarize,control}``."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from occamworm.sources.cli import find_root

# One thread per worker process: fold-level parallelism only (set before numpy/jax load in the workers).
for _var in ("VECLIB_MAXIMUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m occamworm.baselines")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("benchmark", help="nested evaluation of the baselines on the frozen split")
    b.add_argument("--out", type=Path, help="default: artifacts/baselines-v1")
    b.add_argument("--tasks", nargs="+", default=["T2a", "T2b"])
    b.add_argument("--history", nargs="+", default=["no", "yes"], choices=["no", "yes"])
    b.add_argument("--families", nargs="+", help="default: the experiment config")
    b.add_argument("--folds", type=int, nargs="+", help="outer fold indices (default: all)")
    b.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    s = sub.add_parser("summarize", help="paired animal-level comparisons and report from folds.jsonl")
    s.add_argument("--out", type=Path, help="default: artifacts/baselines-v1")
    c = sub.add_parser("control", help="indicator-kinetics control and false-sharing rate (§4.5)")
    c.add_argument("--out", type=Path, help="default: artifacts/baselines-v1")
    c.add_argument("--replicates", type=int, help="default: the experiment config")
    args = ap.parse_args(argv)
    root = find_root(Path.cwd())
    out = args.out or root / "artifacts" / "baselines-v1"

    if args.cmd == "benchmark":
        from occamworm.baselines.benchmark import Config, load_experiment, run

        families = tuple(args.families or load_experiment(root)["models"]["families"])
        configs = [Config(t, h == "yes") for t in args.tasks for h in args.history]
        run(root, out, configs, families, args.workers, args.folds)
        return 0
    if args.cmd == "summarize":
        from occamworm.baselines.summary import summarize

        summarize(root, out)
        return 0
    from occamworm.baselines.control import run_control

    run_control(root, out, args.replicates)
    return 0


if __name__ == "__main__":
    sys.exit(main())
