"""Converged compliance on the four classical two-dimensional benchmarks (Table 6, Figure 2).

Example:
    python experiments/run_benchmarks.py --out results/benchmarks
    python experiments/run_benchmarks.py --scale 0.5 --methods OC MMA AMBIT
"""

import argparse
import os

import numpy as np
from scipy.stats import rankdata

from common import DEFAULT_METHODS, METHODS, ensure_dir, make_model, save_design, seeds_for, write_csv

from ambit.problems import BENCHMARKS, benchmark


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--problems", nargs="+", default=list(BENCHMARKS))
    ap.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    ap.add_argument("--scale", type=float, default=1.0, help="mesh scale relative to Table 3")
    ap.add_argument("--rmin", type=float, default=None, help="filter radius in elements")
    ap.add_argument("--max-iter", type=int, default=500)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--out", default="results/benchmarks")
    args = ap.parse_args()

    ensure_dir(args.out)
    rows = []
    for pname in args.problems:
        problem = benchmark(pname, args.scale)
        model = make_model(problem, args.rmin)
        for mname in args.methods:
            for seed in seeds_for(mname, args.seeds):
                opt = METHODS[mname](seed)
                res = opt.run(model, max_iter=args.max_iter)
                rows.append(dict(problem=pname, nelx=problem.nelx, nely=problem.nely, method=mname,
                                 seed=seed, compliance=res.compliance, iterations=res.iterations,
                                 analyses=res.analyses, time_s=res.time,
                                 time_per_iter_ms=1000 * res.time / max(res.iterations, 1),
                                 vfe_pct=res.vfe, mnd_pct=res.mnd, converged=res.converged))
                print(f"{pname:10s} {mname:9s} seed={seed} C={res.compliance:10.4f} "
                      f"it={res.iterations:4d} an={res.analyses:4d} Mnd={res.mnd:5.2f}% "
                      f"VFE={res.vfe:.3f}% t={res.time:6.1f}s", flush=True)
                save_design(os.path.join(args.out, "designs", f"{pname}_{mname}_s{seed}.npz"),
                            problem, model, res)
    write_csv(os.path.join(args.out, "runs.csv"), rows)

    # seed-averaged summary and average rank over problems
    summary = []
    for pname in args.problems:
        for mname in args.methods:
            sel = [r for r in rows if r["problem"] == pname and r["method"] == mname]
            summary.append(dict(problem=pname, method=mname,
                                compliance=np.mean([r["compliance"] for r in sel]),
                                iterations=np.mean([r["iterations"] for r in sel]),
                                time_per_iter_ms=np.mean([r["time_per_iter_ms"] for r in sel]),
                                time_s=np.mean([r["time_s"] for r in sel])))
    ranks = {m: [] for m in args.methods}
    for pname in args.problems:
        c = [next(s["compliance"] for s in summary if s["problem"] == pname and s["method"] == m)
             for m in args.methods]
        for m, r in zip(args.methods, rankdata(c)):
            ranks[m].append(r)
    for s in summary:
        s["avg_rank"] = float(np.mean(ranks[s["method"]]))
    write_csv(os.path.join(args.out, "summary.csv"), summary)

    print("\nFinal compliance (mean over seeds where applicable)")
    print(f"{'method':10s}" + "".join(f"{p:>12s}" for p in args.problems) + f"{'avg rank':>10s}")
    for m in args.methods:
        vals = [next(s["compliance"] for s in summary if s["problem"] == p and s["method"] == m)
                for p in args.problems]
        print(f"{m:10s}" + "".join(f"{v:12.3f}" for v in vals) + f"{np.mean(ranks[m]):10.2f}")


if __name__ == "__main__":
    main()
