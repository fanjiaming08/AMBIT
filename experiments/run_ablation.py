"""Module ablation (Figure 7): fixed-budget compliance error and non-discreteness on a set of
instances, and the iteration count to convergence on the MBB beam.

Example:
    python experiments/run_ablation.py --count 100 --seed 1 --out results/ablation
"""

import argparse
import os

import numpy as np

from common import ABLATIONS, ensure_dir, make_ablation, make_model, write_csv
from run_dataset import full_field, reference

from ambit import mbb
from ambit.metrics import compliance_error, is_connected
from ambit.problems import random_instances


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--count", type=int, default=100)
    ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--budget", type=int, default=60)
    ap.add_argument("--ref-iter", type=int, default=1000)
    ap.add_argument("--mbb", type=int, nargs=2, default=[300, 100])
    ap.add_argument("--max-iter", type=int, default=500)
    ap.add_argument("--configs", nargs="+", default=list(ABLATIONS) + ["SiMPL"])
    ap.add_argument("--reference-dir", default=None,
                    help="reference cache shared with run_dataset.py (default: <out>/references)")
    ap.add_argument("--out", default="results/ablation")
    args = ap.parse_args()
    ensure_dir(args.out)
    cache = ensure_dir(args.reference_dir or os.path.join(args.out, "references"))

    problems = random_instances(args.count, n=args.size, seed=args.seed)
    models = [make_model(p) for p in problems]
    refs = [reference(p, m, cache, args.ref_iter)[0] for p, m in zip(problems, models)]
    mbb_problem = mbb(*args.mbb)
    mbb_model = make_model(mbb_problem)

    rows = []
    for label in args.configs:
        deterministic = label in ("SiMPL", "Module I only", "Module I only, kappa0=0")
        seeds = [args.seeds[0]] if deterministic else args.seeds
        ce_seed, mnd_seed, vfe_seed, fail_seed = [], [], [], []
        for seed in seeds:
            ce, mnd, vfe, fail = [], [], [], []
            for p, m, c_ref in zip(problems, models, refs):
                if not np.isfinite(c_ref):
                    continue
                res = make_ablation(label, seed).run(m, max_iter=10 * args.budget, max_analyses=args.budget)
                failed = not is_connected(p, full_field(m, res.rho_bar))
                fail.append(failed)
                if not failed:
                    ce.append(compliance_error(res.compliance, c_ref))
                mnd.append(res.mnd)
                vfe.append(res.vfe)
            ce_seed.append(float(np.mean(ce)) if ce else float("nan"))
            mnd_seed.append(np.mean(mnd))
            vfe_seed.append(np.mean(vfe))
            fail_seed.append(100 * np.mean(fail))
        it = make_ablation(label, args.seeds[0]).run(mbb_model, max_iter=args.max_iter).iterations
        ce_mean = float(np.nanmean(ce_seed)) if np.isfinite(ce_seed).any() else float("nan")
        row = dict(config=label, ce_pct=ce_mean, mnd_pct=float(np.mean(mnd_seed)),
                   vfe_pct=float(np.mean(vfe_seed)), failure_pct=float(np.mean(fail_seed)),
                   mbb_iterations=it, n_seeds=len(seeds))
        rows.append(row)
        print(f"{label:28s} CE={row['ce_pct']:6.2f}%  Mnd={row['mnd_pct']:5.2f}%  "
              f"VFE={row['vfe_pct']:.3f}%  fail={row['failure_pct']:4.1f}%  MBB it={it}", flush=True)
    write_csv(os.path.join(args.out, "ablation.csv"), rows)


if __name__ == "__main__":
    main()
