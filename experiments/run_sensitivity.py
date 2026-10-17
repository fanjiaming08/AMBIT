"""One-at-a-time hyperparameter variation about the defaults of AMBIT (Table 9).

Example:
    python experiments/run_sensitivity.py --count 100 --seed 1 --out results/sensitivity
"""

import argparse
import os

import numpy as np

from common import ensure_dir, make_model, write_csv
from run_dataset import full_field, reference

from ambit import AMBIT
from ambit.metrics import compliance_error, is_connected
from ambit.problems import random_instances

GRID = {
    "beta1": [("0.50", dict(beta1=0.50)), ("0.70", dict(beta1=0.70)),
              ("0.90", dict(beta1=0.90)), ("0.95", dict(beta1=0.95))],
    "beta2": [("0.900", dict(beta2=0.900, adaptive_beta2=False)),
              ("0.990", dict(beta2=0.990, adaptive_beta2=False)),
              ("0.999", dict(beta2=0.999, adaptive_beta2=False)),
              ("adaptive", dict())],
    "gamma": [("fixed 0.00", dict(fixed_gamma=0.0)), ("fixed 0.25", dict(fixed_gamma=0.25)),
              ("fixed 0.50", dict(fixed_gamma=0.5)), ("adaptive", dict())],
    "eta0": [("0.25", dict(eta0=0.25)), ("0.5", dict(eta0=0.5)),
             ("1.0", dict(eta0=1.0)), ("2.0", dict(eta0=2.0))],
    "alpha": [("0.5", dict(alpha=0.5)), ("1.0", dict(alpha=1.0)),
              ("1.5", dict(alpha=1.5)), ("2.5", dict(alpha=2.5))],
    "kappa_min": [("0.01", dict(kappa_min=0.01)), ("0.05", dict(kappa_min=0.05)),
                  ("0.10", dict(kappa_min=0.10)), ("0.20", dict(kappa_min=0.20))],
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--count", type=int, default=100)
    ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--budget", type=int, default=60)
    ap.add_argument("--ref-iter", type=int, default=1000)
    ap.add_argument("--params", nargs="+", default=list(GRID))
    ap.add_argument("--reference-dir", default=None)
    ap.add_argument("--out", default="results/sensitivity")
    args = ap.parse_args()
    ensure_dir(args.out)
    cache = ensure_dir(args.reference_dir or os.path.join(args.out, "references"))

    problems = random_instances(args.count, n=args.size, seed=args.seed)
    models = [make_model(p) for p in problems]
    refs = [reference(p, m, cache, args.ref_iter)[0] for p, m in zip(problems, models)]

    rows = []
    for param in args.params:
        for label, kw in GRID[param]:
            ce_seed = []
            for seed in args.seeds:
                ce = []
                for p, m, c_ref in zip(problems, models, refs):
                    if not np.isfinite(c_ref):
                        continue
                    res = AMBIT(seed=seed, **kw).run(m, max_iter=10 * args.budget, max_analyses=args.budget)
                    if is_connected(p, full_field(m, res.rho_bar)):
                        ce.append(compliance_error(res.compliance, c_ref))
                ce_seed.append(float(np.mean(ce)) if ce else float("nan"))
            finite = np.isfinite(ce_seed)
            rows.append(dict(parameter=param, setting=label,
                             ce_pct=float(np.nanmean(ce_seed)) if finite.any() else float("nan"),
                             ce_seed_sd=float(np.nanstd(ce_seed, ddof=1)) if finite.sum() > 1 else 0.0))
            print(f"{param:10s} {label:11s} CE={rows[-1]['ce_pct']:6.2f}%", flush=True)
    write_csv(os.path.join(args.out, "table9.csv"), rows)


if __name__ == "__main__":
    main()
