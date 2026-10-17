"""Fixed-budget comparison on a set of two-dimensional instances (Table 7).

Instances are either generated (64 x 64 squares with volume fractions 0.30-0.50
in steps of 0.02, eight support families and a random boundary point load), or
read from a directory of .npz files written by ambit.problems.save_instance.

Each instance has a reference solution computed once, independently of the
compared methods, by long MMA and OC runs from the uniform design; the
lower-compliance connected result is kept. Compliance errors are relative to it.

Example:
    python experiments/run_dataset.py --count 200 --seed 1 --budget 60 --out results/dataset_test
    python experiments/run_dataset.py --instances data/instances --out results/dataset_custom
"""

import argparse
import glob
import os

import numpy as np

from common import DEFAULT_METHODS, METHODS, ensure_dir, make_model, seeds_for, write_csv

from ambit import MMA, OC
from ambit.metrics import compliance_error, is_connected
from ambit.problems import load_instance, random_instances, save_instance


def full_field(model, rho_bar):
    out = np.zeros(model.mesh.n_elem)
    out[model.idx] = rho_bar
    return out


def reference(problem, model, cache_dir, ref_iter):
    path = os.path.join(cache_dir, f"{problem.name}.npz")
    if os.path.exists(path):
        d = np.load(path)
        return float(d["compliance"]), float(d["uniform"])
    best = None
    for opt in (MMA(continuation=dict(theta_stop=1e-7)), OC(continuation=dict(theta_stop=1e-7))):
        res = opt.run(model, max_iter=ref_iter)
        ok = is_connected(problem, full_field(model, res.rho_bar))
        if ok and (best is None or res.compliance < best.compliance):
            best = res
    if best is None:
        return float("nan"), float("nan")
    p_max = 3.0
    uniform = model.evaluate(np.full(model.n, problem.volfrac), p_max, 1e-6, count=False).c_raw
    np.savez_compressed(path, compliance=best.compliance, uniform=uniform, x=best.x)
    return best.compliance, uniform


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--instances", default=None, help="directory of .npz instances")
    ap.add_argument("--count", type=int, default=100, help="number of generated instances")
    ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=1, help="seed of the generated instance set")
    ap.add_argument("--save-instances", default=None, help="directory to write generated instances")
    ap.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--budget", type=int, default=60, help="finite element analyses per run")
    ap.add_argument("--ref-iter", type=int, default=1000)
    ap.add_argument("--rmin", type=float, default=None, help="filter radius in elements")
    ap.add_argument("--out", default="results/dataset")
    args = ap.parse_args()

    if args.instances:
        problems = [load_instance(p) for p in sorted(glob.glob(os.path.join(args.instances, "*.npz")))]
    else:
        problems = random_instances(args.count, n=args.size, seed=args.seed)
        if args.save_instances:
            ensure_dir(args.save_instances)
            for pr in problems:
                save_instance(pr, os.path.join(args.save_instances, f"{pr.name}.npz"))
    cache = ensure_dir(os.path.join(args.out, "references"))

    rows = []
    for k, problem in enumerate(problems):
        model = make_model(problem, args.rmin)
        c_ref, c_uni = reference(problem, model, cache, args.ref_iter)
        if not np.isfinite(c_ref):
            print(f"{problem.name}: no connected reference, skipped", flush=True)
            continue
        family = problem.meta.get("bc_family", problem.name)
        for mname in args.methods:
            for seed in seeds_for(mname, args.seeds):
                res = METHODS[mname](seed).run(model, max_iter=10 * args.budget, max_analyses=args.budget)
                failed = not is_connected(problem, full_field(model, res.rho_bar))
                rows.append(dict(instance=problem.name, family=family, volfrac=problem.volfrac,
                                 method=mname, seed=seed, compliance=res.compliance, reference=c_ref,
                                 ce_pct=compliance_error(res.compliance, c_ref),
                                 ce_uniform_pct=compliance_error(c_uni, c_ref),
                                 failed=failed, vfe_pct=res.vfe, mnd_pct=res.mnd,
                                 iterations=res.iterations, analyses=res.analyses, time_s=res.time))
        print(f"[{k + 1}/{len(problems)}] {problem.name} ({family}, vf={problem.volfrac:.2f}) done",
              flush=True)
    write_csv(os.path.join(args.out, "runs.csv"), rows)
    print(f"wrote {len(rows)} runs to {os.path.join(args.out, 'runs.csv')}; "
          f"summarize with experiments/analyze_dataset.py")


if __name__ == "__main__":
    main()
