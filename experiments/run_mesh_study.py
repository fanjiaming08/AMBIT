"""MBB beam under mesh refinement with r_min fixed in physical units (Figure 8).

Compliances are normalized by a common reference, an MMA solution on the finest
mesh. The control column re-analyzes one fixed layout, the OC layout of the
coarsest mesh, on every mesh of the sequence, and so isolates discretization
error from layout change.

Example:
    python experiments/run_mesh_study.py --out results/mesh_study
"""

import argparse
import os

import numpy as np

from common import ABLATIONS, ensure_dir, make_model, write_csv

from ambit import AMBIT, MMA, OC, AdaptiveMoment, SiMPL, mbb
from ambit.fem import FESolver
from ambit.utils import cv

METHODS = {
    "OC": lambda: OC(),
    "SiMPL": lambda: SiMPL(),
    "Adam": lambda: AdaptiveMoment("adam"),
    "AMBIT w/o Module III": lambda: AMBIT(name="AMBIT w/o Module III", **ABLATIONS["w/o Module III"]),
    "AMBIT": lambda: AMBIT(),
}


def analyze_layout(problem, rho_bar_full, p=3.0, Emin=1e-9):
    solver = FESolver(problem.mesh, problem.fixed)
    U, _ = solver.solve(Emin + rho_bar_full ** p * (1 - Emin), problem.F)
    return float(problem.F @ U)


def refine(grid, f):
    return np.repeat(np.repeat(grid, f, axis=0), f, axis=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--meshes", nargs="+", default=["150x50", "300x100", "450x150", "600x200"])
    ap.add_argument("--methods", nargs="+", default=list(METHODS))
    ap.add_argument("--max-iter", type=int, default=500)
    ap.add_argument("--ref-iter", type=int, default=1500)
    ap.add_argument("--out", default="results/mesh_study")
    args = ap.parse_args()
    ensure_dir(args.out)

    meshes = [tuple(int(v) for v in m.lower().split("x")) for m in args.meshes]
    finest = mbb(*meshes[-1])
    ref = MMA(continuation=dict(theta_stop=1e-7)).run(make_model(finest), max_iter=args.ref_iter).compliance
    print(f"reference compliance on {args.meshes[-1]}: {ref:.4f}", flush=True)

    rows = []
    coarse_layout = None
    for (nx, ny), label in zip(meshes, args.meshes):
        problem = mbb(nx, ny)
        model = make_model(problem)
        row = {"mesh": label}
        for name in args.methods:
            res = METHODS[name]().run(model, max_iter=args.max_iter)
            row[name] = res.compliance / ref
            if name == "OC" and coarse_layout is None:
                coarse_layout = (nx, ny, model.mesh.to_grid(res.rho_bar))
            print(f"{label:8s} {name:22s} normalized compliance {row[name]:.4f}", flush=True)
        if coarse_layout is not None:
            cx, cy, grid = coarse_layout
            if nx % cx == 0 and ny % cy == 0 and nx // cx == ny // cy:
                row["fixed OC layout"] = analyze_layout(problem, refine(grid, nx // cx).ravel()) / ref
        rows.append(row)
    write_csv(os.path.join(args.out, "mesh_study.csv"), rows)

    cols = [k for k in rows[0] if k != "mesh"]
    cv_row = {"mesh": "CV across meshes (%)"}
    for k in cols:
        vals = [r[k] for r in rows if k in r]
        cv_row[k] = 100 * cv(vals)
    write_csv(os.path.join(args.out, "mesh_study_cv.csv"), [cv_row])
    print("\nCoefficient of variation across meshes (%)")
    for k in cols:
        print(f"  {k:22s} {cv_row[k]:.2f}")


if __name__ == "__main__":
    main()
