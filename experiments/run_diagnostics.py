"""Diagnostic measurements on the MBB beam (Table 5, panels A-F).

Example:
    python experiments/run_diagnostics.py --out results/diagnostics
    python experiments/run_diagnostics.py --meshes 150x50 300x100 --skip-b
"""

import argparse
import os

import numpy as np

from common import ABLATIONS, ensure_dir, make_model, write_json

from ambit import AMBIT, AdaptiveMoment, mbb
from ambit.diagnostics import (by_penalization, correlation_length, dispersion_at, stage_first_iterations,
                               stage_table)


def panel_a(model, max_iter):
    out = {}
    runs = [AdaptiveMoment("adam"), AdaptiveMoment("amsgrad"), AdaptiveMoment("adabelief"),
            AMBIT(name="AMBIT w/o Module I", **ABLATIONS["w/o Module I"]), AMBIT()]
    for opt in runs:
        res = opt.run(model, max_iter=max_iter)
        iota = np.asarray(res.hist["iota"])[:100]
        out[res.name] = float(np.mean(iota))
        print(f"  A  {res.name:20s} iota = {out[res.name]:.3f}", flush=True)
    return out


def panel_b(meshes, window, rmin_frac, alpha):
    rows = []
    for m in meshes:
        nx, ny = (int(v) for v in m.lower().split("x"))
        problem = mbb(nx, ny)
        model = make_model(problem, rmin_frac * ny)
        r_m = alpha * model.rmin
        row = {"mesh": m}
        for label, opt in (("Adam", AdaptiveMoment("adam", record_fields=True)),
                           ("AMBIT", AMBIT(record_fields=True, alpha=alpha))):
            res = opt.run(model, max_iter=window[1])
            P = res.fields["P"]
            lo, hi = window[0] - 1, min(window[1], len(P))
            lens = [correlation_length(model.mesh.to_grid(P[t])) for t in range(lo, hi)]
            ell = float(np.nanmean(lens))
            row[f"{label}_lP_h"] = ell
            row[f"{label}_lP_rm"] = ell / r_m
        rows.append(row)
        print(f"  B  {m:8s} Adam l/h={row['Adam_lP_h']:.2f} l/rm={row['Adam_lP_rm']:.2f}  "
              f"AMBIT l/h={row['AMBIT_lP_h']:.2f} l/rm={row['AMBIT_lP_rm']:.2f}", flush=True)
    return rows


def corollary_windows(hist, xs, delta):
    """Longest window per stage with a fixed active set, condition (B0), and V_I/(delta |I|)."""
    p = np.asarray(hist["p"])
    beta = np.asarray(hist["beta"])
    n = len(p)
    out = []
    start = 0
    for i in range(1, n + 1):
        if i == n or p[i] != p[start] or beta[i] != beta[start]:
            pv = float(p[start])
            row = {"p": pv, "beta": float(beta[start]), "T": i - start}
            if 1.0 < pv < 2.0:
                row.update(covered_pct=None, VI_ratio=None)
            else:
                act = [(x > delta) & (x < 1 - delta) for x in xs[start:i]]
                best, best_rng = 0, None
                j = 0
                while j < len(act):
                    k = j
                    while k + 1 < len(act) and np.array_equal(act[k + 1], act[j]):
                        k += 1
                    # (B0) requires the set to hold also at T+1, so the window is j..k-1
                    length = k - j
                    if length > best:
                        best, best_rng = length, (j, k)
                    j = k + 1
                row["covered_pct"] = 100.0 * best / (i - start)
                if best_rng is not None:
                    a, b = best_rng
                    inactive = ~act[a]
                    if inactive.any():
                        moves = sum(np.abs(xs[start + t + 1][inactive] - xs[start + t][inactive]).sum()
                                    for t in range(a, b))
                        row["VI_ratio"] = float(moves / (delta * inactive.sum()))
                    else:
                        row["VI_ratio"] = None
            out.append(row)
            start = i
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nelx", type=int, default=300)
    ap.add_argument("--nely", type=int, default=100)
    ap.add_argument("--max-iter", type=int, default=500)
    ap.add_argument("--meshes", nargs="+", default=["150x50", "300x100", "450x150", "600x200"])
    ap.add_argument("--window", type=int, nargs=2, default=[20, 60],
                    help="iterations over which the correlation length is averaged")
    ap.add_argument("--iterations-e", type=int, nargs="+", default=[10, 30, 50, 70, 100])
    ap.add_argument("--skip-b", action="store_true", help="skip the mesh-refinement panel")
    ap.add_argument("--out", default="results/diagnostics")
    args = ap.parse_args()
    ensure_dir(args.out)

    problem = mbb(args.nelx, args.nely)
    model = make_model(problem)
    report = {}

    print("Panel A: moment-constraint inconsistency, Eq. (14)")
    report["A_iota"] = panel_a(model, args.max_iter)

    if not args.skip_b:
        print("Panel B: preconditioner correlation length, Eq. (13)")
        report["B_correlation_length"] = panel_b(args.meshes, args.window, 0.024, 1.5)

    print("Full AMBIT and Adam runs with recorded fields")
    amb = AMBIT(record_fields=True)
    res = amb.run(model, max_iter=args.max_iter)
    adam = AdaptiveMoment("adam", record_fields=True).run(model, max_iter=args.max_iter)

    neg = by_penalization(res.hist, "neg_frac")
    report["C_negative_curvature_pct"] = {k: 100 * v for k, v in neg.items()}
    report["D_lambda_v"] = stage_first_iterations(res.hist, "lam_v")
    report["D_equivalent_zeta"] = stage_first_iterations(res.hist, "zeta")
    report["E_cv"] = {
        "Adam |m|/sqrt(v)": dispersion_at(adam.fields["step_scale"], args.iterations_e),
        "AMBIT tau": dispersion_at(res.fields["step_scale"], args.iterations_e),
        "strain energy": dispersion_at(res.fields["strain_energy"], args.iterations_e),
    }
    report["F_stages"] = stage_table(res.hist, amb.eta0)
    report["F_corollary1"] = corollary_windows(res.hist, res.fields["x"], amb.delta)
    report["run"] = dict(iterations=res.iterations, compliance=res.compliance, mnd=res.mnd)

    for k in ("C_negative_curvature_pct", "D_lambda_v", "D_equivalent_zeta"):
        print(f"  {k}: " + ", ".join(f"p={p:.2f}: {v:.3f}" for p, v in sorted(report[k].items())))
    for k, d in report["E_cv"].items():
        print(f"  E  CV {k:18s}: " + ", ".join(f"it{t}={v:.2f}" for t, v in d.items()))
    for row in report["F_stages"]:
        print(f"  F  p={row['p']:.2f} beta={row['beta']:5.1f} T={row['T']:3d} "
              f"A0={row.get('A0_pct', float('nan')):5.1f}% N_kappa={row.get('N_kappa', 0):3d} "
              f"omega={row.get('omega', float('nan')):.2f}")
    write_json(os.path.join(args.out, "table5.json"), report)


if __name__ == "__main__":
    main()
