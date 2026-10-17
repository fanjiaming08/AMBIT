"""Summary and statistical comparison of a dataset run (Tables 7 and 8).

Per-seed summaries are averaged over seeds. For the paired statistics, failed
instances are assigned the compliance error of the uniform initial design and
per-instance errors are averaged over seeds.

Example:
    python experiments/analyze_dataset.py results/dataset/runs.csv --control AMBIT
"""

import argparse
import os

import numpy as np

from common import read_csv, write_csv

from ambit import stats


def as_float(rows, key):
    return np.array([float(r[key]) for r in rows])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs")
    ap.add_argument("--control", default="AMBIT")
    ap.add_argument("--n-boot", type=int, default=10000)
    args = ap.parse_args()
    rows = read_csv(args.runs)
    out_dir = os.path.dirname(os.path.abspath(args.runs))
    for r in rows:
        r["failed"] = r["failed"] in ("True", "true", "1")

    methods = list(dict.fromkeys(r["method"] for r in rows))
    instances = sorted(set(r["instance"] for r in rows))
    ce_uniform = np.mean([float(r["ce_uniform_pct"]) for r in rows])

    # Table 7: per-seed summaries averaged over seeds
    summary = []
    for m in methods:
        sel = [r for r in rows if r["method"] == m]
        per_seed = []
        for s in sorted(set(r["seed"] for r in sel)):
            ss = [r for r in sel if r["seed"] == s]
            ce = as_float(ss, "ce_pct")
            failed = np.array([r["failed"] for r in ss])
            f = failed.mean()
            ce_ok = ce[~failed]
            per_seed.append(dict(
                ce_mean=ce_ok.mean() if ce_ok.size else np.nan,
                ce_imputed=(ce_ok.mean() if ce_ok.size else 0.0) * (1 - f) + ce_uniform * f,
                ce_median=np.median(ce_ok) if ce_ok.size else np.nan,
                vfe=as_float(ss, "vfe_pct").mean(), mnd=as_float(ss, "mnd_pct").mean(),
                failure=100 * f, time=as_float(ss, "time_s").mean()))
        row = {"method": m, "n_seeds": len(per_seed)}
        for k in per_seed[0]:
            row[k] = float(np.mean([p[k] for p in per_seed]))
        if len(per_seed) > 1:
            row["ce_mean_seed_sd"] = float(np.std([p["ce_mean"] for p in per_seed], ddof=1))
        summary.append(row)
    write_csv(os.path.join(out_dir, "table7.csv"), summary)
    print(f"Uniform-design compliance error used for imputation: {ce_uniform:.1f}%\n")
    print(f"{'method':10s}{'CE':>8s}{'CE imp':>9s}{'CE med':>9s}{'VFE':>8s}{'Mnd':>8s}"
          f"{'fail%':>8s}{'time s':>8s}")
    for s in summary:
        print(f"{s['method']:10s}{s['ce_mean']:8.2f}{s['ce_imputed']:9.2f}{s['ce_median']:9.2f}"
              f"{s['vfe']:8.3f}{s['mnd']:8.2f}{s['failure']:8.1f}{s['time']:8.2f}")

    # Table 8: paired statistics on seed-averaged, imputed per-instance errors
    if args.control not in methods:
        return
    X = np.full((len(instances), len(methods)), np.nan)
    fam = {}
    for i, inst in enumerate(instances):
        for j, m in enumerate(methods):
            sel = [r for r in rows if r["instance"] == inst and r["method"] == m]
            vals = [ce_uniform if r["failed"] else float(r["ce_pct"]) for r in sel]
            X[i, j] = np.mean(vals)
            fam[inst] = sel[0]["family"]
    keep = ~np.isnan(X).any(axis=1)
    X = X[keep]
    families = np.array([fam[inst] for inst, k in zip(instances, keep) if k])
    c = methods.index(args.control)
    _, mean_ranks = stats.friedman_ranks(X)
    chi2, p_f = stats.friedman_test(X)
    p_con = stats.conover_vs_control(X, c)
    table8 = []
    fam_p_raw = []
    for j, m in enumerate(methods):
        row = {"method": m, "friedman_rank": float(mean_ranks[j])}
        if j != c:
            diff = X[:, j] - X[:, c]
            d, lo, hi = stats.bootstrap_mean_ci(diff, n_boot=args.n_boot)
            delta, label = stats.cliffs_delta(X[:, j], X[:, c])
            _, flo, fhi = stats.bootstrap_mean_ci(diff, n_boot=args.n_boot, clusters=families)
            fmeans, fp = stats.family_wilcoxon(diff, families)
            fam_p_raw.append(fp)
            row.update(conover_p=float(p_con[j]), dCE=d, ci_lo=lo, ci_hi=hi, cliffs_delta=delta,
                       effect=label, family_ci_lo=flo, family_ci_hi=fhi,
                       families_control_lower=int(np.sum(fmeans > 0)), n_families=len(fmeans))
        table8.append(row)
    adj = stats.holm(fam_p_raw) if fam_p_raw else []
    k = 0
    for row in table8:
        if row["method"] != args.control:
            row["family_wilcoxon_p_holm"] = float(adj[k])
            k += 1
    write_csv(os.path.join(out_dir, "table8.csv"), table8)
    print(f"\nFriedman chi2 = {chi2:.2f}, p = {p_f:.2e}, n = {X.shape[0]} instances, "
          f"{len(np.unique(families))} families")
    print(f"{'method':10s}{'rank':>7s}{'Conover p':>11s}{'dCE':>10s}{'95% CI':>22s}{'Cliff':>8s}{'':>12s}"
          f"{'family p':>10s}")
    for row in table8:
        if row["method"] == args.control:
            print(f"{row['method']:10s}{row['friedman_rank']:7.2f}")
            continue
        print(f"{row['method']:10s}{row['friedman_rank']:7.2f}{row['conover_p']:11.2e}{row['dCE']:10.2f}"
              f"   [{row['ci_lo']:8.2f},{row['ci_hi']:8.2f}]{row['cliffs_delta']:8.2f} {row['effect']:>11s}"
              f"{row['family_wilcoxon_p_holm']:10.2e}")


if __name__ == "__main__":
    main()
