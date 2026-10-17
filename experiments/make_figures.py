"""Figures from the result files written by the experiment scripts.

Example:
    python experiments/make_figures.py --results results --out results/figures
"""

import argparse
import glob
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from common import ensure_dir, read_csv  # noqa: E402


def designs(results, out):
    files = sorted(glob.glob(os.path.join(results, "benchmarks", "designs", "*_s1.npz")))
    if not files:
        return
    problems = sorted({os.path.basename(f).split("_")[0] if not os.path.basename(f).startswith("l_bracket")
                       else "l_bracket" for f in files})
    for pname in problems:
        sel = [f for f in files if os.path.basename(f).startswith(pname + "_")]
        n = len(sel)
        cols = min(4, n)
        rows = int(np.ceil(n / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 1.6 * rows * 1.4), squeeze=False)
        for ax in axes.ravel():
            ax.axis("off")
        for ax, f in zip(axes.ravel(), sel):
            d = np.load(f)
            grid = d["rho_bar"].reshape(int(d["nelx"]), int(d["nely"])).T
            ax.imshow(1 - grid, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            method = os.path.basename(f)[len(pname) + 1:].rsplit("_s", 1)[0]
            ax.set_title(f"{method}  c={float(d['compliance']):.2f}", fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(out, f"designs_{pname}.png"), dpi=200)
        plt.close(fig)


def benchmark_bars(results, out):
    path = os.path.join(results, "benchmarks", "summary.csv")
    if not os.path.exists(path):
        return
    rows = read_csv(path)
    problems = list(dict.fromkeys(r["problem"] for r in rows))
    methods = list(dict.fromkeys(r["method"] for r in rows))
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    w = 0.8 / len(problems)
    x = np.arange(len(methods))
    for k, p in enumerate(problems):
        it = [float(next(r["iterations"] for r in rows if r["problem"] == p and r["method"] == m))
              for m in methods]
        axes[0].bar(x + k * w, it, w, label=p)
    axes[0].set_xticks(x + 0.4 - w / 2, methods, rotation=45, ha="right")
    axes[0].set_ylabel("Iterations to convergence")
    axes[0].legend(fontsize=8)
    mbb = [r for r in rows if r["problem"] == problems[0]]
    for r in mbb:
        axes[1].scatter(float(r["time_per_iter_ms"]), float(r["iterations"]))
        axes[1].annotate(r["method"], (float(r["time_per_iter_ms"]), float(r["iterations"])), fontsize=8)
    axes[1].set_xlabel("Time per iteration (ms)")
    axes[1].set_ylabel(f"Iterations ({problems[0]})")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "convergence_cost.png"), dpi=200)
    plt.close(fig)


def ablation_bars(results, out):
    path = os.path.join(results, "ablation", "ablation.csv")
    if not os.path.exists(path):
        return
    rows = read_csv(path)
    labels = [r["config"] for r in rows]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    for ax, key, xl in zip(axes, ("ce_pct", "mnd_pct", "mbb_iterations"),
                           ("Mean compliance error (%)", "Non-discreteness (%)", "Iterations (MBB beam)")):
        vals = [float(r[key]) for r in rows]
        ax.barh(np.arange(len(rows)), vals)
        ax.axvline(vals[0], color="k", lw=0.8, ls="--")
        ax.set_xlabel(xl)
    axes[0].set_yticks(np.arange(len(rows)), labels)
    axes[0].invert_yaxis()
    fig.tight_layout()
    fig.savefig(os.path.join(out, "ablation.png"), dpi=200)
    plt.close(fig)


def mesh_study(results, out):
    path = os.path.join(results, "mesh_study", "mesh_study.csv")
    if not os.path.exists(path):
        return
    rows = read_csv(path)
    keys = [k for k in rows[0] if k != "mesh"]
    fig, ax = plt.subplots(figsize=(6, 4))
    for k in keys:
        vals = [float(r[k]) if r.get(k) not in (None, "") else np.nan for r in rows]
        ax.plot([r["mesh"] for r in rows], vals, marker="o", label=k)
    ax.set_ylabel("Normalized compliance")
    ax.set_xlabel("Mesh")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "mesh_study.png"), dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="results/figures")
    args = ap.parse_args()
    ensure_dir(args.out)
    designs(args.results, args.out)
    benchmark_bars(args.results, args.out)
    ablation_bars(args.results, args.out)
    mesh_study(args.results, args.out)
    print(f"figures written to {args.out}")


if __name__ == "__main__":
    main()
