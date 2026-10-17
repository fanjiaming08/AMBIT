"""Statistical comparison used in Section 4.4: Friedman test, Conover post hoc procedure with
Holm correction, paired bootstrap intervals, Cliff's delta, and family-level inference.
"""

import numpy as np
from scipy import stats


def friedman_ranks(X):
    """Per-block ranks (1 = best, i.e. lowest) of an (n_blocks, k) array and the mean ranks."""
    R = np.apply_along_axis(stats.rankdata, 1, np.asarray(X, dtype=float))
    return R, R.mean(axis=0)


def friedman_test(X):
    X = np.asarray(X, dtype=float)
    res = stats.friedmanchisquare(*[X[:, j] for j in range(X.shape[1])])
    return float(res.statistic), float(res.pvalue)


def holm(pvalues):
    """Holm step-down adjusted p-values."""
    p = np.asarray(pvalues, dtype=float)
    m = p.size
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for i, idx in enumerate(order):
        running = max(running, (m - i) * p[idx])
        adj[idx] = min(1.0, running)
    return adj


def conover_vs_control(X, control):
    """Conover post hoc p-values of every column against the control column, Holm-adjusted."""
    R, _ = friedman_ranks(X)
    b, k = R.shape
    Rj = R.sum(axis=0)
    A1 = float((R ** 2).sum())
    df = (b - 1) * (k - 1)
    se = np.sqrt(2.0 * (b * A1 - float((Rj ** 2).sum())) / df)
    raw = np.full(k, np.nan)
    for j in range(k):
        if j == control:
            continue
        t = abs(Rj[j] - Rj[control]) / se
        raw[j] = 2.0 * stats.t.sf(t, df)
    mask = ~np.isnan(raw)
    adj = np.full(k, np.nan)
    adj[mask] = holm(raw[mask])
    return adj


def cliffs_delta(x, y):
    """Cliff's delta P(x > y) - P(x < y) with its conventional magnitude label."""
    x = np.asarray(x, dtype=float)
    y = np.sort(np.asarray(y, dtype=float))
    lt = np.searchsorted(y, x, side="left")
    le = np.searchsorted(y, x, side="right")
    gt_count = lt.sum()
    lt_count = (y.size - le).sum()
    d = float(gt_count - lt_count) / (x.size * y.size)
    a = abs(d)
    label = "negligible" if a < 0.147 else "small" if a < 0.33 else "medium" if a < 0.474 else "large"
    return d, label


def bootstrap_mean_ci(diff, n_boot=10000, level=0.95, seed=0, clusters=None):
    """Percentile bootstrap interval for the mean of paired differences.

    With clusters given, whole clusters are resampled (cluster bootstrap).
    """
    rng = np.random.default_rng(seed)
    diff = np.asarray(diff, dtype=float)
    if clusters is None:
        idx = rng.integers(0, diff.size, size=(n_boot, diff.size))
        means = diff[idx].mean(axis=1)
    else:
        clusters = np.asarray(clusters)
        groups = [diff[clusters == c] for c in np.unique(clusters)]
        sums = np.array([g.sum() for g in groups])
        counts = np.array([g.size for g in groups])
        pick = rng.integers(0, len(groups), size=(n_boot, len(groups)))
        means = sums[pick].sum(axis=1) / counts[pick].sum(axis=1)
    lo, hi = np.quantile(means, [(1 - level) / 2, 1 - (1 - level) / 2])
    return float(diff.mean()), float(lo), float(hi)


def family_wilcoxon(diff, clusters):
    """Wilcoxon signed-rank test on cluster means of paired differences."""
    clusters = np.asarray(clusters)
    means = np.array([np.mean(np.asarray(diff)[clusters == c]) for c in np.unique(clusters)])
    if np.allclose(means, 0):
        return means, 1.0
    return means, float(stats.wilcoxon(means).pvalue)
