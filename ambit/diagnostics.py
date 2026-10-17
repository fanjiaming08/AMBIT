"""Measured indices for the mismatches of Section 3.2 and the coverage statistics of Panel F."""

import math

import numpy as np

from .utils import cv


def correlation_length(field2d, max_r=None):
    """Empirical correlation length of a field on a structured grid, Eq. (13), in element units.

    The correlation at separation r is the Pearson correlation of the field with
    its shift by r along x and along y, averaged over both directions. The
    length is the smallest r at which it falls below 1/e, linearly interpolated.
    """
    f = np.asarray(field2d, dtype=float)
    nx, ny = f.shape
    if max_r is None:
        max_r = max(2, min(nx, ny) // 2)
    target = math.exp(-1.0)
    prev_r, prev_c = 0.0, 1.0
    for r in range(1, max_r + 1):
        cs = []
        if r < nx:
            a, b = f[:-r, :].ravel(), f[r:, :].ravel()
            if a.std() > 0 and b.std() > 0:
                cs.append(np.corrcoef(a, b)[0, 1])
        if r < ny:
            a, b = f[:, :-r].ravel(), f[:, r:].ravel()
            if a.std() > 0 and b.std() > 0:
                cs.append(np.corrcoef(a, b)[0, 1])
        if not cs:
            return float("nan")
        c = float(np.mean(cs))
        if c < target:
            return prev_r + (prev_c - target) / (prev_c - c) * (r - prev_r)
        prev_r, prev_c = r, c
    return float(max_r)


def stage_table(hist, eta0, delta_used=True):
    """Per-stage statistics: length T, fraction with (A0), polarized iterations, and omega.

    omega = max_t eta0 / (sqrt(t) eta_t), with t the stage-local counter.
    """
    p = np.asarray(hist["p"])
    beta = np.asarray(hist["beta"])
    out = []
    start = 0
    n = len(p)
    for i in range(1, n + 1):
        if i == n or p[i] != p[start] or beta[i] != beta[start]:
            sl = slice(start, i)
            row = {"p": float(p[start]), "beta": float(beta[start]), "start": start + 1, "T": i - start}
            if "A0" in hist:
                row["A0_pct"] = 100.0 * float(np.mean(hist["A0"][sl]))
            if "kappa" in hist:
                row["N_kappa"] = int(np.sum(np.asarray(hist["kappa"][sl]) > 0))
            if "eta" in hist:
                eta = np.asarray(hist["eta"][sl])
                k = np.arange(1, eta.size + 1)
                row["omega"] = float(np.max(eta0 / (np.sqrt(k) * eta)))
            out.append(row)
            start = i
    return out


def stage_first_iterations(hist, key, n_first=5):
    """Mean of hist[key] over the first n_first iterations of each stage, grouped by p."""
    p = np.asarray(hist["p"])
    beta = np.asarray(hist["beta"])
    vals = np.asarray(hist[key], dtype=float)
    res = {}
    start = 0
    n = len(p)
    for i in range(1, n + 1):
        if i == n or p[i] != p[start] or beta[i] != beta[start]:
            seg = vals[start:min(i, start + n_first)]
            seg = seg[np.isfinite(seg)]
            if seg.size:
                res.setdefault(float(p[start]), []).append(float(seg.mean()))
            start = i
    return {k: float(np.mean(v)) for k, v in res.items()}


def by_penalization(hist, key):
    """Mean of hist[key] over the iterations at each value of p."""
    p = np.asarray(hist["p"])
    vals = np.asarray(hist[key], dtype=float)
    res = {}
    for pv in np.unique(p):
        seg = vals[p == pv]
        seg = seg[np.isfinite(seg)]
        if seg.size:
            res[float(pv)] = float(seg.mean())
    return res


def dispersion_at(fields, iterations):
    """Coefficient of variation of a recorded field at the given 1-based iterations."""
    out = {}
    for t in iterations:
        if t - 1 < len(fields):
            out[t] = cv(np.abs(fields[t - 1]))
    return out
