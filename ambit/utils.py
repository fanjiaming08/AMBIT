"""Small numerical helpers."""

import numpy as np
from scipy.stats import rankdata


def spearman(a, b):
    """Spearman rank correlation; zero when either argument is constant."""
    ra = rankdata(a)
    rb = rankdata(b)
    ra -= ra.mean()
    rb -= rb.mean()
    den = np.sqrt((ra @ ra) * (rb @ rb))
    return float(ra @ rb / den) if den > 0 else 0.0


def subset(rng, n, size):
    """Random subset of indices of the given size, without replacement."""
    if size >= n:
        return np.arange(n)
    return rng.choice(n, size=size, replace=False)


def cv(a):
    """Coefficient of variation."""
    a = np.asarray(a, dtype=float)
    m = a.mean()
    return float(a.std() / m) if m != 0 else float("nan")
