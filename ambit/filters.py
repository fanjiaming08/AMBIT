"""Linear density filter, Eq. (4), and smoothed Heaviside projection, Eqs. (5)-(6)."""

import numpy as np
import scipy.sparse as sp
from scipy.spatial import cKDTree


def cone_filter(centroids, radius, volumes=None):
    """Row-normalized cone filter with weights max(0, radius - |x_e - x_i|) * v_i."""
    centroids = np.asarray(centroids, dtype=float)
    n = centroids.shape[0]
    v = np.ones(n) if volumes is None else np.asarray(volumes, dtype=float)
    tree = cKDTree(centroids)
    pairs = tree.query_pairs(radius, output_type="ndarray")
    if pairs.size:
        d = np.linalg.norm(centroids[pairs[:, 0]] - centroids[pairs[:, 1]], axis=1)
        w = np.maximum(0.0, radius - d)
        rows = np.concatenate([pairs[:, 0], pairs[:, 1], np.arange(n)])
        cols = np.concatenate([pairs[:, 1], pairs[:, 0], np.arange(n)])
        vals = np.concatenate([w, w, np.full(n, float(radius))])
    else:
        rows = cols = np.arange(n)
        vals = np.full(n, float(radius))
    W = sp.csr_matrix((vals * v[cols], (rows, cols)), shape=(n, n))
    s = np.asarray(W.sum(axis=1)).ravel()
    return sp.diags(1.0 / s) @ W


def restrict_filter(H, idx):
    """Filter restricted to a subset of elements and renormalized row-wise."""
    Hs = H[idx][:, idx].tocsr()
    s = np.asarray(Hs.sum(axis=1)).ravel()
    s[s <= 0] = 1.0
    return (sp.diags(1.0 / s) @ Hs).tocsr()


def heaviside(x, beta, eta):
    """Smoothed Heaviside map and its derivative."""
    a = np.tanh(beta * eta)
    b = np.tanh(beta * (1.0 - eta))
    t = np.tanh(beta * (x - eta))
    return np.clip((a + t) / (a + b), 0.0, 1.0), beta * (1.0 - t ** 2) / (a + b)
