"""Evaluation metrics: compliance error, volume fraction error and the connectivity failure test."""

import numpy as np
from scipy import ndimage


def compliance_error(c, c_ref):
    """Relative excess over the reference compliance, in percent."""
    return 100.0 * (c - c_ref) / c_ref


def imputed_mean(ce_values, failed, ce_fail):
    """CE mean with failed instances assigned the compliance error ce_fail, Section 4.1.3."""
    ce = np.asarray(ce_values, dtype=float)
    failed = np.asarray(failed, dtype=bool)
    f = failed.mean() if failed.size else 0.0
    ce_excl = ce[~failed].mean() if (~failed).any() else np.nan
    return float(ce_excl * (1 - f) + ce_fail * f)


def _adjacent_elements(nelx, nely, ix, iy):
    out = []
    for ex in (ix - 1, ix):
        for ey in (iy - 1, iy):
            if 0 <= ex < nelx and 0 <= ey < nely:
                out.append((ex, ey))
    return out


def is_connected(problem, rho_bar_full, threshold=0.5):
    """True if every loaded node is linked to at least one supported node by solid elements.

    Solid elements are those with projected density above the threshold; two
    elements are linked when they share an edge.
    """
    nelx, nely = problem.nelx, problem.nely
    solid = (np.asarray(rho_bar_full).reshape(nelx, nely) > threshold)
    labels, _ = ndimage.label(solid)

    def node_xy(dof):
        node = dof // 2
        return node // (nely + 1), node % (nely + 1)

    def node_labels(dofs):
        labs = set()
        for d in dofs:
            ix, iy = node_xy(int(d))
            for ex, ey in _adjacent_elements(nelx, nely, ix, iy):
                if labels[ex, ey] > 0:
                    labs.add(int(labels[ex, ey]))
        return labs

    support_labels = node_labels(problem.fixed)
    loaded_nodes = np.unique(np.flatnonzero(np.asarray(problem.F) != 0) // 2)
    for nd in loaded_nodes:
        labs = node_labels([2 * nd])
        if not labs & support_labels:
            return False
    return True
