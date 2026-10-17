"""Fermi-Dirac Bregman mirror step in the latent logit coordinate, Eqs. (15)-(22),
and the SiMPL method of Kim, Lazarov, Surowiec and Keith [24] used as a baseline.
"""

import numpy as np
from scipy.special import expit, logit

from .base import Optimizer


def volume_multiplier(base, coef, volume, V_star, tol=1e-9, max_iter=200):
    """Solve Phi(lam) = volume(sigmoid(base - lam * coef)) - V_star = 0 for lam >= 0, Eq. (22).

    Phi is nonincreasing in lam because coef >= 0 and volume is monotone.
    Returns lam = 0 when the constraint is inactive, Phi(0) <= 0.
    """
    if volume(expit(base)) <= V_star:
        return 0.0
    lo, hi = 0.0, 1.0
    while volume(expit(base - hi * coef)) > V_star:
        hi *= 2.0
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        if volume(expit(base - mid * coef)) > V_star:
            lo = mid
        else:
            hi = mid
        if hi - lo <= tol * max(1.0, hi):
            break
    return hi


def mirror_update(z, direction, metric, eta, dV, volume, V_star, kappa=0.0, z_ref=0.0):
    """Polarized mirror step in the logit coordinate, Eqs. (20)-(22):

    z_{t+1} = z_ref + (1 + eta kappa)(z - z_ref) - eta (direction + lam dV) / metric,

    with lam >= 0 fixed by a single monotone bisection on the resource constraint.
    """
    zk = z_ref + (1.0 + eta * kappa) * (z - z_ref)
    base = zk - eta * direction / metric
    coef = eta * dV / metric
    lam = volume_multiplier(base, coef, volume, V_star)
    return base - lam * coef, lam


def bregman_sym(x_new, x_old, z_new, z_old):
    """Symmetrized Fermi-Dirac Bregman divergence <x_new - x_old, z_new - z_old>."""
    return float(np.dot(x_new - x_old, z_new - z_old))


class SiMPL(Optimizer):
    """Mirror step with identity metric, Barzilai-Borwein initial step and Bregman backtracking.

    Every trial point costs one finite element analysis; the accepted trial is
    reused as the evaluation of the next iteration.
    """

    name = "SiMPL"

    def __init__(self, alpha0=1.0, c1=1e-4, shrink=0.5, max_backtrack=12,
                 alpha_bounds=(1e-4, 1e2), **kw):
        super().__init__(**kw)
        self.alpha0, self.c1, self.shrink = alpha0, c1, shrink
        self.max_backtrack, self.alpha_bounds = max_backtrack, alpha_bounds

    def init_state(self, model, x0):
        self.z = logit(np.clip(x0, 1e-6, 1 - 1e-6))
        self.z_prev = None
        self.g_prev = None

    def step(self, model, x, ev, cont):
        g = ev.g
        alpha = self.alpha0
        if self.z_prev is not None:
            dz = self.z - self.z_prev
            dg = g - self.g_prev
            den = abs(float(dz @ dg))
            if den > 0:
                alpha = float(dz @ dz) / den
        alpha = float(np.clip(alpha, *self.alpha_bounds))
        ones = np.ones_like(g)
        dV = model.volume_grad(x, cont.beta)
        vol = lambda y: model.volume(y, cont.beta)
        for _ in range(self.max_backtrack):
            z_new, lam = mirror_update(self.z, g, ones, alpha, dV, vol, model.V_star)
            x_new = expit(z_new)
            ev_new = model.evaluate(x_new, cont.p, cont.beta)
            D = bregman_sym(x_new, x, z_new, self.z)
            if ev_new.c <= ev.c - self.c1 / alpha * D:
                break
            if self.max_analyses is not None and model.n_analyses >= self.max_analyses:
                break
            alpha *= self.shrink
        self.z_prev, self.g_prev = self.z, g
        self.z = z_new
        self.log("alpha", alpha)
        self.log("lam", lam)
        return x_new, ev_new
