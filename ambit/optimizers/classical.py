"""Classical update schemes: optimality criteria [6] and the method of moving asymptotes [9]."""

import numpy as np

from .base import Optimizer


class OC(Optimizer):
    """Optimality criteria update with damping zeta and a move limit."""

    name = "OC"

    def __init__(self, zeta=0.5, move=0.2, tol=1e-9, **kw):
        super().__init__(**kw)
        self.zeta, self.move, self.tol = zeta, move, tol

    def step(self, model, x, ev, cont):
        dV = np.maximum(model.volume_grad(x, cont.beta), 1e-30)
        B = np.maximum(-ev.g, 1e-30) / dV
        vol = lambda y: model.volume(y, cont.beta)
        lo_b = np.maximum(0.0, x - self.move)
        hi_b = np.minimum(1.0, x + self.move)

        def update(lam):
            return np.clip(x * (B / lam) ** self.zeta, lo_b, hi_b)

        lo, hi = 0.0, 1.0
        while vol(update(hi)) > model.V_star:
            hi *= 2.0
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if vol(update(mid)) > model.V_star:
                lo = mid
            else:
                hi = mid
            if (hi - lo) <= self.tol * (hi + lo):
                break
        lam = 0.5 * (lo + hi)
        self.log("lam", lam)
        return update(lam), None


class MMA(Optimizer):
    """Method of moving asymptotes for one resource constraint, with the subproblem solved
    through its one-dimensional dual."""

    name = "MMA"

    def __init__(self, move=0.2, asyinit=0.5, asyincr=1.2, asydecr=0.7, raa0=1e-5, tol=1e-10, **kw):
        super().__init__(**kw)
        self.move, self.asyinit = move, asyinit
        self.asyincr, self.asydecr, self.raa0, self.tol = asyincr, asydecr, raa0, tol

    def init_state(self, model, x0):
        self.k = 0
        self.xold1 = x0.copy()
        self.xold2 = x0.copy()
        self.low = None
        self.upp = None

    def step(self, model, x, ev, cont):
        self.k += 1
        xmin, xmax = 0.0, 1.0
        rng = xmax - xmin
        if self.k <= 2:
            low = x - self.asyinit * rng
            upp = x + self.asyinit * rng
        else:
            zzz = (x - self.xold1) * (self.xold1 - self.xold2)
            fac = np.ones_like(x)
            fac[zzz > 0] = self.asyincr
            fac[zzz < 0] = self.asydecr
            low = x - fac * (self.xold1 - self.low)
            upp = x + fac * (self.upp - self.xold1)
            low = np.clip(low, x - 10.0 * rng, x - 0.01 * rng)
            upp = np.clip(upp, x + 0.01 * rng, x + 10.0 * rng)
        alpha = np.maximum.reduce([np.full_like(x, xmin), low + 0.1 * (x - low), x - self.move])
        beta = np.minimum.reduce([np.full_like(x, xmax), upp - 0.1 * (upp - x), x + self.move])

        df0 = ev.g
        df1 = model.volume_grad(x, cont.beta) / model.V_star
        f1 = model.volume(x, cont.beta) / model.V_star - 1.0
        ux, xl = upp - x, x - low
        ux2, xl2 = ux ** 2, xl ** 2
        r = self.raa0 / rng
        p0 = ux2 * (1.001 * np.maximum(df0, 0) + 0.001 * np.maximum(-df0, 0) + r)
        q0 = xl2 * (0.001 * np.maximum(df0, 0) + 1.001 * np.maximum(-df0, 0) + r)
        p1 = ux2 * (1.001 * np.maximum(df1, 0) + 0.001 * np.maximum(-df1, 0) + r)
        q1 = xl2 * (0.001 * np.maximum(df1, 0) + 1.001 * np.maximum(-df1, 0) + r)
        b1 = np.sum(p1 / ux + q1 / xl) - f1

        def x_of(lam):
            sp_ = np.sqrt(p0 + lam * p1)
            sq_ = np.sqrt(q0 + lam * q1)
            return np.clip((sp_ * low + sq_ * upp) / (sp_ + sq_), alpha, beta)

        def con(lam):
            xs = x_of(lam)
            return np.sum(p1 / (upp - xs) + q1 / (xs - low)) - b1

        if con(0.0) <= 0.0:
            lam = 0.0
        else:
            lo, hi = 0.0, 1.0
            while con(hi) > 0.0:
                hi *= 2.0
            for _ in range(200):
                mid = 0.5 * (lo + hi)
                if con(mid) > 0.0:
                    lo = mid
                else:
                    hi = mid
                if hi - lo <= self.tol * max(1.0, hi):
                    break
            lam = hi
        x_new = x_of(lam)
        self.xold2, self.xold1 = self.xold1, x.copy()
        self.low, self.upp = low, upp
        self.log("lam", lam)
        return x_new, None
