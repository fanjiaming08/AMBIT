"""Common driver shared by all iterative methods.

All methods share the density filter, the Heaviside map and the p/beta
continuation of Eq. (38); they differ only in the design update.
"""

import time
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from ..continuation import Continuation, nondiscreteness


@dataclass
class RunResult:
    name: str
    x: np.ndarray
    rho_bar: np.ndarray
    compliance: float
    vfe: float
    mnd: float
    iterations: int
    analyses: int
    time: float
    converged: bool
    hist: dict = field(default_factory=dict)
    fields: dict = field(default_factory=dict)


def project_box_volume(y, volume, V_star, tol=1e-12, max_iter=200):
    """Clipping and rescaling: x = clip(y - mu, 0, 1) with the smallest mu >= 0 such that
    volume(x) <= V_star. For the raw-field volume this is the Euclidean projection onto
    the box intersected with the resource constraint.
    """
    x = np.clip(y, 0.0, 1.0)
    if volume(x) <= V_star:
        return x
    lo, hi = 0.0, 1.0
    while volume(np.clip(y - hi, 0.0, 1.0)) > V_star:
        hi *= 2.0
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        if volume(np.clip(y - mid, 0.0, 1.0)) > V_star:
            lo = mid
        else:
            hi = mid
        if hi - lo <= tol * max(1.0, hi):
            break
    return np.clip(y - hi, 0.0, 1.0)


class Optimizer:
    name = "optimizer"
    uses_curvature = False

    def __init__(self, continuation=None, record_fields=False, seed=0):
        self.cont_kwargs = dict(continuation or {})
        self.record_fields = record_fields
        self.seed = seed
        self.max_analyses = None

    # interface -----------------------------------------------------------
    def init_state(self, model, x0):
        pass

    def step(self, model, x, ev, cont):
        """Return the next design and, optionally, an evaluation at it."""
        raise NotImplementedError

    def after_continuation(self, cont):
        pass

    # helpers -------------------------------------------------------------
    def log(self, key, value):
        self.hist[key].append(value)

    def log_field(self, key, value):
        if self.record_fields:
            self.fields[key].append(np.array(value, copy=True))

    # driver --------------------------------------------------------------
    def run(self, model, x0=None, max_iter=400, max_analyses=None):
        model.reset()
        cont = Continuation(**self.cont_kwargs)
        x = np.full(model.n, model.problem.volfrac) if x0 is None else np.array(x0, dtype=float)
        self.hist = defaultdict(list)
        self.fields = defaultdict(list)
        self.max_analyses = max_analyses
        self.rng = np.random.default_rng(self.seed)
        self.init_state(model, x)
        t0 = time.perf_counter()
        ev = None
        it = 0
        converged = False
        while it < max_iter:
            if max_analyses is not None and model.n_analyses >= max_analyses:
                break
            if ev is None or ev.p != cont.p or ev.beta != cont.beta:
                ev = model.evaluate(x, cont.p, cont.beta, curvature=self.uses_curvature)
            it += 1
            mnd = nondiscreteness(ev.rho_bar)
            self.log("c", ev.c_raw)
            self.log("p", cont.p)
            self.log("beta", cont.beta)
            self.log("mnd", mnd)
            self.log_field("strain_energy", ev.strain_energy)
            x_new, ev_next = self.step(model, x, ev, cont)
            cont.update(ev.c, mnd)
            self.log("analyses", model.n_analyses)
            self.log("stage_changed", cont.stage_changed)
            self.after_continuation(cont)
            x, ev = x_new, ev_next
            if cont.converged():
                converged = True
                break
        elapsed = time.perf_counter() - t0
        fin = model.evaluate(x, cont.p_max, cont.beta_max, count=False)
        vf = float(np.mean(fin.rho_bar))
        return RunResult(
            name=self.name, x=x, rho_bar=fin.rho_bar, compliance=fin.c_raw,
            vfe=abs(vf - model.problem.volfrac) * 100.0, mnd=nondiscreteness(fin.rho_bar),
            iterations=it, analyses=model.n_analyses, time=elapsed, converged=converged,
            hist={k: np.asarray(v) for k, v in self.hist.items()},
            fields={k: np.asarray(v) for k, v in self.fields.items()},
        )
