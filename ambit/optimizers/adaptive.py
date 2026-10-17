"""Adaptive moment baselines, Eqs. (9)-(11): Adam [11], AMSGrad [15], AdaBelief [16], Lion [17].

The nominal step is followed by the clipping-and-rescaling projection onto the
box and the resource constraint. The moments accumulate the nominal step.
"""

import numpy as np

from .base import Optimizer, project_box_volume


class AdaptiveMoment(Optimizer):
    DEFAULTS = {
        "adam": dict(lr=0.02, beta1=0.9, beta2=0.999),
        "amsgrad": dict(lr=0.02, beta1=0.9, beta2=0.999),
        "adabelief": dict(lr=0.01, beta1=0.9, beta2=0.999),
        "lion": dict(lr=0.002, beta1=0.9, beta2=0.99),
    }
    NAMES = {"adam": "Adam", "amsgrad": "AMSGrad", "adabelief": "AdaBelief", "lion": "Lion"}

    def __init__(self, variant="adam", lr=None, beta1=None, beta2=None, eps=1e-8, **kw):
        super().__init__(**kw)
        d = self.DEFAULTS[variant]
        self.variant = variant
        self.name = self.NAMES[variant]
        self.lr = d["lr"] if lr is None else lr
        self.beta1 = d["beta1"] if beta1 is None else beta1
        self.beta2 = d["beta2"] if beta2 is None else beta2
        self.eps = eps

    def init_state(self, model, x0):
        n = x0.size
        self.m = np.zeros(n)
        self.v2 = np.zeros(n)
        self.vmax = np.zeros(n)
        self.t = 0

    def step(self, model, x, ev, cont):
        g = ev.g
        b1, b2 = self.beta1, self.beta2
        self.t += 1
        if self.variant == "lion":
            d = np.sign(b1 * self.m + (1 - b1) * g)
            self.m = b2 * self.m + (1 - b2) * g
            precond = np.ones_like(g)
            m_hat = self.m
        else:
            self.m = b1 * self.m + (1 - b1) * g
            if self.variant == "adabelief":
                self.v2 = b2 * self.v2 + (1 - b2) * (g - self.m) ** 2 + self.eps
            else:
                self.v2 = b2 * self.v2 + (1 - b2) * g * g
            vv = self.v2
            if self.variant == "amsgrad":
                self.vmax = np.maximum(self.vmax, self.v2)
                vv = self.vmax
            m_hat = self.m / (1 - b1 ** self.t)
            v_hat = vv / (1 - b2 ** self.t)
            precond = np.sqrt(v_hat) + self.eps
            d = m_hat / precond
        delta = -self.lr * d
        x_nom = x + delta
        x_new = project_box_volume(x_nom, lambda y: model.volume(y, cont.beta), model.V_star)
        nd = np.linalg.norm(delta)
        iota = np.linalg.norm(x_new - x_nom) / nd if nd > 0 else 0.0
        self.log("iota", iota)
        self.log_field("P", precond)
        self.log_field("step_scale", np.abs(d))
        return x_new, None
