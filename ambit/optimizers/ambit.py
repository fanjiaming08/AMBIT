"""AMBIT: Fermi-Dirac Bregman mirror step with a curvature-informed adaptive diagonal metric.

Module I    constraint-intrinsic polarized mirror step, Eqs. (15)-(22)
Module II   curvature-informed moment fusion, Eqs. (24)-(31)
Module III  filter-consistent moment smoothing and envelope, Eqs. (32)-(33)
Module IV   active-set-aware bias correction and feedback schedules, Eqs. (35)-(38)
Fallback    secant reliability index and identity-metric switch, Eq. (41) (AMBIT-R)

Each module can be disabled for the ablation study.

Step size. eta0 refers to the logit coordinate and to the sensitivity
normalization g <- g / mean|g_1|. The equivalent OC damping of Eq. (23),
eta * lambda * v, is logged at every iteration and summarized per stage by
experiments/run_diagnostics.py (panel D).
"""

import math

import numpy as np
from scipy.special import expit, logit

from ..utils import spearman, subset
from .base import Optimizer, project_box_volume
from .mirror import mirror_update


class AMBIT(Optimizer):
    name = "AMBIT"

    def __init__(
        self,
        beta1=0.9,
        beta2=0.99,
        beta2_range=(0.85, 0.999),
        eta0=0.5,
        eps=1e-8,
        eps_s=1e-8,
        gamma_range=(0.20, 0.80),
        n_subset=2048,
        kappa0=0.02,
        kappa_min=0.05,
        alpha=1.5,
        c1=0.30,
        c2=0.10,
        c3=0.05,
        delta=0.02,
        M_tol=5.0,
        mirror=True,
        metric="fused",
        filter_moments=True,
        envelope=True,
        momentum=True,
        active_set=True,
        step_schedule="feedback",
        adaptive_beta2=True,
        median_rescale=True,
        metric_bias=True,
        fixed_gamma=None,
        fallback=False,
        rho_min=0.30,
        rho_hysteresis=0.10,
        fallback_window=5,
        name=None,
        **kw,
    ):
        super().__init__(**kw)
        if metric not in ("fused", "belief", "identity"):
            raise ValueError("metric must be 'fused', 'belief' or 'identity'")
        if step_schedule not in ("feedback", "cap", "constant"):
            raise ValueError("step_schedule must be 'feedback', 'cap' or 'constant'")
        self.beta1, self.beta2_init, self.beta2_range = beta1, beta2, beta2_range
        self.eta0, self.eps, self.eps_s = eta0, eps, eps_s
        self.gamma_range, self.n_subset = gamma_range, n_subset
        self.kappa0, self.kappa_min, self.alpha = kappa0, kappa_min, alpha
        self.c1, self.c2, self.c3 = c1, c2, c3
        self.delta, self.M_tol = delta, M_tol
        self.mirror, self.metric = mirror, metric
        self.filter_moments, self.envelope = filter_moments, envelope
        self.momentum, self.active_set = momentum, active_set
        self.step_schedule, self.adaptive_beta2 = step_schedule, adaptive_beta2
        self.median_rescale, self.metric_bias = median_rescale, metric_bias
        self.fixed_gamma = fixed_gamma
        self.fallback, self.rho_min = fallback, rho_min
        self.rho_hysteresis, self.fallback_window = rho_hysteresis, fallback_window
        if name is not None:
            self.name = name
        elif fallback:
            self.name = "AMBIT-R"
        self.uses_curvature = metric == "fused" or fallback

    # ------------------------------------------------------------------
    def init_state(self, model, x0):
        n = x0.size
        self.z = logit(np.clip(x0, 1e-6, 1 - 1e-6))
        self.m = np.zeros(n)
        self.s = np.zeros(n)
        self.te = np.zeros(n, dtype=np.int64)
        self.prod2 = np.ones(n)
        self.P_env = np.full(n, self.eps)
        self.eta = self.eta0
        self.kappa = self.kappa0
        self.beta2 = self.beta2_init
        self.Hs = model.moment_filter(self.alpha) if self.filter_moments else None
        self.S_next = subset(self.rng, n, self.n_subset)
        self.prev = None
        self.rho_vals = []
        self.identity_on = False
        self.after_stage_change = False

    # ------------------------------------------------------------------
    def _secant_index(self, ev, S):
        """Secant reliability index of Eq. (41) on the subset drawn one iteration in advance."""
        if self.prev is None or self.after_stage_change:
            return math.nan
        cp, rb, h_hat = self.prev
        a = np.abs(ev.dc_drho_bar[S] - cp[S])
        b = h_hat[S] * np.abs(ev.rho_bar[S] - rb[S])
        return spearman(a, b)

    def _metric(self, ev, s_hat):
        n = s_hat.size
        info = {"gamma": math.nan, "r_S": math.nan, "neg_frac": math.nan, "h_hat": None}
        if ev.h1 is not None:
            hc = ev.h1 + ev.h2
            habs = np.abs(hc)
            h_hat = np.maximum(habs, self.kappa_min * np.median(habs))
            info["neg_frac"] = float(np.mean(hc < 0))
            info["h_hat"] = h_hat
        if self.metric == "identity":
            return np.ones(n), info
        a = np.sqrt(s_hat)
        if self.metric == "belief":
            return (a / np.median(a) if self.median_rescale else a), info
        h_hat = info["h_hat"]
        S = self.S_cur
        r_S = spearman(a[S], h_hat[S])
        if self.fixed_gamma is None:
            gamma = float(np.clip(0.5 * (1.0 + r_S), *self.gamma_range))
        else:
            gamma = float(self.fixed_gamma)
        info["gamma"], info["r_S"] = gamma, r_S
        if self.median_rescale:
            D = (1 - gamma) * a / np.median(a) + gamma * h_hat / np.median(h_hat)
        else:
            D = (1 - gamma) * a + gamma * h_hat
        return D, info

    def step(self, model, x, ev, cont):
        g = ev.g
        n = g.size
        b1, b2, d = self.beta1, self.beta2, self.delta
        self.S_cur = self.S_next
        self.S_next = subset(self.rng, n, self.n_subset)

        # Module IV (a): active set and moments, Eqs. (24), (25), (35), (36)
        chi = (x > d) & (x < 1 - d) if self.active_set else np.ones(n, dtype=bool)
        if self.momentum:
            self.m[chi] = b1 * self.m[chi] + (1 - b1) * g[chi]
        else:
            self.m = g.copy()
        floor = self.eps_s * float(np.mean(g * g))
        self.s[chi] = b2 * self.s[chi] + (1 - b2) * ((g[chi] - self.m[chi]) ** 2 + floor)
        self.te[chi] += 1
        self.prod2[chi] *= b2
        if self.momentum:
            m_hat = np.where(self.te > 0, self.m / (1 - b1 ** np.maximum(self.te, 1)), g)
        else:
            m_hat = g
        bias = np.where(self.prod2 < 1.0, 1.0 - self.prod2, 1.0)
        s_hat = np.where(self.prod2 < 1.0, self.s / bias, floor)

        # Module II: fused metric, Eqs. (29)-(31)
        D, info = self._metric(ev, s_hat)

        # Module III: filtering and monotone envelope, Eq. (32)
        if self.metric == "identity":
            P_hat = np.ones(n)
        else:
            P_bar = self.Hs @ D if self.Hs is not None else D
            self.P_env = np.maximum(self.P_env, P_bar) if self.envelope else P_bar
            P_hat = self.P_env / bias if self.metric_bias else self.P_env.copy()

        # Fallback: secant reliability index and identity switch, Eq. (41)
        rho = math.nan
        if self.uses_curvature and info["h_hat"] is not None:
            rho = self._secant_index(ev, self.S_cur)
            if not math.isnan(rho):
                self.rho_vals.append(rho)
            self.prev = (ev.dc_drho_bar.copy(), ev.rho_bar.copy(), info["h_hat"])
        self.after_stage_change = False
        if self.fallback and len(self.rho_vals) >= self.fallback_window:
            rho_bar = float(np.mean(self.rho_vals[-self.fallback_window:]))
            if not self.identity_on and rho_bar < self.rho_min:
                self.identity_on = True
            elif self.identity_on and rho_bar > self.rho_min + self.rho_hysteresis:
                self.identity_on = False
        if self.fallback and self.identity_on:
            P_hat = np.ones(n)

        # Module I: polarized mirror step, Eqs. (18)-(22)
        eta, kappa = self.eta, self.kappa
        vol = lambda y: model.volume(y, cont.beta)
        if self.mirror:
            dV = model.volume_grad(x, cont.beta)
            z_new, lam = mirror_update(self.z, m_hat, P_hat, eta, dV, vol, model.V_star, kappa)
            lam_v = lam * float(np.mean(dV))
            self.z = z_new
            x_new = expit(z_new)
            iota = 0.0
        else:
            xk = 0.5 + (1 + eta * kappa) * (x - 0.5)
            x_nom = xk - eta * m_hat / P_hat
            x_new = project_box_volume(x_nom, vol, model.V_star)
            step = x_nom - x
            ns = np.linalg.norm(step)
            iota = float(np.linalg.norm(x_new - x_nom) / ns) if ns > 0 else 0.0
            lam = lam_v = math.nan

        tau = eta / P_hat
        self.log("eta", eta)
        self.log("kappa", kappa)
        self.log("lam", lam)
        self.log("lam_v", lam_v)
        self.log("zeta", eta * lam_v)
        self.log("gamma", info["gamma"])
        self.log("r_S", info["r_S"])
        self.log("neg_frac", info["neg_frac"])
        self.log("secant", rho)
        self.log("identity_on", bool(self.fallback and self.identity_on))
        self.log("iota", iota)
        self.log("beta2", b2)
        self.log("active_frac", float(np.mean(chi)))
        self.log("A0", bool(np.all((x >= d) & (x <= 1 - d))))
        self.log("stage_iter", cont.stage_iter)
        self.log_field("x", x)
        self.log_field("P", P_hat)
        self.log_field("step_scale", tau)
        return x_new, None

    # ------------------------------------------------------------------
    def after_continuation(self, cont):
        """Module IV (b): polarization gain, second-moment horizon and step size, Eq. (38)."""
        self.kappa = self.kappa0 * float(cont.mnd > self.M_tol)
        if cont.stage_changed:
            if self.adaptive_beta2 and cont.p_changed:
                lo, hi = self.beta2_range
                self.beta2 = float(np.clip(self.beta2 * (1 - self.c3), lo, hi))
            self.eta = self.eta0
            self.P_env = np.full_like(self.P_env, self.eps)
            self.after_stage_change = True
            return
        k = cont.stage_iter
        if self.step_schedule == "feedback":
            dc = cont.dc
            th = cont.theta_c
            fac = 1.0 if not math.isfinite(dc) else float(
                np.clip(1 + self.c1 * (th - dc) / th, 1 - self.c2, 1 + self.c2))
            self.eta = min(self.eta0 / math.sqrt(k), self.eta * fac)
        elif self.step_schedule == "cap":
            self.eta = self.eta0 / math.sqrt(k)
        else:
            self.eta = self.eta0
