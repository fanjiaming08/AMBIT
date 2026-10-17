"""Feedback continuation of the penalization p and projection sharpness beta, Eqs. (37)-(38).

A stage is a maximal run of iterations with constant p and beta. The penalization
advances by dp once the current stage has settled, i.e. the trailing relative
compliance change is below theta_c and the stage is older than T_hold iterations.

Two rules are available for beta. With beta_rule="every_iteration", beta is
multiplied by mu at every iteration at which the non-discreteness exceeds M_tol,
which is the literal form of Eq. (38). With beta_rule="settled" (default), beta
is advanced together with p under the same settling test, while the
non-discreteness exceeds M_tol or once p has reached p_max, so that every run
terminates at beta_max.
"""

import math

import numpy as np


def nondiscreteness(rho_bar):
    """Measure of non-discreteness in percent, Eq. (37)."""
    rho_bar = np.asarray(rho_bar)
    return float(np.mean(4.0 * rho_bar * (1.0 - rho_bar)) * 100.0)


class Continuation:
    def __init__(self, p0=1.0, p_max=3.0, dp=0.25, beta0=1.0, beta_max=16.0, mu=2.0,
                 theta_c=1e-3, T_hold=8, M_tol=5.0, theta_stop=1e-4, M_stop=4.0,
                 window=3, mode="feedback", period=20, T_max=None, beta_rule="settled"):
        if beta_rule not in ("settled", "every_iteration"):
            raise ValueError("beta_rule must be 'settled' or 'every_iteration'")
        self.beta_rule = beta_rule
        self.p0, self.p_max, self.dp = p0, p_max, dp
        self.beta0, self.beta_max, self.mu = beta0, beta_max, mu
        self.theta_c, self.T_hold, self.M_tol = theta_c, T_hold, M_tol
        self.theta_stop, self.M_stop = theta_stop, M_stop
        self.window, self.mode, self.period, self.T_max = window, mode, period, T_max
        self.reset()

    def reset(self):
        self.p = self.p0
        self.beta = self.beta0
        self.t = 0
        self.t_last = 0
        self.c_stage = []
        self.dc = math.inf
        self.mnd = 100.0
        self.p_changed = False
        self.stage_changed = False

    @property
    def stage_iter(self):
        """Stage-local index of the next iteration, starting at 1."""
        return self.t - self.t_last + 1

    def _rel_change(self):
        c = self.c_stage
        if len(c) <= self.window:
            return math.inf
        ct = c[-1]
        return max(abs(ct - c[-1 - k]) for k in range(1, self.window + 1)) / max(abs(ct), 1e-300)

    def update(self, c, mnd):
        """Record iteration t and advance p and beta. Returns True if a new stage begins."""
        self.t += 1
        self.c_stage.append(float(c))
        self.dc = self._rel_change()
        self.mnd = float(mnd)
        in_stage = self.t - self.t_last
        p_new, beta_new = self.p, self.beta
        if self.mode == "feedback":
            settled = self.dc < self.theta_c
            if self.T_max is not None and in_stage >= self.T_max:
                settled = True
            if settled and in_stage > self.T_hold:
                p_new = min(self.p_max, self.p + self.dp)
                if self.beta_rule == "settled" and (mnd > self.M_tol or p_new >= self.p_max):
                    beta_new = min(self.beta_max, self.beta * self.mu)
            if self.beta_rule == "every_iteration" and mnd > self.M_tol:
                beta_new = min(self.beta_max, self.beta * self.mu)
        else:
            if in_stage >= self.period:
                if self.p < self.p_max:
                    p_new = min(self.p_max, self.p + self.dp)
                else:
                    beta_new = min(self.beta_max, self.beta * self.mu)
        self.p_changed = p_new != self.p
        self.stage_changed = self.p_changed or beta_new != self.beta
        self.p, self.beta = p_new, beta_new
        if self.stage_changed:
            self.t_last = self.t
            self.c_stage = []
        return self.stage_changed

    def converged(self):
        """Termination test of step (ix) of Algorithm 1, applied at full penalization."""
        return (self.p >= self.p_max and self.beta >= self.beta_max and not self.stage_changed
                and self.dc < self.theta_stop and self.mnd < self.M_stop)
