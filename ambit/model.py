"""Compliance model on the raw design field: Eqs. (1)-(8) and the curvature terms (26)-(28)."""

from dataclasses import dataclass

import numpy as np

from .fem import FESolver
from .filters import cone_filter, heaviside, restrict_filter


@dataclass
class Evaluation:
    p: float
    beta: float
    c: float                  # normalized compliance
    c_raw: float              # compliance F^T U
    g: np.ndarray             # normalized dc/drho on design elements
    dc_drho_bar: np.ndarray   # normalized dc/drho_bar on design elements
    rho_bar: np.ndarray       # projected density on design elements
    strain_energy: np.ndarray # E(rho_bar) u^T k0 u on design elements
    h1: np.ndarray = None     # frozen curvature, Eq. (26), normalized
    h2: np.ndarray = None     # Jacobi coupling curvature, Eq. (28), normalized


class TopOptModel:
    """SIMP compliance with density filter and Heaviside projection.

    The design vector x holds the raw densities of the non-passive elements.
    Passive elements are void and element volumes are one. The resource
    constraint is V(x) <= volfrac * n_design, where V sums either the raw
    densities (volume="raw", the convention of Eq. (3)) or the projected
    densities (volume="physical"). Both choices keep the multiplier function
    of Eq. (22) monotone, since the filter weights are nonnegative and the
    Heaviside map is increasing.
    """

    def __init__(self, problem, rmin=None, eta_H=0.5, E0=1.0, Emin=1e-9, nu=0.3, rho_fl=1e-3,
                 volume="physical"):
        if volume not in ("raw", "physical"):
            raise ValueError("volume must be 'raw' or 'physical'")
        self.volume_on = volume
        self.problem = problem
        self.mesh = problem.mesh
        self.solver = FESolver(self.mesh, problem.fixed, nu=nu)
        self.F = np.asarray(problem.F, dtype=float)
        self.design = problem.design
        self.idx = np.flatnonzero(self.design)
        self.n = self.idx.size
        self.v = np.ones(self.n)
        self.V_star = problem.volfrac * self.n
        if rmin is None:
            rmin = 0.024 * self.mesh.nely
        self.rmin = float(rmin)
        self.H = cone_filter(self.mesh.centroids, self.rmin)
        self.Ht = self.H.T.tocsr()
        self.eta_H = eta_H
        self.E0, self.Emin = E0, Emin
        self.rho_fl = rho_fl
        self.KE = self.solver.KE
        self.n_analyses = 0
        self.g_scale = None

    def reset(self):
        self.n_analyses = 0
        self.g_scale = None

    def moment_filter(self, alpha):
        """Filter of radius alpha * rmin on the design elements (Module III)."""
        H = cone_filter(self.mesh.centroids, max(alpha * self.rmin, 1.0))
        return restrict_filter(H, self.idx)

    def design_filter(self, radius):
        return restrict_filter(cone_filter(self.mesh.centroids, radius), self.idx)

    def full(self, x):
        rho = np.zeros(self.mesh.n_elem)
        rho[self.idx] = x
        return rho

    def physical(self, x, beta):
        rho_t = self.H @ self.full(x)
        rho_b, d_rho_b = heaviside(rho_t, beta, self.eta_H)
        if self.problem.passive is not None:
            rho_b[self.problem.passive] = 0.0
            d_rho_b[self.problem.passive] = 0.0
        return rho_t, rho_b, d_rho_b

    def volume(self, x, beta):
        """Constrained volume V(x)."""
        if self.volume_on == "raw":
            return float(np.sum(x))
        return float(np.sum(self.physical(x, beta)[1][self.idx]))

    def volume_grad(self, x, beta):
        """Gradient of V with respect to the raw design."""
        if self.volume_on == "raw":
            return np.ones(self.n)
        _, _, drb = self.physical(x, beta)
        return (self.Ht @ drb)[self.idx]

    def evaluate(self, x, p, beta, curvature=False, count=True):
        _, rb, drb = self.physical(x, beta)
        dE = self.E0 - self.Emin
        E = self.Emin + rb ** p * dE
        U, inv_diag = self.solver.solve(E, self.F)
        ue = U[self.mesh.edof]
        w = ue @ self.KE
        ce = np.einsum("ij,ij->i", w, ue)
        c_raw = float(self.F @ U)
        dEdr = p * rb ** (p - 1.0) * dE
        dc_drb = -dEdr * ce
        g_full = self.Ht @ (drb * dc_drb)
        g = g_full[self.idx]
        if self.g_scale is None:
            self.g_scale = float(np.mean(np.abs(g))) or 1.0
        s = self.g_scale
        ev = Evaluation(
            p=p, beta=beta, c=c_raw / s, c_raw=c_raw, g=g / s,
            dc_drho_bar=dc_drb[self.idx] / s, rho_bar=rb[self.idx],
            strain_energy=(E * ce)[self.idx],
        )
        if curvature:
            h1 = -p * (p - 1.0) * np.maximum(rb, self.rho_fl) ** (p - 2.0) * dE * ce
            h2 = 2.0 * dEdr ** 2 * np.einsum("ij,ij->i", w ** 2, inv_diag[self.mesh.edof])
            ev.h1 = h1[self.idx] / s
            ev.h2 = h2[self.idx] / s
        if count:
            self.n_analyses += 1
        return ev

    def compliance(self, x, p, beta):
        """Un-normalized compliance, not counted as an analysis."""
        return self.evaluate(x, p, beta, count=False).c_raw
