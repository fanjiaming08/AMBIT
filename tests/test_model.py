import unittest

import numpy as np

from ambit import TopOptModel, cantilever, l_bracket, mbb
from ambit.fem import FESolver, q4_stiffness


class TestElement(unittest.TestCase):
    def test_stiffness_symmetric_with_rigid_modes(self):
        KE = q4_stiffness(0.3)
        self.assertTrue(np.allclose(KE, KE.T))
        w = np.linalg.eigvalsh(KE)
        self.assertEqual(int(np.sum(np.abs(w) < 1e-10)), 3)
        self.assertTrue(np.all(w > -1e-10))

    def test_compliance_equals_energy(self):
        prob = mbb(30, 10)
        solver = FESolver(prob.mesh, prob.fixed)
        E = np.ones(prob.mesh.n_elem)
        U, _ = solver.solve(E, prob.F)
        K = solver.assemble(E)
        self.assertAlmostEqual(float(prob.F @ U), float(U @ (K @ U)), places=8)


class TestSensitivities(unittest.TestCase):
    def _fd_check(self, problem, p, beta, volume):
        model = TopOptModel(problem, rmin=1.5, volume=volume)
        rng = np.random.default_rng(0)
        x = rng.uniform(0.2, 0.8, model.n)
        ev = model.evaluate(x, p, beta)
        idx = rng.choice(model.n, 6, replace=False)
        h = 1e-6
        for i in idx:
            xp, xm = x.copy(), x.copy()
            xp[i] += h
            xm[i] -= h
            fd = (model.evaluate(xp, p, beta).c - model.evaluate(xm, p, beta).c) / (2 * h)
            self.assertAlmostEqual(ev.g[i], fd, delta=1e-5 * max(1.0, abs(fd)))
            vfd = (model.volume(xp, beta) - model.volume(xm, beta)) / (2 * h)
            self.assertAlmostEqual(model.volume_grad(x, beta)[i], vfd, delta=1e-5 * max(1.0, abs(vfd)))

    def test_gradient_mbb(self):
        self._fd_check(mbb(20, 8), 3.0, 4.0, "physical")

    def test_gradient_passive(self):
        self._fd_check(l_bracket(16, 16), 2.0, 2.0, "raw")

    def test_curvature_terms(self):
        """h1 + exact coupling term equals the second derivative of c in rho_bar_e."""
        prob = cantilever(12, 6)
        solver = FESolver(prob.mesh, prob.fixed)
        rng = np.random.default_rng(1)
        rb = rng.uniform(0.3, 0.9, prob.mesh.n_elem)
        p, Emin = 3.0, 1e-9

        def comp(r):
            U, _ = solver.solve(Emin + r ** p * (1 - Emin), prob.F)
            return float(prob.F @ U), U

        c0, U = comp(rb)
        KE = solver.KE
        e = 17
        ue = U[prob.mesh.edof[e]]
        ce = ue @ KE @ ue
        h1 = -p * (p - 1) * rb[e] ** (p - 2) * ce
        K = solver.assemble(Emin + rb ** p * (1 - Emin)).toarray()
        free = solver.free
        Kinv = np.zeros_like(K)
        Kinv[np.ix_(free, free)] = np.linalg.inv(K[np.ix_(free, free)])
        w = np.zeros(prob.mesh.n_dof)
        w[prob.mesh.edof[e]] = KE @ ue
        Ep = p * rb[e] ** (p - 1)
        h2 = 2 * Ep ** 2 * (w @ Kinv @ w)
        hstep = 1e-4
        rp, rm = rb.copy(), rb.copy()
        rp[e] += hstep
        rm[e] -= hstep
        fd = (comp(rp)[0] - 2 * c0 + comp(rm)[0]) / hstep ** 2
        self.assertAlmostEqual(h1 + h2, fd, delta=1e-3 * abs(fd))

        model = TopOptModel(prob, rmin=1.0)
        x = np.full(model.n, 0.5)
        ev = model.evaluate(x, 3.0, 1.0, curvature=True)
        self.assertTrue(np.all(ev.h1 <= 0))
        self.assertTrue(np.all(ev.h2 >= 0))


if __name__ == "__main__":
    unittest.main()
