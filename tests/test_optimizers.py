import unittest

import numpy as np
from scipy.special import expit

from ambit import AMBIT, MMA, OC, AdaptiveMoment, Continuation, SiMPL, TopOptModel, mbb
from ambit.diagnostics import correlation_length
from ambit.optimizers import mirror_update, project_box_volume
from ambit.stats import cliffs_delta, holm


class TestMirrorStep(unittest.TestCase):
    def test_volume_and_bounds(self):
        rng = np.random.default_rng(0)
        n = 500
        z = rng.normal(size=n)
        d = rng.normal(size=n)
        P = rng.uniform(0.5, 2.0, n)
        dV = np.ones(n)
        V_star = 0.3 * n
        z_new, lam = mirror_update(z, d, P, 0.5, dV, np.sum, V_star, kappa=0.02)
        x = expit(z_new)
        self.assertGreaterEqual(lam, 0.0)
        self.assertTrue(np.all((x > 0) & (x < 1)))
        self.assertAlmostEqual(x.sum(), V_star, delta=1e-6 * V_star)

    def test_inactive_constraint(self):
        n = 100
        z = np.full(n, -3.0)
        z_new, lam = mirror_update(z, np.zeros(n), np.ones(n), 0.1, np.ones(n), np.sum, 0.5 * n)
        self.assertEqual(lam, 0.0)
        self.assertTrue(np.allclose(z_new, z))

    def test_fixed_point(self):
        """If the direction balances the multiplier, the unpolarized step leaves z unchanged."""
        n = 50
        rng = np.random.default_rng(3)
        z = rng.normal(size=n)
        V_star = float(expit(z).sum())
        z_new, lam = mirror_update(z, -2.0 * np.ones(n), np.ones(n), 0.3, np.ones(n), np.sum, V_star)
        self.assertAlmostEqual(lam, 2.0, places=6)
        self.assertTrue(np.allclose(z_new, z, atol=1e-6))

    def test_projection(self):
        rng = np.random.default_rng(1)
        y = rng.normal(0.5, 0.6, 200)
        x = project_box_volume(y, np.sum, 80.0)
        self.assertTrue(np.all((x >= 0) & (x <= 1)))
        self.assertLessEqual(x.sum(), 80.0 + 1e-8)


class TestContinuation(unittest.TestCase):
    def test_literal_beta_rule(self):
        c = Continuation(beta_rule="every_iteration")
        for _ in range(4):
            c.update(1.0, 50.0)
        self.assertEqual(c.beta, 16.0)

    def test_settled_rule(self):
        c = Continuation()
        for _ in range(20):
            c.update(1.0, 50.0)
        self.assertGreater(c.p, 1.0)
        self.assertGreater(c.beta, 1.0)


class TestOptimizers(unittest.TestCase):
    def test_all_methods_feasible(self):
        prob = mbb(40, 14)
        model = TopOptModel(prob, rmin=1.5)
        methods = [OC(), MMA(), SiMPL(), AdaptiveMoment("adam"), AdaptiveMoment("amsgrad"),
                   AdaptiveMoment("adabelief"), AdaptiveMoment("lion"), AMBIT(), AMBIT(fallback=True),
                   AMBIT(mirror=False), AMBIT(metric="belief"), AMBIT(metric="identity", momentum=False)]
        for opt in methods:
            res = opt.run(model, max_iter=40)
            self.assertTrue(np.isfinite(res.compliance), opt.name)
            beta_last = float(res.hist["beta"][-1])
            self.assertLessEqual(model.volume(res.x, beta_last), model.V_star * (1 + 1e-6) + 1e-9, opt.name)

    def test_seed_reproducibility(self):
        prob = mbb(40, 14)
        model = TopOptModel(prob, rmin=1.5)
        a = AMBIT(seed=3).run(model, max_iter=30).compliance
        b = AMBIT(seed=3).run(model, max_iter=30).compliance
        self.assertEqual(a, b)


class TestDiagnostics(unittest.TestCase):
    def test_correlation_length(self):
        rng = np.random.default_rng(0)
        noise = rng.normal(size=(200, 200))
        self.assertLess(correlation_length(noise), 1.0)
        x = np.linspace(0, 6 * np.pi, 200)
        smooth = np.add.outer(np.sin(x), np.sin(x))
        self.assertGreater(correlation_length(smooth), 5.0)

    def test_stats(self):
        self.assertTrue(np.allclose(holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06]))
        d, _ = cliffs_delta([2, 3, 4], [0, 1, 1.5])
        self.assertEqual(d, 1.0)


if __name__ == "__main__":
    unittest.main()
