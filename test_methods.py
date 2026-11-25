import unittest
from dataclasses import replace

import numpy as np
from scipy.special import ndtri
from scipy.stats import norm, t

from methods import Config, METHODS, draw, risk, run, surrogate
from models import MODELS, TAIL


class MethodTests(unittest.TestCase):
    def test_var_es_atoms_and_fractional_mass(self):
        loss = np.r_[np.zeros(990), np.full(5, 4.), np.full(5, 10.)]
        np.testing.assert_allclose(risk(loss), [0, 7], atol=1e-13)
        np.testing.assert_allclose(risk([0, 5, 10], np.array([.986, .009, .005])), [5, 7.5])
        np.testing.assert_allclose(risk(loss), risk(loss, np.full(len(loss), 2.)))

    def test_student_transform_and_reference(self):
        z = np.array([[-6.], [-2.], [0.], [2.], [6.]])
        np.testing.assert_allclose(MODELS['student'].value(z), np.sign(z[:, 0]) * t.isf(norm.sf(np.abs(z[:, 0])), 5), rtol=1e-9, atol=1e-12)
        q, es = MODELS['student'].analytic()
        np.testing.assert_allclose(t.expect(lambda x: x, args=(5,), lb=q)/TAIL, es, rtol=1e-9)

    def test_option_quadrature_reference(self):
        from benchmark import reference
        ref = reference(MODELS['options_1d'], 12)
        q, es = ref['value']
        self.assertTrue(16 < q < es < 21)
        self.assertLess(max(ref['truncation_delta']), 1e-8)

    def test_mixture_reference(self):
        q, es = MODELS['mixture'].analytic()
        self.assertAlmostEqual(.88*norm.sf(q)+.12*norm.sf(q, 2.8, 1.7), .01, places=12)
        self.assertGreater(es, q)

    def test_gaussian_proposal_mass_and_moments(self):
        model = MODELS['options_4d']
        for kind in ('is', 'is_strat'):
            z, weight = draw(np.random.default_rng(177), 200_000, model, Config(kind))
            self.assertTrue(np.all((weight > 0) & (weight <= 5. + 1e-14)))
            self.assertAlmostEqual(weight.mean(), 1., delta=.018)
            np.testing.assert_allclose(np.mean(z*weight[:, None], axis=0), 0., atol=.018)
            self.assertAlmostEqual(np.mean(weight*(z[:, 0] > 2)), norm.sf(2), delta=.001)

    def test_stratified_normal_draw(self):
        u = (np.arange(256) + .5)/256
        rng = np.random.default_rng(3)
        z, weight = draw(rng, len(u), MODELS['normal'], Config('strat'))
        self.assertIsNone(weight)
        np.testing.assert_array_equal(np.floor(norm.cdf(z[:, 0])*len(u)), np.arange(len(u)))
        np.testing.assert_allclose(np.mean(z), 0., atol=.02)

    def test_split_preserves_covariance_after_cholesky(self):
        model = MODELS['options_4d']
        z, _ = draw(np.random.default_rng(51), 40_000, model, Config('split'))
        x = model.coordinates(z)
        np.testing.assert_allclose(np.cov(x.T), model.cholesky @ model.cholesky.T, atol=.027)

    def test_conditional_split_recovers_stratified_sum(self):
        model = MODELS['options_4d']
        z, _ = draw(np.random.default_rng(915), 1024, model, Config('strat'))
        split, w = draw(np.random.default_rng(915), 1024, model, Config('split_strat'))
        self.assertIsNone(w)
        np.testing.assert_allclose(split, z, atol=2e-15)

    def test_exact_budget_and_reproducibility(self):
        rng = np.random.default_rng(5)
        for name, config in METHODS.items():
            with self.subTest(name=name):
                seed = rng.integers(1_000_000)
                model = MODELS['options_4d']
                value, diag = run(model, config, 512, seed)
                np.testing.assert_array_equal(value, run(model, config, 512, seed)[0])
                self.assertEqual(diag['exact'], 512 if config.corrected else config.nodes+32)
                if config.grid:
                    self.assertEqual(diag['training'], config.nodes)
                    self.assertEqual(diag['checks'], 32)
                    self.assertGreaterEqual(diag['fit_rmse'], 0.)

    def test_linear_interpolation_inside_and_outside_grid(self):
        u = np.array([1e-14, .01, .5, .99, 1-1e-14])
        z = ndtri(u)[:, None]
        model = MODELS['normal']
        for grid in ('uniform', 'quantile', 'tail', 'adaptive'):
            cfg = replace(Config(grid=grid), width=4.)
            predict, _ = surrogate(model, cfg)
            fitted, outside = predict(z)
            np.testing.assert_allclose(fitted, model.value(z), atol=5e-15)
            self.assertTrue(outside[0] and outside[-1])


if __name__ == '__main__':
    unittest.main()
