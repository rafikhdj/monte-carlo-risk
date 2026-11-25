import unittest

import numpy as np
from scipy.special import ndtr

from research.benchmarks.run import methods
from research.estimators import Evaluator, draws, fit, run
from research.problems import PROBLEMS, reference


class ExpectationTest(unittest.TestCase):
    def test_budget_reproducible_and_finite(self):
        for key in ('polynomial','asian_nested','interaction_20d'):
            p = PROBLEMS[key]
            for name,method in methods(p).items():
                with self.subTest(problem=key,method=name):
                    first = run(p,method,128,381)
                    self.assertEqual(first,run(p,method,128,381))
                    self.assertTrue(np.isfinite(first[0]))
                    self.assertLessEqual(first[1],128)
                    if name not in ('Quadrature','Gauss-Hermite','Gauss-Kronrod adaptatif'):
                        self.assertEqual(first[1],128)

    def test_weights_and_rare_event(self):
        p = PROBLEMS['rare_tail']
        x,w = draws('is',100_000,p,np.random.default_rng(7))
        self.assertTrue(np.all((w>0)&(w<=5.)))
        self.assertAlmostEqual(w.mean(),1.,delta=.025)
        self.assertAlmostEqual(np.mean(w*p.integrand(x)),ndtr(-3.4),delta=.000014)

    def test_surrogate_mean_against_independent_rule(self):
        p = PROBLEMS['interaction_2d']
        pred,integral = fit(Evaluator(p),'auto',80,p,np.random.default_rng(101))
        x,_ = draws('sobol',2**17,p,np.random.default_rng(125))
        self.assertAlmostEqual(pred(x).mean(),integral,delta=.00006)

    def test_conditional_asian_reference(self):
        ref = reference(PROBLEMS['asian_nested'])
        self.assertTrue(.04 < ref['value'] < .25)
        self.assertLess(ref['uncertainty'],1e-7)
        self.assertLess(ref['inner_delta'],1e-7)


if __name__=='__main__':
    unittest.main()
