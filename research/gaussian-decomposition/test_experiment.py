import unittest

import numpy as np
from scipy.stats import kstest

from experiment import choose_counts, estimate, grid_points, latin, method_names, problems


class DecompositionTests(unittest.TestCase):
    def test_exact_count_and_reproducibility(self):
        for problem in problems():
            for method in method_names(problem):
                value,n = estimate(problem,128,51,method)
                self.assertTrue(np.isfinite(value))
                self.assertEqual(n,128)
                self.assertEqual((value,n),estimate(problem,128,51,method))

    def test_neyman_allocation_has_full_support(self):
        for scores in (np.zeros(16),np.arange(16),np.r_[100.,np.zeros(15)]):
            counts = choose_counts(scores,80)
            self.assertEqual(counts.sum(),80)
            self.assertGreaterEqual(counts.min(),1)

    def test_lhs_marginal_strata(self):
        from scipy.special import ndtr
        z = latin(np.random.default_rng(8),256,16)
        bins = np.floor(256*ndtr(z)).astype(int)
        np.testing.assert_array_equal(np.sort(bins,axis=0),np.broadcast_to(np.arange(256)[:,None],bins.shape))

    def test_joint_gaussian_independent_replicates(self):
        # Take one uniformly sampled row per independent design for valid KS p values.
        problem = next(p for p in problems() if p.name=='projection 5D')
        sections = {'grid': [], 'marginal': []}
        for seed in range(2400):
            rng = np.random.default_rng(813000+seed)
            grid = grid_points(rng,problem,np.full(16,2),(4,4))
            marginal = latin(rng,32,20).reshape(32,4,5).sum(axis=1)/2
            sections['grid'].append(grid[rng.integers(32)])
            sections['marginal'].append(marginal[rng.integers(32)])
        for values in sections.values():
            x = np.array(values)
            for column in x.T:
                self.assertGreater(kstest(column,'norm').pvalue,.001)
            self.assertGreater(kstest((x[:,0]-x[:,1])/np.sqrt(2),'norm').pvalue,.001)
            np.testing.assert_allclose(np.cov(x.T),np.eye(5),atol=.09)

    def test_weighted_estimates_and_antithetic_identity(self):
        linear,carr = list(problems())[:2]
        for method in ('LHS k2 antith.','Antithétique'):
            self.assertLess(abs(estimate(linear,512,7,method)[0]),1e-14)
        for method in ('Grille k2','Grille k2 Neyman','Projection Neyman','LHS k2 remise'):
            values = np.array([estimate(carr,512,850000+i,method)[0] for i in range(250)])
            se = values.std(ddof=1)/np.sqrt(len(values))
            self.assertLess(abs(values.mean()-carr.mean),4.5*se)


if __name__=='__main__':
    unittest.main()
