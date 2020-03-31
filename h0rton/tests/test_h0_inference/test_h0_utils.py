import unittest
import numpy as np
from scipy.stats import lognorm
from h0rton.h0_inference import h0_utils, plotting_utils

class TestH0Utils(unittest.TestCase):
    """A suite of tests verifying that the input PDFs and the sample distributions
    match.
    
    """

    def setUp(self):
        np.random.seed(331)

    def test_get_lognormal_stats_no_weights(self):
        """Test if the correct lognormal stats are generated from lognormal samples

        """
        mu_in = 8.5
        sigma_in = 0.3
        n_samples = int(1e7)
        samples = np.exp(np.random.randn(n_samples)*sigma_in + mu_in)
        stats = h0_utils.get_lognormal_stats(samples)
        np.testing.assert_almost_equal(stats['mu'], mu_in, 2)
        np.testing.assert_almost_equal(stats['sigma'], sigma_in, 2)
        #plotting_utils.plot_D_dt_histogram(samples, lens_i=99999, true_D_dt=stats['mode'], save_dir='.')

    def test_get_lognormal_stats_weighted(self):
        """Test if the correct lognormal stats are generated from uniform samples with lognormal weights

        """
        mu_in = 8.5
        sigma_in = 0.3
        n_samples = int(1e7)
        lognorm_obj = lognorm(scale=np.exp(mu_in), s=sigma_in)
        min_val = 0.0
        max_val = 10000.0
        samples = np.random.rand(n_samples)*(max_val - min_val) + min_val
        weights = lognorm_obj.pdf(samples)
        stats = h0_utils.get_lognormal_stats(samples, weights)
        np.testing.assert_almost_equal(stats['mu'], mu_in, 1)
        np.testing.assert_almost_equal(stats['sigma'], sigma_in, 1)
        #plotting_utils.plot_weighted_D_dt_histogram(samples, weights, lens_i=99999, true_D_dt=stats['mode'], save_dir='.')

if __name__ == '__main__':
    unittest.main()