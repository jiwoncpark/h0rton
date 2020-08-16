import unittest
import numpy as np
from scipy.stats import norm, lognorm, multivariate_normal
from h0rton.h0_inference import h0_utils

class TestH0Utils(unittest.TestCase):
    """A suite of tests verifying that the input PDFs and the sample distributions
    match.
    
    """

    @classmethod
    def setUpClass(cls):
        np.random.seed(331)
        cls.Y_dim = 2
        cls.pred_mu = np.random.randn(cls.Y_dim) #np.zeros(cls.Y_dim) #
        # Some unique non-singular cov mat
        L = np.tril(np.random.randn(cls.Y_dim, cls.Y_dim)) #np.eye(cls.Y_dim) 
        cls.pred_cov = np.matmul(L, L.T)
        cls.pred_cov[np.diag_indices(cls.Y_dim)] = np.abs(cls.pred_cov[np.diag_indices(cls.Y_dim)])
        cls.shift = np.random.randn(cls.Y_dim)
        cls.scale = np.abs(np.random.randn(cls.Y_dim))

    def setUp(self):
        np.random.seed(331)

    def test_gaussian_ll_pdf(self):
        """Test the unnormalized log normal PDF

        """
        m = 0.5
        s = 1
        eval_x = np.linspace(-3, 3, 100)
        truth = norm.logpdf(eval_x, m, s)
        # Note output of gaussian_nll_pdf is unnormalized
        pred = h0_utils.gaussian_ll_pdf(eval_x, m, s) - np.log(s) - 0.5*np.log(2.0*np.pi) 
        np.testing.assert_array_almost_equal(pred, truth)

    def test_pred_to_natural_gaussian(self):
        """Test if the predicted mu, cov are being transformed back correctly into natural (original) space

        """
        # Generate pred_X ~ N(mu, cov)
        pred_X = multivariate_normal.rvs(mean=self.pred_mu, cov=self.pred_cov, size=1000000, random_state=331)
        # Transform back to get orig_X
        orig_X = pred_X*self.scale.reshape([1, -1]) + self.shift.reshape([1, -1])
        # Get mu, cov of orig_X
        orig_mu = np.mean(orig_X, axis=0)
        orig_cov = np.cov(orig_X, rowvar=0, bias=False)
        actual_orig_mu, actual_orig_cov = h0_utils.pred_to_natural_gaussian(self.pred_mu, self.pred_cov, self.shift, self.scale)
        np.testing.assert_array_almost_equal(actual_orig_mu, orig_mu, decimal=2, err_msg="transformed-back mu")
        np.testing.assert_array_almost_equal(actual_orig_cov, orig_cov, decimal=2, err_msg="transformed-back cov")

    def test_reorder_to_tdlmc(self):
        """Test if reordering the lenstronomy dec array --> increasing dec --> ABCD ordering of TDLMC is correct

        """
        # D
        # B
        # A
        # C
        img_array = np.array([0.7, -0.3, 0.8, -1.2])
        increasing_dec_i = np.argsort(img_array)
        abcd_ordering_i = np.array([1, 2, 0, 3]) # send abcd_ordering[i] to ith index
        reordered_actual = h0_utils.reorder_to_tdlmc(img_array, increasing_dec_i, abcd_ordering_i)
        reordered = np.array([-0.3, 0.7, -1.2, 0.8])
        np.testing.assert_array_equal(reordered, reordered_actual, err_msg="reorder_to_tdlmc") 

    def test_CosmoConverter(self):
        """Test that CosmoConverter can handle array types 

        """
        random_z = np.random.rand(2)*2.0 + 0.5 # sample two redshifts in range [0.5, 2.5]
        cosmo_converter = h0_utils.CosmoConverter(z_lens=min(random_z), z_src=max(random_z), H0=70.0, Om0=0.3)
        H0_candidates = np.linspace(50, 90, 100)
        actual_D_dt = cosmo_converter.get_D_dt(H0_candidates)
        recovered_H0 = cosmo_converter.get_H0(actual_D_dt)
        np.testing.assert_array_almost_equal(recovered_H0, H0_candidates, decimal=10, err_msg="test_CosmoConverter") 

    def test_get_lognormal_stats(self):
        """Test if the correct lognormal stats are generated from lognormal samples

        """
        mu_in = 8.5
        sigma_in = 0.3
        n_samples = int(1e7)
        samples = np.exp(np.random.randn(n_samples)*sigma_in + mu_in)
        samples[0] = np.nan
        stats = h0_utils.get_lognormal_stats(samples)
        np.testing.assert_almost_equal(stats['mu'], mu_in, decimal=2)
        np.testing.assert_almost_equal(stats['sigma'], sigma_in, decimal=2)
        #plotting_utils.plot_D_dt_histogram(samples, lens_i=99999, true_D_dt=stats['mode'], save_dir='.')

    def test_get_lognormal_stats_naive(self):
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
        samples[0] = np.nan
        weights[0] = np.nan
        weights[1] = -np.inf
        stats = h0_utils.get_lognormal_stats_naive(samples, weights)
        np.testing.assert_almost_equal(stats['mu'], mu_in, decimal=1)
        np.testing.assert_almost_equal(stats['sigma'], sigma_in, decimal=1)
        #plotting_utils.plot_weighted_D_dt_histogram(samples, weights, lens_i=99999, true_D_dt=stats['mode'], save_dir='.')

    def test_get_normal_stats(self):
        """Test if the correct normal stats are generated from normal samples

        """
        mean_in = np.random.randn()
        std_in = np.random.rand() + 1.0 # range [1, 2]
        n_samples = int(1e7)
        norm_obj = norm(loc=mean_in, scale=std_in)
        samples = np.random.normal(mean_in, std_in, size=n_samples)
        # Corrupt the samples
        samples[0] = np.nan
        samples[1] = -np.inf
        stats = h0_utils.get_normal_stats(samples)
        np.testing.assert_almost_equal(stats['mean'], mean_in, decimal=2)
        np.testing.assert_almost_equal(stats['std'], std_in, decimal=2)

    def test_get_normal_stats_naive(self):
        """Test if the correct normal stats are generated from uniform samples with normal weights

        """
        mean_in = np.random.randn()
        std_in = np.random.rand() + 1.0 # range [1, 2]
        n_samples = int(1e7)
        norm_obj = norm(loc=mean_in, scale=std_in)
        min_val = -10
        max_val = 10
        samples = np.random.rand(n_samples)*(max_val - min_val) + min_val
        weights = norm_obj.pdf(samples)
        # Corrupt the samples and weights
        samples[0] = np.nan
        weights[0] = np.nan
        weights[1] = -np.inf
        stats = h0_utils.get_normal_stats_naive(samples, weights)
        np.testing.assert_almost_equal(stats['mean'], mean_in, decimal=2)
        np.testing.assert_almost_equal(stats['std'], std_in, decimal=2)

    def test_remove_outliers_from_lognormal(self):
        """Test if extreme outliers 3-STD away from the mean are removed

        """
        mu_in = 8.5
        sigma_in = 0.3
        n_samples = int(1e5)
        lognorm_obj = lognorm(scale=np.exp(mu_in), s=sigma_in)
        samples = lognorm_obj.rvs(n_samples)
        cleaned = h0_utils.remove_outliers_from_lognormal(samples, level=3)
        np.testing.assert_almost_equal(0.997, len(cleaned)/n_samples, decimal=3)

    def test_combine_lenses(self):
        """Test execution of HierArc combination of lenses. Results aren't validated here, only in HierArc.

        """
        n_lenses = 7
        random_z = np.random.rand(7, 2)*2.0 + 0.5 # sample two redshifts in range [0.5, 2.5]
        z_lens = np.amin(random_z, axis=0)
        z_src = np.amax(random_z, axis=0)
        ddt_mean = np.random.normal(5000, 1000, size=n_lenses)
        ddt_mu = np.log(ddt_mean)
        mcmc_samples_normal, log_prob_cosmo_normal = h0_utils.combine_lenses('DdtGaussian', z_lens, z_src, true_Om0=0.2, samples_save_path=None, corner_save_path=None, n_run=1, n_burn=4, n_walkers=3, ddt_mean=ddt_mean, ddt_sigma=np.random.normal(500, 5, size=n_lenses))
        mcmc_samples_lognormal, log_prob_cosmo_lognormal = h0_utils.combine_lenses('DdtLogNorm', z_lens, z_src, true_Om0=0.2, samples_save_path=None, corner_save_path=None, n_run=1, n_burn=4, n_walkers=3, ddt_mu=ddt_mu, ddt_sigma=np.random.normal(1, 0.5, size=n_lenses))

if __name__ == '__main__':
    unittest.main()