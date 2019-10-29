import os, sys
import unittest
import numpy as np
import torch

class TestGaussianBNNPosterior(unittest.TestCase):
    """A suite of tests verifying that the input PDFs and the sample distributions
    match.
    
    """
    def test_diagonal_gaussian_bnn_posterior(self):
        """Test the sampling of `DiagonalGaussianBNNPosterior`

        """
        from h0rton.h0_inference import DiagonalGaussianBNNPosterior
        from scipy.stats import multivariate_normal
        Y_dim = 2
        batch_size = 3
        rank = 2
        sample_seed = 1113
        device = torch.device('cpu')
        mu = np.random.randn(batch_size, Y_dim)
        logvar = np.abs(np.random.randn(batch_size, Y_dim))
        pred = np.concatenate([mu, logvar], axis=1)
        # Get h0rton samples
        diagonal_bnn_post = DiagonalGaussianBNNPosterior(torch.Tensor(pred), Y_dim, device)
        h0rton_samples = diagonal_bnn_post.sample(10**6, sample_seed)
        # Get h0rton summary stats
        h0rton_mean = np.mean(h0rton_samples, axis=1)
        h0rton_covmat = np.zeros((batch_size, Y_dim, Y_dim))
        exp_covmat = np.zeros((batch_size, Y_dim, Y_dim))
        for b in range(batch_size):
            cov_b = np.cov(h0rton_samples[b, :, :].swapaxes(0, 1), ddof=0)
            h0rton_covmat[b, :, :] = cov_b
            exp_covmat[b, :, :] += np.diagflat(np.exp(logvar[b, :]))
        # Get expected summary stats
        exp_mean = mu
        np.testing.assert_array_almost_equal(h0rton_mean, exp_mean, decimal=2)
        np.testing.assert_array_almost_equal(h0rton_covmat, exp_covmat, decimal=2)

    def test_low_rank_gaussian_bnn_posterior(self):
        """Test the sampling of `LowRankGaussianBNNPosterior`

        """
        from h0rton.h0_inference import LowRankGaussianBNNPosterior
        from scipy.stats import multivariate_normal
        Y_dim = 2
        batch_size = 3
        rank = 2
        sample_seed = 1113
        device = torch.device('cpu')
        mu = np.random.randn(batch_size, Y_dim)
        logvar = np.abs(np.random.randn(batch_size, Y_dim))
        F = np.random.randn(batch_size, Y_dim*rank)
        F_unraveled = F.reshape(batch_size, Y_dim, rank)
        FFT = np.matmul(F_unraveled, np.swapaxes(F_unraveled, 1, 2))
        pred = np.concatenate([mu, logvar, F], axis=1)
        # Get h0rton samples
        low_rank_bnn_post = LowRankGaussianBNNPosterior(torch.Tensor(pred), Y_dim, device)
        h0rton_samples = low_rank_bnn_post.sample(10**7, sample_seed)
        # Get h0rton summary stats
        h0rton_mean = np.mean(h0rton_samples, axis=1)
        h0rton_covmat = np.empty((batch_size, Y_dim, Y_dim))
        exp_covmat = FFT
        for b in range(batch_size):
            cov_b = np.cov(h0rton_samples[b, :, :].swapaxes(0, 1), ddof=0)
            h0rton_covmat[b, :, :] = cov_b
            exp_covmat[b, :, :] += np.diagflat(np.exp(logvar[b, :]))
        # Get expected summary stats
        exp_mean = mu
        np.testing.assert_array_almost_equal(h0rton_mean, exp_mean, decimal=2)
        np.testing.assert_array_almost_equal(h0rton_covmat, exp_covmat, decimal=2)

    def test_double_gaussian_bnn_posterior(self):
        """Test the sampling of `DoubleGaussianBNNPosterior`
    
        Note
        ----
        Only compares the true and sample means

        """
        from h0rton.h0_inference import DoubleGaussianBNNPosterior
        from h0rton.losses import sigmoid
        from scipy.stats import multivariate_normal
        Y_dim = 2
        batch_size = 3
        rank = 2
        sample_seed = 1113
        device = torch.device('cpu')
        # First gaussian
        mu = np.random.randn(batch_size, Y_dim)
        logvar = np.abs(np.random.randn(batch_size, Y_dim))
        F = np.random.randn(batch_size, Y_dim*rank)
        F_unraveled = F.reshape(batch_size, Y_dim, rank)
        FFT = np.matmul(F_unraveled, np.swapaxes(F_unraveled, 1, 2))
        # Second gaussian
        mu2 = np.random.randn(batch_size, Y_dim)
        logvar2 = np.abs(np.random.randn(batch_size, Y_dim))
        F2 = np.random.randn(batch_size, Y_dim*rank)
        F2_unraveled = F2.reshape(batch_size, Y_dim, rank)
        FFT2 = np.matmul(F2_unraveled, np.swapaxes(F2_unraveled, 1, 2))
        alpha = np.random.randn(batch_size, 1)
        pred = np.concatenate([mu, logvar, F, mu2, logvar2, F2, alpha], axis=1)
        # Get h0rton samples
        double_bnn_post = DoubleGaussianBNNPosterior(torch.Tensor(pred), Y_dim, device)
        h0rton_samples = double_bnn_post.sample(10**7, sample_seed)
        # Get h0rton summary stats
        h0rton_mean = np.mean(h0rton_samples, axis=1)
        # Get expected summary stats
        w2 = 0.5*sigmoid(alpha)
        w1 = 1.0 - w2
        exp_mean = mu*w1 + mu2*w2
        np.testing.assert_array_almost_equal(h0rton_mean, exp_mean, decimal=2)

if __name__ == '__main__':
    unittest.main()
