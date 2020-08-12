import unittest
import numpy as np
import torch
from h0rton.h0_inference.gaussian_bnn_posterior_cpu import DiagonalGaussianBNNPosteriorCPU, LowRankGaussianBNNPosteriorCPU, DoubleLowRankGaussianBNNPosteriorCPU, FullRankGaussianBNNPosteriorCPU, DoubleGaussianBNNPosteriorCPU
from h0rton.h0_inference.gaussian_bnn_posterior_cpu import sigmoid

class TestGaussianBNNPosteriorCPU(unittest.TestCase):
    """A suite of tests verifying that the input PDFs and the sample distributions
    match.
    
    """
    def test_diagonal_gaussian_bnn_posterior_cpu(self):
        """Test the sampling of `DiagonalGaussianBNNPosteriorCPU`

        """
        Y_dim = 2
        batch_size = 3
        rank = 2
        sample_seed = 1113
        mu = np.random.randn(batch_size, Y_dim)
        logvar = np.abs(np.random.randn(batch_size, Y_dim))
        pred = np.concatenate([mu, logvar], axis=1)
        # Get h0rton samples
        #Y_mean = np.random.randn(batch_size, Y_dim)
        #Y_std = np.abs(np.random.randn(batch_size, Y_dim))
        Y_mean = np.zeros(Y_dim)
        Y_std = np.ones(Y_dim)
        diagonal_bnn_post = DiagonalGaussianBNNPosteriorCPU(Y_dim, Y_mean, Y_std)
        diagonal_bnn_post.set_sliced_pred(pred)
        h0rton_samples = diagonal_bnn_post.sample(10**7, sample_seed)
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

    def test_low_rank_gaussian_bnn_posterior_cpu(self):
        """Test the sampling of `LowRankGaussianBNNPosteriorCPU`

        """
        Y_dim = 2
        batch_size = 3
        rank = 2
        sample_seed = 1113
        mu = np.random.randn(batch_size, Y_dim)
        logvar = np.abs(np.random.randn(batch_size, Y_dim))
        F = np.random.randn(batch_size, Y_dim*rank)
        F_unraveled = F.reshape(batch_size, Y_dim, rank)
        FFT = np.matmul(F_unraveled, np.swapaxes(F_unraveled, 1, 2))
        pred = np.concatenate([mu, logvar, F], axis=1)
        # Get h0rton samples
        #Y_mean = np.random.randn(batch_size, Y_dim)
        #Y_std = np.abs(np.random.randn(batch_size, Y_dim))
        Y_mean = np.zeros(Y_dim)
        Y_std = np.ones(Y_dim)
        low_rank_bnn_post = LowRankGaussianBNNPosteriorCPU(Y_dim, Y_mean, Y_std)
        low_rank_bnn_post.set_sliced_pred(pred)
        h0rton_samples = low_rank_bnn_post.sample(10**7, sample_seed)
        #import matplotlib.pyplot as plt
        #plt.hist(h0rton_samples[0, :, 0], bins=30)
        #plt.axvline(mu[0, 0], color='r')
        #plt.show()
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

    def test_double_low_rank_gaussian_bnn_posterior_cpu(self):
        """Test the sampling of `DoubleLowRankGaussianBNNPosteriorCPU`
    
        Note
        ----
        Only compares the true and sample means

        """
        Y_dim = 2
        batch_size = 3
        rank = 2
        sample_seed = 1113
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
        #Y_mean = np.random.randn(batch_size, Y_dim)
        #Y_std = np.abs(np.random.randn(batch_size, Y_dim))
        Y_mean = np.zeros(Y_dim)
        Y_std = np.ones(Y_dim)
        double_bnn_post = DoubleLowRankGaussianBNNPosteriorCPU(Y_dim, Y_mean, Y_std)
        double_bnn_post.set_sliced_pred(pred,)
        h0rton_samples = double_bnn_post.sample(10**7, sample_seed)
        # Get h0rton summary stats
        h0rton_mean = np.mean(h0rton_samples, axis=1)
        # Get expected summary stats
        w2 = 0.5*sigmoid(alpha)
        w1 = 1.0 - w2
        exp_mean = mu*w1 + mu2*w2
        np.testing.assert_array_almost_equal(h0rton_mean, exp_mean, decimal=2)

    def test_full_rank_gaussian_bnn_posterior_cpu(self):
        """Test the sampling of `FullRankGaussianBNNPosteriorCPU`

        """
        Y_dim = 2
        batch_size = 3
        tril_idx = np.tril_indices(Y_dim)
        tril_len = len(tril_idx[0])
        sample_seed = 1113
        # Get h0rton evaluation
        mu = np.random.randn(batch_size, Y_dim)
        tril_elements = np.random.randn(batch_size, tril_len)
        pred = np.concatenate([mu, tril_elements], axis=1).astype(np.float32)
        # Get h0rton samples
        #Y_mean = np.random.randn(batch_size, Y_dim)
        #Y_std = np.abs(np.random.randn(batch_size, Y_dim))
        Y_mean = np.zeros(Y_dim)
        Y_std = np.ones(Y_dim)
        post = FullRankGaussianBNNPosteriorCPU(Y_dim, Y_mean, Y_std)
        post.set_sliced_pred(pred)
        h0rton_samples = post.sample(10**7, sample_seed)
        #import matplotlib.pyplot as plt
        #plt.hist(h0rton_samples[0, :, 0], bins=30)
        #plt.axvline(mu[0, 0], color='r')
        #plt.show()
        # Get h0rton summary stats
        h0rton_mean = np.mean(h0rton_samples, axis=1)
        h0rton_covmat = np.empty((batch_size, Y_dim, Y_dim))
        np_covmat = np.empty((batch_size, Y_dim, Y_dim))
        for b in range(batch_size):
            # Cov mat calculated from H0rton samples
            cov_b = np.cov(h0rton_samples[b, :, :].swapaxes(0, 1), ddof=0)
            h0rton_covmat[b, :, :] = cov_b
            # Cov mat expected from the PDF
            tril = np.zeros((Y_dim, Y_dim))
            tril[tril_idx[0], tril_idx[1]] = tril_elements[b, :]
            log_diag_tril = np.diagonal(tril, offset=0, axis1=0, axis2=1).copy()
            tril[np.eye(Y_dim).astype(bool)] = np.exp(log_diag_tril)
            prec_mat = np.dot(tril, tril.T)
            cov_mat = np.linalg.inv(prec_mat)
            np_covmat[b, :, :] = cov_mat
        # Get expected summary stats
        np_mean = mu
        np.testing.assert_array_almost_equal(h0rton_mean, np_mean, decimal=2)
        np.testing.assert_array_almost_equal(h0rton_covmat, np_covmat, decimal=2)

    def test_double_gaussian_bnn_posterior_cpu(self):
        """Test the sampling of `DoubleGaussianBNNPosteriorCPU`
    
        Note
        ----
        Only compares the true and sample means

        """
        Y_dim = 2
        batch_size = 3
        tril_idx = np.tril_indices(Y_dim)
        tril_len = len(tril_idx[0])
        sample_seed = 1113
        # Get h0rton evaluation
        # First gaussian
        mu = np.random.randn(batch_size, Y_dim)
        tril_elements = np.random.randn(batch_size, tril_len)
        # Second gaussian
        mu2 = np.random.randn(batch_size, Y_dim)
        tril_elements2 = np.random.randn(batch_size, tril_len)
        alpha = np.random.randn(batch_size, 1)
        pred = np.concatenate([mu, tril_elements, mu2, tril_elements2, alpha], axis=1)
        # Get h0rton samples
        #Y_mean = np.random.randn(batch_size, Y_dim)
        #Y_std = np.abs(np.random.randn(batch_size, Y_dim))
        Y_mean = np.zeros(Y_dim)
        Y_std = np.ones(Y_dim)
        post = DoubleGaussianBNNPosteriorCPU(Y_dim, Y_mean, Y_std)
        post.set_sliced_pred(pred,)
        h0rton_samples = post.sample(10**7, sample_seed)
        # Get h0rton summary stats
        h0rton_mean = np.mean(h0rton_samples, axis=1)
        # Get expected summary stats
        w2 = 0.5*sigmoid(alpha)
        w1 = 1.0 - w2
        np_mean = mu*w1 + mu2*w2
        np.testing.assert_array_almost_equal(h0rton_mean, np_mean, decimal=2)

if __name__ == '__main__':
    unittest.main()
