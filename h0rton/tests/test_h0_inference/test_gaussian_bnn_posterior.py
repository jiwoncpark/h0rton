import unittest
import numpy as np
import pandas as pd
import torch
from h0rton.h0_inference import DiagonalGaussianBNNPosterior, LowRankGaussianBNNPosterior, DoubleGaussianBNNPosterior
import h0rton.trainval_data.data_utils as data_utils


class TestGaussianBNNPosterior(unittest.TestCase):
    """A suite of tests verifying that the input PDFs and the sample distributions
    match.
    
    """
    def test_diagonal_gaussian_bnn_posterior(self):
        """Test the sampling of `DiagonalGaussianBNNPosterior`

        """
        Y_dim = 2
        batch_size = 3
        rank = 2
        sample_seed = 1113
        device = torch.device('cpu')
        mu = np.random.randn(batch_size, Y_dim)
        logvar = np.abs(np.random.randn(batch_size, Y_dim))
        pred = np.concatenate([mu, logvar], axis=1)
        # Get h0rton samples
        diagonal_bnn_post = DiagonalGaussianBNNPosterior(Y_dim, device)
        diagonal_bnn_post.set_sliced_pred(torch.Tensor(pred))
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
        np.testing.assert_array_almost_equal(h0rton_mean, exp_mean, decimal=3)
        np.testing.assert_array_almost_equal(h0rton_covmat, exp_covmat, decimal=3)

    def test_low_rank_gaussian_bnn_posterior(self):
        """Test the sampling of `LowRankGaussianBNNPosterior`

        """
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
        low_rank_bnn_post = LowRankGaussianBNNPosterior(Y_dim, device)
        low_rank_bnn_post.set_sliced_pred(torch.Tensor(pred),)
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

    def test_double_gaussian_bnn_posterior(self):
        """Test the sampling of `DoubleGaussianBNNPosterior`
    
        Note
        ----
        Only compares the true and sample means

        """
        from h0rton.h0_inference import DoubleGaussianBNNPosterior
        from h0rton.losses import sigmoid
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
        double_bnn_post = DoubleGaussianBNNPosterior(Y_dim, device)
        double_bnn_post.set_sliced_pred(torch.Tensor(pred),)
        h0rton_samples = double_bnn_post.sample(10**7, sample_seed)
        # Get h0rton summary stats
        h0rton_mean = np.mean(h0rton_samples, axis=1)
        # Get expected summary stats
        w2 = 0.5*sigmoid(alpha)
        w1 = 1.0 - w2
        exp_mean = mu*w1 + mu2*w2
        np.testing.assert_array_almost_equal(h0rton_mean, exp_mean, decimal=2)

    def test_reverse_transformation_low_rank(self):
        """Verify that sampling from the "whitened" posterior and unwhitening the samples is equivalent to sampling from the "unwhitened" whitened posterior

        """
        sample_seed = 1113
        batch_size = 4
        Y_dim = 2
        rank = 2
        np.random.randn(1113)
        whitening_mean = np.array([0.5, 0.4]).reshape(1, -1)
        whitening_std = np.array([0.1, 0.2]).reshape(1, -1)
        # Method 1: sample from the whitened posterior and unwhiten samples
        # True parameters of whitened posterior
        mu = 3*np.random.randn(batch_size, Y_dim)
        logvar = np.abs(np.random.randn(batch_size, Y_dim))
        F = np.random.randn(batch_size, Y_dim*rank)
        # Input to sampler
        pred = np.concatenate([mu, logvar, F,], axis=1)
        # Get samples
        n_samples = int(1e7)
        bnn_post = LowRankGaussianBNNPosterior(Y_dim, 'cpu', Y_mean=whitening_mean, Y_std=whitening_std)
        bnn_post.set_sliced_pred(torch.Tensor(pred))
        samples = bnn_post.sample(n_samples, sample_seed)
        # Unwhiten samples
        method_1_mean = np.mean(samples, axis=1, keepdims=False)
        method_1_cov = np.empty([batch_size, Y_dim, Y_dim])
        for b in range(batch_size):
            method_1_cov[b, :, :] = np.cov(samples[b, :, :].T, ddof=1)

        # Method 2: sample from the "unwhitened" whitened posterior
        natural_mu = bnn_post.transform_back_mu(bnn_post.mu)
        natural_cov = bnn_post.transform_back_cov_mat(bnn_post.cov_mat)

        #import matplotlib.pyplot as plt
        #plt_idx = 1
        #print(w1[plt_idx])
        #plt.hist(samples[0, :, plt_idx], bins=20, range=[0, 5], density=True)
        #plt.axvline(exp_mean[0, plt_idx], color='r')
        #plt.axvline(mu[0, plt_idx], color='b')
        #plt.axvline(mu2[0, plt_idx], color='g')
        #plt.show()
        np.testing.assert_array_equal(method_1_mean.shape, [batch_size, Y_dim])
        np.testing.assert_array_equal(natural_mu.shape, [batch_size, Y_dim])
        np.testing.assert_array_almost_equal(method_1_mean, natural_mu, decimal=2, err_msg='Transformed-back sample mean does not equal analytically derived mean.')
        np.testing.assert_array_equal(method_1_cov.shape, [batch_size, Y_dim, Y_dim])
        np.testing.assert_array_equal(natural_cov.shape, [batch_size, Y_dim, Y_dim])
        np.testing.assert_array_almost_equal(method_1_cov, natural_cov, decimal=1, err_msg='Transformed-back sample cov does not equal analytically derived cov.')


    def test_reverse_transformation_double(self):
        """Test the reverse transformation of the samples (unwhitening and exponentiating)

        """
        sample_seed = 1113
        batch_size = 2
        Y_dim = 3
        rank = 2
        Y_cols = ['a', 'b', 'c']
        cols_to_whiten = ['a', 'b', 'c'] #['a', 'b']
        cols_to_log_parameterize = []
        col_mapping = dict(zip(Y_cols, np.arange(Y_dim)))
        cols_to_whiten_idx = list(map(col_mapping.get, cols_to_whiten))
        if cols_to_log_parameterize is None:
            cols_to_log_parameterize_idx = None
        else:
            cols_to_log_parameterize_idx = list(map(col_mapping.get, cols_to_log_parameterize))
        mean = np.array([3.0, 2.0, 1.0]).reshape(1, -1)
        std = np.array([2.1, 1.5, 0.5]).reshape(1, -1)
        # First gaussian
        mu = np.arange(batch_size*Y_dim).reshape(batch_size, Y_dim) + 1
        logvar = np.zeros((batch_size, Y_dim))
        F = np.zeros((batch_size, Y_dim*rank))
        F_unraveled = F.reshape(batch_size, Y_dim, rank)
        FFT = np.matmul(F_unraveled, np.swapaxes(F_unraveled, 1, 2))
        # Second gaussian
        mu2 = mu + 1
        logvar2 = np.zeros((batch_size, Y_dim))
        F2 = np.zeros((batch_size, Y_dim*rank))
        F2_unraveled = F2.reshape(batch_size, Y_dim, rank)
        FFT2 = np.matmul(F2_unraveled, np.swapaxes(F2_unraveled, 1, 2))
        alpha = np.random.randn(batch_size, 1)

        # Transform mu, mu2
        mu_trans = pd.DataFrame(mu, columns=['a', 'b', 'c'])
        mu2_trans = pd.DataFrame(mu2, columns=['a', 'b', 'c'])
        if cols_to_log_parameterize is not None:
            mu_trans = data_utils.log_parameterize_Y_cols(mu_trans.copy(), cols_to_log_parameterize)
            mu2_trans = data_utils.log_parameterize_Y_cols(mu2_trans.copy(), cols_to_log_parameterize)
        mu_trans = data_utils.whiten_Y_cols(mu_trans, cols_to_whiten, mean, std)
        mu2_trans = data_utils.whiten_Y_cols(mu2_trans, cols_to_whiten, mean, std)
        mu_trans = mu_trans.values
        mu2_trans = mu2_trans.values

        # Input to sampler
        pred = np.concatenate([mu_trans, logvar, F, mu2_trans, logvar2, F2, alpha], axis=1)

        # Get samples
        n_samples = 10**7
        bnn_post = DoubleGaussianBNNPosterior(Y_dim, 'cpu', whitened_Y_cols_idx=cols_to_whiten_idx, Y_mean=mean, Y_std=std, log_parameterized_Y_cols_idx=cols_to_log_parameterize_idx)
        bnn_post.set_sliced_pred(torch.Tensor(pred),)
        print("======================== reverse")
        samples = bnn_post.sample(n_samples, sample_seed)
        actual_mean = np.mean(samples, axis=1)
        print("=========================")
        
        # Get expected summary stats
        w2 = bnn_post.w2
        w1 = 1.0 - w2
        exp_mean = mu*w1 + mu2*w2
        #import matplotlib.pyplot as plt
        #plt_idx = 1
        #print(w1[plt_idx])
        #plt.hist(samples[0, :, plt_idx], bins=20, range=[0, 5], density=True)
        #plt.axvline(exp_mean[0, plt_idx], color='r')
        #plt.axvline(mu[0, plt_idx], color='b')
        #plt.axvline(mu2[0, plt_idx], color='g')
        #plt.show()
        np.testing.assert_array_almost_equal(actual_mean, exp_mean, decimal=2, err_msg='transformed-back pred not equal to the original Y')

if __name__ == '__main__':
    unittest.main()
