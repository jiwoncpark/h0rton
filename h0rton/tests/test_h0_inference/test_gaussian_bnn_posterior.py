import unittest
import numpy as np
import pandas as pd
import torch

class TestGaussianBNNPosterior(unittest.TestCase):
    """A suite of tests verifying that the input PDFs and the sample distributions
    match.
    
    """
    def test_diagonal_gaussian_bnn_posterior(self):
        """Test the sampling of `DiagonalGaussianBNNPosterior`

        """
        from h0rton.h0_inference import DiagonalGaussianBNNPosterior
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

    def test_reverse_transformation(self):
        """Test the reverse transformation of the samples (unwhitening and exponentiating)

        Note
        ----
        Transformation is done on the predicted output that equals the labels, rather than on the samples.

        """
        from h0rton.h0_inference import DoubleGaussianBNNPosterior
        import h0rton.trainval_data.data_utils as data_utils
        batch_size = 5
        Y_dim = 3
        Y_cols = ['a', 'b', 'c']
        cols_to_whiten = ['a', 'b']
        cols_to_log_parameterize = ['b', 'c']
        col_mapping = dict(zip(Y_cols, np.arange(Y_dim)))
        cols_to_whiten_idx = list(map(col_mapping.get, cols_to_whiten))
        cols_to_log_parameterize_idx = list(map(col_mapping.get, cols_to_log_parameterize))
        mean = np.array([0.5, 0.4]).reshape(1, -1)
        std = np.array([0.6, 0.7]).reshape(1, -1)

        orig_Y = pd.DataFrame(np.abs(np.random.randn(batch_size, Y_dim)), columns=['a', 'b', 'c'])
        data_utils
        # Transform
        trans_Y = data_utils.log_parameterize_Y_cols(orig_Y.copy(), cols_to_log_parameterize)
        trans_Y = data_utils.whiten_Y_cols(trans_Y, cols_to_whiten, mean, std)
        trans_Y = trans_Y.values[:, np.newaxis, :]
        # Reverse transform
        dummy_bnn_post = DoubleGaussianBNNPosterior(Y_dim, 'cpu', whitened_Y_cols_idx=cols_to_whiten_idx, Y_mean=mean, Y_std=std, log_parameterized_Y_cols_idx=cols_to_log_parameterize_idx)
        #print(trans_samples.squeeze())
        returned_pred = dummy_bnn_post.unwhiten_back(torch.Tensor(trans_Y))
        #print(returned_samples.squeeze())
        returned_pred = dummy_bnn_post.exponentiate_back(returned_pred)
        #print(returned_samples.squeeze())

        np.testing.assert_array_almost_equal(returned_pred.squeeze(), orig_Y[Y_cols].values, err_msg='transformed-back pred not equal to the original Y')

if __name__ == '__main__':
    unittest.main()
