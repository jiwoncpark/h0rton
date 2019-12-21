import os, sys
import numpy as np
import unittest
import torch

class TestGaussianNLL(unittest.TestCase):
    """A suite of tests verifying the PDF evaluation of GaussianNLL
    
    """

    @classmethod
    def setUpClass(cls):
        """Set global defaults for tests

        """
        torch.set_default_tensor_type(torch.DoubleTensor)

    def test_diagonal_gaussian_nll(self):
        """Test the PDF evaluation of a single Gaussian with diagonal covariance matrix

        """
        from h0rton.losses import DiagonalGaussianNLL
        from scipy.stats import multivariate_normal
        # Instantiate NLL class
        Y_dim = 2
        device = torch.device('cpu')
        diagonal_gaussian_nll = DiagonalGaussianNLL(Y_dim, device)
        # Get h0rton evaluation
        batch_size = 3
        target = np.random.randn(batch_size, Y_dim)
        mu = np.random.randn(batch_size, Y_dim)
        logvar = np.abs(2.0*np.random.randn(batch_size, Y_dim))
        pred = np.concatenate([mu, logvar], axis=1)
        h0rton_nll = diagonal_gaussian_nll(torch.from_numpy(pred), torch.from_numpy(target))
        # Get scipy evaluation
        matched_nll = 0.0
        for b in range(batch_size):
            target_b = target[b, :]
            mu_b = mu[b, :]
            cov_b = np.diagflat(np.exp(logvar[b, :]))
            nll = -np.log(multivariate_normal.pdf(target_b, mean=mu_b, cov=cov_b)) # real nll, not scaled and shifted
            matched_nll += nll/batch_size
            #matched_nll += (2.0*nll - Y_dim*np.log(2.0*np.pi))/batch_size # kernel version
        np.testing.assert_array_almost_equal(h0rton_nll, matched_nll, decimal=6)

    def test_low_rank_gaussian_nll(self):
        """Test the PDF evaluation of a single Gaussian with a full but low-rank plus diagonal covariance matrix

        """
        from h0rton.losses import LowRankGaussianNLL
        from scipy.stats import multivariate_normal
        # Instantiate NLL class
        Y_dim = 2
        rank = 2
        device = torch.device('cpu')
        low_rank_gaussian_nll = LowRankGaussianNLL(Y_dim, device)
        # Get h0rton evaluation
        batch_size = 3
        target = np.random.randn(batch_size, Y_dim)
        mu = np.random.randn(batch_size, Y_dim)
        logvar = np.abs(np.random.randn(batch_size, Y_dim))
        F = np.random.randn(batch_size, rank*Y_dim)
        pred = np.concatenate([mu, logvar, F], axis=1)
        h0rton_nll = low_rank_gaussian_nll(torch.from_numpy(pred), torch.from_numpy(target))
        # Get scipy evaluation
        matched_nll = 0.0
        for b in range(batch_size):
            target_b = target[b, :]
            mu_b = mu[b, :]
            diag_b = np.diagflat(np.exp(logvar[b, :]))
            F_b = F[b, :].reshape(Y_dim, rank)
            low_rank_b =  np.matmul(F_b, F_b.T)
            nll = -np.log(multivariate_normal.pdf(target_b, mean=mu_b, cov=diag_b + low_rank_b)) # real nll, not scaled and shifted
            matched_nll += nll/batch_size
            #matched_nll += (2.0*nll - Y_dim*np.log(2.0*np.pi))/batch_size # kernel version
        np.testing.assert_array_almost_equal(h0rton_nll, matched_nll, decimal=6)

    def test_double_gaussian_nll(self):
        """Test the PDF evaluation of a mixture of two Gaussians, each with a full but low-rank plus diagonal covariance matrix

        """
        from h0rton.losses import DoubleGaussianNLL, sigmoid
        from scipy.stats import multivariate_normal
        # Instantiate NLL class
        Y_dim = 2
        rank = 2
        device = torch.device('cpu')
        double_gaussian_nll = DoubleGaussianNLL(Y_dim, device)
        # Get h0rton evaluation
        batch_size = 3
        target = np.random.randn(batch_size, Y_dim)
        mu = np.random.randn(batch_size, Y_dim)
        logvar = np.abs(np.random.randn(batch_size, Y_dim))
        F = np.random.randn(batch_size, rank*Y_dim)
        mu2 = np.random.randn(batch_size, Y_dim)
        logvar2 = np.abs(np.random.randn(batch_size, Y_dim))
        F2 = np.random.randn(batch_size, rank*Y_dim)
        alpha = np.random.randn(batch_size, 1)

        pred = np.concatenate([mu, logvar, F, mu2, logvar2, F2, alpha], axis=1)
        h0rton_nll = double_gaussian_nll(torch.from_numpy(pred), torch.from_numpy(target))
        # Get scipy evaluation
        matched_nll = 0.0
        for b in range(batch_size):
            target_b = target[b, :]
            # First gaussian
            mu_b = mu[b, :]
            diag_b = np.diagflat(np.exp(logvar[b, :]))
            F_b = F[b, :].reshape(Y_dim, rank)
            low_rank_b =  np.matmul(F_b, F_b.T)
            # Second gaussian
            mu2_b = mu2[b, :]
            diag2_b = np.diagflat(np.exp(logvar2[b, :]))
            F2_b = F2[b, :].reshape(Y_dim, rank)
            low_rank2_b =  np.matmul(F2_b, F2_b.T)
            # Relative weight
            w2_b = 0.5*sigmoid(alpha[b])

            nll1 = -np.log(multivariate_normal.pdf(target_b, mean=mu_b, cov=diag_b + low_rank_b)) # real likelihood, not scaled and shifted
            nll2 = -np.log(multivariate_normal.pdf(target_b, mean=mu2_b, cov=diag2_b + low_rank2_b)) # real likelihood, not scaled and shifted
            # Kernel version
            #matched_nll1 = (2.0*nll1 - Y_dim*np.log(2.0*np.pi))
            #matched_nll2 = (2.0*nll2 - Y_dim*np.log(2.0*np.pi))
            #matched_nll += (-np.log((1.0 - w2_b) * np.exp(-matched_nll1) + w2_b * np.exp(-matched_nll2)))/batch_size 
            matched_nll += (-np.log((1.0 - w2_b) * np.exp(-nll1) + w2_b * np.exp(-nll2)))/batch_size # logsumexp
        np.testing.assert_array_almost_equal(h0rton_nll, matched_nll, decimal=6)

if __name__ == '__main__':
    unittest.main()

        

