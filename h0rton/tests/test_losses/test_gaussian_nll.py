import os, sys
import numpy as np
import unittest
import torch

class TestGaussianNLL(unittest.TestCase):
    """A suite of tests verifying the PDF evaluation of GaussianNLL
    
    """
    def test_diagonal_gaussian_ll_pdf(self):
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
            matched_nll += (2.0*nll - Y_dim*np.log(2.0*np.pi))/batch_size
        np.testing.assert_array_almost_equal(h0rton_nll, matched_nll, decimal=6)

    def test_lowrank_gaussian_ll_pdf(self):
        """Test the PDF evaluation of a single Gaussian with a full but low-rank plus diagonal covariance matrix

        """
        from h0rton.losses import LowRankGaussianNLL
        from scipy.stats import multivariate_normal
        # Instantiate NLL class
        Y_dim = 2
        rank = 2
        device = torch.device('cpu')
        diagonal_gaussian_nll = LowRankGaussianNLL(Y_dim, device)
        # Get h0rton evaluation
        batch_size = 3
        target = np.random.randn(batch_size, Y_dim)
        mu = np.random.randn(batch_size, Y_dim)
        logvar = np.abs(2.0*np.random.randn(batch_size, Y_dim))
        F = np.random.randn(batch_size, rank*Y_dim)
        pred = np.concatenate([mu, logvar, F], axis=1)
        h0rton_nll = diagonal_gaussian_nll(torch.from_numpy(pred), torch.from_numpy(target))
        # Get scipy evaluation
        matched_nll = 0.0
        for b in range(batch_size):
            target_b = target[b, :]
            mu_b = mu[b, :]
            cov_b = np.diagflat(np.exp(logvar[b, :]))
            nll = -np.log(multivariate_normal.pdf(target_b, mean=mu_b, cov=cov_b)) # real nll, not scaled and shifted
            matched_nll += (2.0*nll - Y_dim*np.log(2.0*np.pi))/batch_size
        np.testing.assert_array_almost_equal(h0rton_nll, matched_nll, decimal=6)

        

