import numpy as np
import unittest
from scipy.stats import multivariate_normal
from h0rton.h0_inference.gaussian_bnn_posterior_cpu import sigmoid
from h0rton.losses import DiagonalGaussianNLLCPU, FullRankGaussianNLLCPU, DoubleGaussianNLLCPU

class TestGaussianNLLCPU(unittest.TestCase):
    """A suite of tests verifying the PDF evaluation of GaussianNLLCPU
    
    """

    @classmethod
    def setUpClass(cls):
        """Set global defaults for tests

        """
        pass

    def test_diagonal_gaussian_nll_cpu(self):
        """Test the PDF evaluation of a single Gaussian with diagonal covariance matrix

        """
        # Instantiate NLL class
        Y_dim = 2
        diagonal_gaussian_nll_cpu = DiagonalGaussianNLLCPU(Y_dim)
        # Get h0rton evaluation
        batch_size = 5
        target = np.random.randn(batch_size, Y_dim).astype(np.float32)
        mu = np.random.randn(batch_size, Y_dim)
        logvar = np.abs(2.0*np.random.randn(batch_size, Y_dim))
        pred = np.concatenate([mu, logvar], axis=1).astype(np.float32)
        h0rton_nll = diagonal_gaussian_nll_cpu(pred, target)
        # Get scipy evaluation
        matched_nll = 0.0
        for b in range(batch_size):
            target_b = target[b, :]
            mu_b = mu[b, :]
            cov_b = np.diagflat(np.exp(logvar[b, :]))
            nll = -np.log(multivariate_normal.pdf(target_b, mean=mu_b, cov=cov_b)) # real nll, not scaled and shifted
            matched_nll += nll/batch_size
            #matched_nll += (2.0*nll - Y_dim*np.log(2.0*np.pi))/batch_size # kernel version
        np.testing.assert_array_almost_equal(h0rton_nll, matched_nll, decimal=5)

    def test_full_rank_gaussian_nll_cpu(self):
        """Test the PDF evaluation of a single Gaussian with a full covariance matrix

        """
        Y_dim = 4
        tril_len = Y_dim*(Y_dim + 1)//2
        tril_idx = np.tril_indices(Y_dim)
        batch_size = 3
        gaussian_nll_cpu = FullRankGaussianNLLCPU(Y_dim)
        # Get h0rton evaluation
        target = np.random.randn(batch_size, Y_dim).astype(np.float32)
        mu = np.random.randn(batch_size, Y_dim)
        tril_elements = np.random.randn(batch_size, tril_len)
        pred = np.concatenate([mu, tril_elements], axis=1).astype(np.float32)
        h0rton_nll = gaussian_nll_cpu(pred, target)
        # Get scipy evaluation
        matched_nll = 0.0
        for b in range(batch_size):
            target_b = target[b, :]
            mu_b = mu[b, :]
            tril = np.zeros((Y_dim, Y_dim))
            tril[tril_idx[0], tril_idx[1]] = tril_elements[b, :]
            log_diag_tril = np.diagonal(tril, offset=0, axis1=0, axis2=1).copy()
            tril[np.eye(Y_dim).astype(bool)] = np.exp(log_diag_tril)
            prec_mat = np.dot(tril, tril.T)
            cov_mat = np.linalg.inv(prec_mat)
            nll = -multivariate_normal.logpdf(target_b, mean=mu_b, cov=cov_mat) # real nll, not scaled and shifted
            matched_nll += nll/batch_size
            #matched_nll += (2.0*nll - Y_dim*np.log(2.0*np.pi))/batch_size # kernel version
        np.testing.assert_array_almost_equal(h0rton_nll, matched_nll, decimal=5)

    def test_double_gaussian_nll_cpu(self):
        """Test the PDF evaluation of a mixture of two Gaussians, each with a full covariance matrix

        """
        Y_dim = 4
        tril_idx = np.tril_indices(Y_dim)
        tril_len = len(tril_idx[0])
        batch_size = 3
        loss = DoubleGaussianNLLCPU(Y_dim)
        # Get h0rton evaluation
        target = np.random.randn(batch_size, Y_dim).astype(np.float32)
        mu = np.random.randn(batch_size, Y_dim)
        tril_elements = np.random.randn(batch_size, tril_len)
        mu2 = np.random.randn(batch_size, Y_dim)
        tril_elements2 = np.random.randn(batch_size, tril_len)
        alpha = np.random.randn(batch_size, 1)
        pred = np.concatenate([mu, tril_elements, mu2, tril_elements2, alpha], axis=1).astype(np.float32)
        h0rton_nll = loss(pred, target)
        # Get scipy evaluation
        matched_nll = 0.0
        for b in range(batch_size):
            target_b = target[b, :]
            # First gaussian
            mu_b = mu[b, :]
            tril = np.zeros((Y_dim, Y_dim))
            tril[tril_idx[0], tril_idx[1]] = tril_elements[b, :]
            log_diag_tril = np.diagonal(tril)
            tril[np.eye(Y_dim).astype(bool)] = np.exp(log_diag_tril)
            prec_mat = np.dot(tril, tril.T)
            cov_mat = np.linalg.inv(prec_mat)
            nll1 = -np.log(multivariate_normal.pdf(target_b, mean=mu_b, cov=cov_mat))
            # Second gaussian
            mu2_b = mu2[b, :]
            tril2 = np.zeros((Y_dim, Y_dim))
            tril2[tril_idx[0], tril_idx[1]] = tril_elements2[b, :]
            log_diag_tril2 = np.diagonal(tril2)
            tril2[np.eye(Y_dim).astype(bool)] = np.exp(log_diag_tril2)
            prec_mat2 = np.dot(tril2, tril2.T)
            cov_mat2 = np.linalg.inv(prec_mat2)
            nll2 = -np.log(multivariate_normal.pdf(target_b, mean=mu2_b, cov=cov_mat2))
            # Relative weight
            w2_b = sigmoid(alpha[b])
            #matched_nll1 = (2.0*nll1 - Y_dim*np.log(2.0*np.pi))
            #matched_nll2 = (2.0*nll2 - Y_dim*np.log(2.0*np.pi))
            #matched_nll += (-np.log((1.0 - w2_b) * np.exp(-matched_nll1) + w2_b * np.exp(-matched_nll2)))/batch_size 
            matched_nll += (-np.log((1.0 - 0.5*w2_b) * np.exp(-nll1) + 0.5*w2_b * np.exp(-nll2)))/batch_size # logsumexp
        np.testing.assert_array_almost_equal(h0rton_nll, matched_nll, decimal=5)

if __name__ == '__main__':
    unittest.main()

        

