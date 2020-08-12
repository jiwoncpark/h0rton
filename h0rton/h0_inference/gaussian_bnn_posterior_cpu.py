from abc import ABC, abstractmethod
import random
import numpy as np
from scipy.stats import multivariate_normal
__all__ = ['BaseGaussianBNNPosteriorCPU', 'DiagonalGaussianBNNPosteriorCPU', 'LowRankGaussianBNNPosteriorCPU', 'DoubleLowRankGaussianBNNPosteriorCPU', 'FullRankGaussianBNNPosteriorCPU', 'DoubleGaussianBNNPosteriorCPU', 'sigmoid', 'logsigmoid']

def sigmoid(x):
    return np.where(x >= 0, 
                    1 / (1 + np.exp(-x)), 
                    np.exp(x) / (1 + np.exp(x)))

def logsigmoid(x):
    return np.where(x >= 0, 
                    -np.log1p(np.exp(-x)), 
                    x - np.log1p(np.exp(x)))

class BaseGaussianBNNPosteriorCPU(ABC):
    """Abstract base class to represent the Gaussian BNN posterior

    Gaussian posteriors or mixtures thereof with various forms of the covariance matrix inherit from this class.

    """
    def __init__(self, Y_dim, Y_mean=None, Y_std=None):
        """
        Parameters
        ----------
        pred : torch.Tensor of shape `[1, out_dim]` or `[out_dim,]`
            raw network output for the predictions
        Y_dim : int
            number of parameters to predict
        Y_mean : list
            mean values for the original values of `whitened_Y_cols`
        Y_std : list
            std values for the original values of `whitened_Y_cols`
        device : torch.device object

        """
        self.Y_dim = Y_dim
        self.Y_mean = Y_mean.reshape(1, -1)
        self.Y_std = Y_std.reshape(1, -1)
        self.sigmoid = sigmoid
        self.logsigmoid = logsigmoid

    def seed_samples(self, sample_seed):
        """Seed the sampling for reproducibility

        Parameters
        ----------
        sample_seed : int

        """
        np.random.seed(sample_seed)
        random.seed(sample_seed)

    @abstractmethod
    def sample(self, n_samples, sample_seed=None):
        """Sample from the Gaussian posterior. Must be overridden by subclasses.

        Parameters
        ----------
        n_samples : int
            how many samples to obtain
        sample_seed : int
            seed for the samples. Default: None

        Returns
        -------
        np.array of shape `[n_samples, self.Y_dim]`
            samples

        """
        return NotImplemented

    @abstractmethod
    def get_hpd_interval(self):
        """Get the highest posterior density (HPD) interval

        """
        return NotImplemented

    def transform_back_mu(self, array):
        """Transform back, i.e. unwhiten, the tensor of central values

        Parameters
        ----------
        array : np.array of shape `[batch_size, Y_dim]`

        Returns
        -------
        torch.tensor of shape `[batch_size, Y_dim]`
            the original tensor

        """
        array = np.expand_dims(array, axis=1)
        array = self.unwhiten_back(array)
        return array.squeeze()

    def unwhiten_back(self, sample):
        """Scale and shift back to the unwhitened state

        Parameters
        ----------
        pred : torch.Tensor
            network prediction of shape `[batch_size, n_samples, self.Y_dim]`

        Returns
        -------
        torch.Tensor
            the unwhitened pred
        
        """
        sample = sample*np.expand_dims(self.Y_std, 1) + np.expand_dims(self.Y_mean, 1)
        return sample

    def sample_low_rank(self, n_samples, mu, logvar, F):
        """Sample from a single Gaussian posterior with a full but low-rank plus diagonal covariance matrix

        Parameters
        ----------
        n_samples : int
            how many samples to obtain
        mu : torch.Tensor of shape `[self.batch_size, self.Y_dim]`
            network prediction of the mu (mean parameter) of the BNN posterior
        logvar : torch.Tensor of shape `[self.batch_size, self.Y_dim]`
            network prediction of the log of the diagonal elements of the covariance matrix
        F : torch.Tensor of shape `[self.batch_size, self.Y_dim, self.rank]`
            network prediction of the low rank portion of the covariance matrix

        Returns
        -------
        np.array of shape `[self.batch_size, n_samples, self.Y_dim]`
            samples

        """
        #F = torch.unsqueeze(F, dim=1).repeat(1, n_samples, 1, 1) # [self.batch_size, n_samples, self.Y_dim, self.rank]
        F = np.repeat(F, repeats=n_samples, axis=0) # [self.batch_size*n_samples, self.Y_dim, self.rank]
        mu = np.repeat(mu, repeats=n_samples, axis=0) # [self.batch_size*n_samples, self.Y_dim]
        logvar = np.repeat(logvar, repeats=n_samples, axis=0) # [self.batch_size*n_samples, self.Y_dim]
        eps_low_rank = np.random.randn(self.batch_size*n_samples, self.rank, 1)
        eps_diag = np.random.randn(self.batch_size*n_samples, self.Y_dim)
        half_var = np.exp(0.5*logvar) # [self.batch_size*n_samples, self.Y_dim]
        samples = np.matmul(F, eps_low_rank).squeeze() + mu + half_var*eps_diag # [self.batch_size*n_samples, self.Y_dim]
        samples = samples.reshape(self.batch_size, n_samples, self.Y_dim)
        #samples = samples.transpose((1, 0, 2)) # [self.batch_size, n_samples, self.Y_dim]
        return samples
    
    def sample_full_rank(self, n_samples, mu, tril_elements):
        """Sample from a single Gaussian posterior with a full-rank covariance matrix

        Parameters
        ----------
        n_samples : int
            how many samples to obtain
        mu : torch.Tensor of shape `[self.batch_size, self.Y_dim]`
            network prediction of the mu (mean parameter) of the BNN posterior
        tril_elements : torch.Tensor of shape `[self.batch_size, tril_len]`
            network prediction of lower-triangular matrix in the log-Cholesky decomposition of the precision matrix

        Returns
        -------
        np.array of shape `[self.batch_size, n_samples, self.Y_dim]`
            samples

        """
        samples = np.zeros([self.batch_size, n_samples, self.Y_dim])
        for b in range(self.batch_size):
            tril = np.zeros([self.Y_dim, self.Y_dim])
            tril[self.tril_idx[0], self.tril_idx[1]] = tril_elements[b, :]
            log_diag_tril = np.diagonal(tril, offset=0, axis1=0, axis2=1)
            tril[np.eye(self.Y_dim).astype(bool)] = np.exp(log_diag_tril)
            prec_mat = np.dot(tril, tril.T) # [Y_dim, Y_dim]
            cov_mat = np.linalg.inv(prec_mat)
            sample_b = multivariate_normal.rvs(mean=mu[b, :], cov=cov_mat, size=[n_samples,])
            samples[b, :, :] = sample_b
        samples = self.unwhiten_back(samples)
        return samples

class DiagonalGaussianBNNPosteriorCPU(BaseGaussianBNNPosteriorCPU):
    """The negative log likelihood (NLL) for a single Gaussian with diagonal covariance matrix
        
    `BaseGaussianNLL.__init__` docstring for the parameter description.

    """
    def __init__(self, Y_dim, Y_mean=None, Y_std=None):
        super(DiagonalGaussianBNNPosteriorCPU, self).__init__(Y_dim, Y_mean, Y_std)
        self.out_dim = self.Y_dim*2

    def set_sliced_pred(self, pred):
        d = self.Y_dim # for readability
        #pred = pred.cpu().numpy()
        self.batch_size = pred.shape[0]
        self.mu = pred[:, :d]
        self.logvar = pred[:, d:]
        self.cov_diag = np.exp(self.logvar)
        #F_tran_F = np.matmul(normal.F, np.swapaxes(normal.F, 1, 2))
        #cov_mat = np.apply_along_axis(np.diag, -1, np.exp(normal.logvar)) + F_tran_F
        #cov_diag = np.exp(normal.logvar) + np.diagonal(F_tran_F, axis1=1, axis2=2)
        #assert np.array_equal(cov_mat.shape, [batch_size, self.Y_dim, self.Y_dim])
        #assert np.array_equal(cov_diag.shape, [batch_size, self.Y_dim])
        #np.apply_along_axis(np.diag, -1, np.exp(logvar)) # for diagonal

    def sample(self, n_samples, sample_seed):
        """Sample from a Gaussian posterior with diagonal covariance matrix

        Parameters
        ----------
        n_samples : int
            how many samples to obtain
        sample_seed : int
            seed for the samples. Default: None

        Returns
        -------
        np.array of shape `[n_samples, self.Y_dim]`
            samples

        """
        self.seed_samples(sample_seed)
        eps = np.random.randn(self.batch_size, n_samples, self.Y_dim)
        samples = eps*np.exp(0.5*np.expand_dims(self.logvar, 1)) + np.expand_dims(self.mu, 1)
        samples = self.unwhiten_back(samples)
        return samples

    def get_hpd_interval(self):
        return NotImplementedError

class LowRankGaussianBNNPosteriorCPU(BaseGaussianBNNPosteriorCPU):
    """The negative log likelihood (NLL) for a single Gaussian with diagonal covariance matrix
        
    `BaseGaussianNLL.__init__` docstring for the parameter description.

    """
    def __init__(self, Y_dim, Y_mean=None, Y_std=None):
        super(LowRankGaussianBNNPosteriorCPU, self).__init__(Y_dim, Y_mean, Y_std)
        self.out_dim = self.Y_dim*4
        self.rank = 2 # FIXME: hardcoded

    def set_sliced_pred(self, pred):
        d = self.Y_dim # for readability
        #pred = pred.cpu().numpy()
        self.batch_size = pred.shape[0]
        self.mu = pred[:, :d]
        self.logvar = pred[:, d:2*d]
        self.F = pred[:, 2*d:].reshape([self.batch_size, self.Y_dim, self.rank])
        #F_tran_F = np.matmul(self.F, np.swapaxes(self.F, 1, 2))
        #self.cov_diag = np.exp(self.logvar) + np.diagonal(F_tran_F, axis1=1, axis2=2)

    def sample(self, n_samples, sample_seed):
        self.seed_samples(sample_seed)
        return self.sample_low_rank(n_samples, self.mu, self.logvar, self.F)

    def get_hpd_interval(self):
        return NotImplementedError

class DoubleLowRankGaussianBNNPosteriorCPU(BaseGaussianBNNPosteriorCPU):
    """The negative log likelihood (NLL) for a single Gaussian with diagonal covariance matrix
        
    `BaseGaussianNLL.__init__` docstring for the parameter description.

    """
    def __init__(self, Y_dim, Y_mean=None, Y_std=None):
        super(DoubleLowRankGaussianBNNPosteriorCPU, self).__init__(Y_dim, Y_mean, Y_std)
        self.out_dim = self.Y_dim*8 + 1
        self.rank = 2 # FIXME: hardcoded

    def set_sliced_pred(self, pred):
        d = self.Y_dim # for readability
        #pred = pred.cpu().numpy()
        self.w2 = 0.5*self.sigmoid(pred[:, -1].reshape(-1, 1))
        self.batch_size = pred.shape[0]
        self.mu = pred[:, :d]
        self.logvar = pred[:, d:2*d]
        self.F = pred[:, 2*d:4*d].reshape([self.batch_size, self.Y_dim, self.rank])
        #F_tran_F = np.matmul(self.F, np.swapaxes(self.F, 1, 2))
        #self.cov_diag = np.exp(self.logvar) + np.diagonal(F_tran_F, axis1=1, axis2=2)
        self.mu2 = pred[:, 4*d:5*d]
        self.logvar2 = pred[:, 5*d:6*d]
        self.F2 = pred[:, 6*d:8*d].reshape([self.batch_size, self.Y_dim, self.rank])
        #F_tran_F2 = np.matmul(self.F2, np.swapaxes(self.F2, 1, 2))
        #self.cov_diag2 = np.exp(self.logvar2) + np.diagonal(F_tran_F2, axis1=1, axis2=2)
        
        
    def sample(self, n_samples, sample_seed):
        """Sample from a mixture of two Gaussians, each with a full but constrained as low-rank plus diagonal covariance

        Parameters
        ----------
        n_samples : int
            how many samples to obtain
        sample_seed : int
            seed for the samples. Default: None

        Returns
        -------
        np.array of shape `[self.batch_size, n_samples, self.Y_dim]`
            samples

        """
        self.seed_samples(sample_seed)
        samples = np.zeros([self.batch_size, n_samples, self.Y_dim])
        # Determine first vs. second Gaussian
        unif2 = np.random.rand(self.batch_size, n_samples)
        second_gaussian = (self.w2 > unif2)
        # Sample from second Gaussian
        samples2 = self.sample_low_rank(n_samples, self.mu2, self.logvar2, self.F2)
        samples[second_gaussian, :] = samples2[second_gaussian, :]
        # Sample from first Gaussian
        samples1 = self.sample_low_rank(n_samples, self.mu, self.logvar, self.F)
        samples[~second_gaussian, :] = samples1[~second_gaussian, :]
        return samples

    def get_hpd_interval(self):
        return NotImplementedError

class FullRankGaussianBNNPosteriorCPU(BaseGaussianBNNPosteriorCPU):
    """The negative log likelihood (NLL) for a single Gaussian with diagonal covariance matrix
        
    `BaseGaussianNLL.__init__` docstring for the parameter description.

    """
    def __init__(self, Y_dim, Y_mean=None, Y_std=None):
        super(FullRankGaussianBNNPosteriorCPU, self).__init__(Y_dim, Y_mean, Y_std)
        self.tril_idx = np.tril_indices(self.Y_dim) # lower-triangular indices
        self.tril_len = len(self.tril_idx[0])
        self.out_dim = self.Y_dim + self.Y_dim*(self.Y_dim + 1)//2

    def set_sliced_pred(self, pred):
        d = self.Y_dim # for readability
        #pred = pred.cpu().numpy()
        self.batch_size = pred.shape[0]
        self.mu = pred[:, :d]
        self.tril_elements = pred[:, d:self.out_dim]

    def sample(self, n_samples, sample_seed):
        self.seed_samples(sample_seed)
        return self.sample_full_rank(n_samples, self.mu, self.tril_elements)

    def get_hpd_interval(self):
        return NotImplementedError

class DoubleGaussianBNNPosteriorCPU(BaseGaussianBNNPosteriorCPU):
    """The negative log likelihood (NLL) for a single Gaussian with diagonal covariance matrix
        
    `BaseGaussianNLL.__init__` docstring for the parameter description.

    """
    def __init__(self, Y_dim, Y_mean=None, Y_std=None):
        super(DoubleGaussianBNNPosteriorCPU, self).__init__(Y_dim, Y_mean, Y_std)
        self.tril_idx = np.tril_indices(self.Y_dim) # lower-triangular indices
        self.tril_len = len(self.tril_idx[0])
        self.out_dim = self.Y_dim**2 + 3*self.Y_dim + 1

    def set_sliced_pred(self, pred):
        d = self.Y_dim # for readability
        #pred = pred.cpu().numpy()
        self.batch_size = pred.shape[0]
        # First gaussian
        self.mu = pred[:, :d]
        self.tril_elements = pred[:, d:d+self.tril_len]
        self.mu2 = pred[:, d+self.tril_len:2*d+self.tril_len]
        self.tril_elements2 = pred[:, 2*d+self.tril_len:-1]
        self.w2 = 0.5*self.sigmoid(pred[:, -1].reshape(-1, 1))
        
    def sample(self, n_samples, sample_seed):
        """Sample from a mixture of two Gaussians, each with a full but constrained as low-rank plus diagonal covariance

        Parameters
        ----------
        n_samples : int
            how many samples to obtain
        sample_seed : int
            seed for the samples. Default: None

        Returns
        -------
        np.array of shape `[self.batch_size, n_samples, self.Y_dim]`
            samples

        """
        self.seed_samples(sample_seed)
        samples = np.zeros([self.batch_size, n_samples, self.Y_dim])
        # Determine first vs. second Gaussian
        unif2 = np.random.rand(self.batch_size, n_samples)
        second_gaussian = (self.w2 > unif2)
        # Sample from second Gaussian
        samples2 = self.sample_full_rank(n_samples, self.mu2, self.tril_elements2)
        samples[second_gaussian, :] = samples2[second_gaussian, :]
        # Sample from first Gaussian
        samples1 = self.sample_full_rank(n_samples, self.mu, self.tril_elements)
        samples[~second_gaussian, :] = samples1[~second_gaussian, :]
        return samples

    def get_hpd_interval(self):
        return NotImplementedError


