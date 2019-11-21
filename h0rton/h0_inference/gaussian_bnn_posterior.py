from abc import ABC, abstractmethod
import random
import numpy as np
import torch
__all__ = ['BaseGaussianBNNPosterior', 'DiagonalGaussianBNNPosterior', 'LowRankGaussianBNNPosterior', 'DoubleGaussianBNNPosterior']

class BaseGaussianBNNPosterior(ABC):
    """Abstract base class to represent the Gaussian BNN posterior

    Gaussian posteriors or mixtures thereof with various forms of the covariance matrix inherit from this class.

    """
    def __init__(self, pred, Y_dim, whitened_Y_cols_idx, Y_mean, Y_std, log_parameterized_Y_cols_idx, device):
        """
        Parameters
        ----------
        pred : torch.Tensor of shape `[1, out_dim]` or `[out_dim,]`
            raw network output for the predictions
        Y_dim : int
            number of parameters to predict
        whitened_Y_cols_idx : list
            list of Y_cols indices that were whitened
        Y_mean : list
            mean values for the original values of `whitened_Y_cols`
        Y_std : list
            std values for the original values of `whitened_Y_cols`
        log_parameterized_Y_cols_idx : list
            list of Y_cols indices that were log-parameterized
        device : torch.device object

        """
        self.pred = pred
        self.batch_size = pred.shape[0]
        self.Y_dim = Y_dim
        self.whitened_Y_cols_idx = whitened_Y_cols_idx
        self.Y_mean = torch.Tensor(Y_mean)
        self.Y_std = torch.Tensor(Y_std)
        self.log_parameterized_Y_cols_idx = log_parameterized_Y_cols_idx
        self.device = device
        self.sigmoid = torch.nn.Sigmoid()
        self.logsigmoid = torch.nn.LogSigmoid()

    def _check_input_shape(self, out_dim):
        """Check if the input tensors have the shapes consistent with the number of parameters to predict

        Parameters
        ----------
        out_dim : int
            dimension of the output expected from `self.Y_dim`

        """
        if self.pred.shape[1] != out_dim:
            raise ValueError("Value supplied for `Y_dim` and the shape of `pred` are not consistent.")

    def seed_samples(self, sample_seed):
        """Seed the sampling for reproducibility

        Parameters
        ----------
        sample_seed : int

        """
        np.random.seed(sample_seed)
        random.seed(sample_seed)
        torch.manual_seed(sample_seed)
        torch.cuda.manual_seed(sample_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

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
        F = F.repeat(n_samples, 1, 1) # [self.batch_size*n_samples, self.Y_dim, self.rank]
        mu = mu.repeat(n_samples, 1)
        logvar = logvar.repeat(n_samples, 1)
        eps_low_rank = torch.randn(self.batch_size*n_samples, self.rank, 1)
        eps_diag = torch.randn(self.batch_size*n_samples, self.Y_dim)
        half_var = torch.exp(0.5*logvar) # [self.batch_size*n_samples, self.Y_dim]
        samples = torch.bmm(F, eps_low_rank).squeeze() + mu + half_var*eps_diag
        samples = samples.reshape(n_samples, self.batch_size, self.Y_dim)
        samples = samples.data.cpu().numpy()
        samples = samples.swapaxes(0, 1)
        return samples

    def unwhiten_back(self, pred):
        """Scale and shift back to the unwhitened state

        Parameters
        ----------
        pred : torch.Tensor
            network prediction

        Returns
        -------
        torch.Tensor
            the unwhitened pred
        
        """
        whitened_subpred = pred[:, self.whitened_Y_cols_idx]
        unwhitened_subpred = whitened_subpred*self.Y_std + self.Y_mean
        pred[:, self.whitened_Y_cols_idx] = unwhitened_subpred
        return pred

    def exponentiate_back(self, pred):
        """Exponentiate back the log-parameterized Y values

        Parameters
        ----------
        pred : torch.Tensor
            network prediction

        Returns
        -------
        torch.Tensor
            the unwhitened pred
        
        """
        log_subpred = pred[:, self.log_parameterized_Y_cols_idx]
        linear_subpred = torch.exp(log_subpred)
        pred[:, self.log_parameterized_Y_cols_idx] = linear_subpred
        return pred

class DiagonalGaussianBNNPosterior(BaseGaussianBNNPosterior):
    """The negative log likelihood (NLL) for a single Gaussian with diagonal covariance matrix
        
    `BaseGaussianNLL.__init__` docstring for the parameter description.

    """
    def __init__(self, pred, Y_dim, whitened_Y_cols_idx, Y_mean, Y_std, log_parameterized_Y_cols_idx, device):
        super(DiagonalGaussianBNNPosterior, self).__init__(pred, Y_dim, whitened_Y_cols_idx, Y_mean, Y_std, log_parameterized_Y_cols_idx, device)
        out_dim = self.Y_dim*2
        self._check_input_shape(out_dim)
        d = self.Y_dim # for readability
        mu = pred[:, :d].reshape(self.batch_size, -1)
        #mu = self.unwhiten_back(mu)
        #mu = self.exponentiate_back(mu)
        self.mu = mu
        self.logvar = pred[:, d:].reshape(self.batch_size, -1)

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
        eps = torch.randn(self.batch_size, n_samples, self.Y_dim)
        samples = eps*torch.exp(0.5*self.logvar.unsqueeze(1)) + self.mu.unsqueeze(1)
        samples = samples.data.cpu().numpy()
        return samples

    def get_hpd_interval(self):
        return NotImplementedError

class LowRankGaussianBNNPosterior(BaseGaussianBNNPosterior):
    """The negative log likelihood (NLL) for a single Gaussian with diagonal covariance matrix
        
    `BaseGaussianNLL.__init__` docstring for the parameter description.

    """
    def __init__(self, pred, Y_dim, whitened_Y_cols_idx, Y_mean, Y_std, log_parameterized_Y_cols_idx, device):
        super(LowRankGaussianBNNPosterior, self).__init__(pred, Y_dim, whitened_Y_cols_idx, Y_mean, Y_std, log_parameterized_Y_cols_idx, device)
        out_dim = self.Y_dim*4
        self._check_input_shape(out_dim)
        self.rank = 2 # FIXME: hardcoded
        d = self.Y_dim # for readability
        mu = pred[:, :d].reshape(self.batch_size, -1)
        #mu = self.unwhiten_back(mu)
        #mu = self.exponentiate_back(mu)
        self.mu = mu
        self.logvar = pred[:, d:2*d].reshape(self.batch_size, -1)
        self.F = pred[:, 2*d:].reshape(self.batch_size, self.Y_dim, self.rank)

    def sample(self, n_samples, sample_seed):
        self.seed_samples(sample_seed)
        return self.sample_low_rank(n_samples, self.mu, self.logvar, self.F)

    def get_hpd_interval(self):
        return NotImplementedError

class DoubleGaussianBNNPosterior(BaseGaussianBNNPosterior):
    """The negative log likelihood (NLL) for a single Gaussian with diagonal covariance matrix
        
    `BaseGaussianNLL.__init__` docstring for the parameter description.

    """
    def __init__(self, pred, Y_dim, whitened_Y_cols_idx, Y_mean, Y_std, log_parameterized_Y_cols_idx, device):
        super(DoubleGaussianBNNPosterior, self).__init__(pred, Y_dim, whitened_Y_cols_idx, Y_mean, Y_std, log_parameterized_Y_cols_idx, device)
        out_dim = self.Y_dim*8 + 1
        self._check_input_shape(out_dim)
        self.rank = 2 # FIXME: hardcoded
        d = self.Y_dim # for readability
        mu = pred[:, :d].reshape(self.batch_size, -1)
        #mu = self.unwhiten_back(mu)
        #mu = self.exponentiate_back(mu)
        self.mu = mu
        self.logvar = pred[:, d:2*d].reshape(self.batch_size, -1)
        self.F = pred[:, 2*d:4*d].reshape(self.batch_size, self.Y_dim, self.rank)
        mu2 = pred[:, 4*d:5*d].reshape(self.batch_size, -1)
        #mu2 = self.unwhiten_back(mu2)
        #mu2 = self.exponentiate_back(mu2)
        self.mu2 = mu2
        self.logvar2 = pred[:, 5*d:6*d].reshape(self.batch_size, -1)
        self.F2 = pred[:, 6*d:8*d].reshape(self.batch_size, self.Y_dim, self.rank)
        self.w2 = 0.5*self.sigmoid(pred[:, -1].reshape(self.batch_size, -1))
        
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
        samples = torch.empty([self.batch_size, n_samples, self.Y_dim], device=self.device)
        # Determine first vs. second Gaussian
        unif2 = torch.rand(self.batch_size, n_samples)
        second_gaussian = (self.w2 > unif2)
        # Sample from second Gaussian
        samples2 = torch.Tensor(self.sample_low_rank(n_samples, self.mu2, self.logvar2, self.F2))
        samples[second_gaussian, :] = samples2[second_gaussian, :]
        # Sample from first Gaussian
        samples1 = torch.Tensor(self.sample_low_rank(n_samples, self.mu, self.logvar, self.F))
        samples[~second_gaussian, :] = samples1[~second_gaussian, :]
        samples = samples.data.cpu().numpy()
        return samples

    def get_hpd_interval(self):
        return NotImplementedError




