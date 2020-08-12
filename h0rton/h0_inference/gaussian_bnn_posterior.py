from abc import ABC, abstractmethod
import random
import numpy as np
import torch
__all__ = ['BaseGaussianBNNPosterior', 'DiagonalGaussianBNNPosterior', 'LowRankGaussianBNNPosterior', 'DoubleLowRankGaussianBNNPosterior', 'FullRankGaussianBNNPosterior', 'DoubleGaussianBNNPosterior']

class BaseGaussianBNNPosterior(ABC):
    """Abstract base class to represent the Gaussian BNN posterior

    Gaussian posteriors or mixtures thereof with various forms of the covariance matrix inherit from this class.

    """
    def __init__(self, Y_dim, device, Y_mean=None, Y_std=None):
        """
        Parameters
        ----------
        Y_dim : int
            number of parameters to predict
        whitened_Y_cols_idx : list
            list of Y_cols indices that were whitened
        Y_mean : list
            mean values for the original values of `whitened_Y_cols`
        Y_std : list
            std values for the original values of `whitened_Y_cols`
        device : torch.device object

        """
        self.Y_dim = Y_dim
        self.Y_mean = torch.Tensor(Y_mean).reshape(1, -1)
        self.Y_std = torch.Tensor(Y_std).reshape(1, -1)
        self.device = device
        self.sigmoid = torch.nn.Sigmoid()
        self.logsigmoid = torch.nn.LogSigmoid()

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

    def transform_back_mu(self, tensor):
        """Transform back, i.e. unwhiten, the tensor of central values

        Parameters
        ----------
        tensor : torch.Tensor of shape `[batch_size, Y_dim]`

        Returns
        -------
        torch.tensor of shape `[batch_size, Y_dim]`
            the original tensor

        """
        tensor = tensor.unsqueeze(1)
        tensor = self.unwhiten_back(tensor)
        return tensor.squeeze()

    def transform_back_logvar(self, logvar):
        """Transform back, i.e. unwhiten, the tensor of predicted log of the diagonal entries of the cov mat

        Parameters
        ----------
        tensor : torch.Tensor of shape `[batch_size, Y_dim]`

        Returns
        -------
        torch.tensor of shape `[batch_size, Y_dim]`
            the original tensor

        """
        natural_logvar = logvar*self.Y_std*self.Y_std # note self.Y_std is shape [1, Y_dim]
        return natural_logvar

    def transform_back_cov_mat(self, cov_mat):
        """Transform back, i.e. unwhiten, the tensor of predicted covariance matrix

        Parameters
        ----------
        tensor : torch.Tensor of shape `[batch_size, Y_dim, Y_dim]`

        Returns
        -------
        torch.tensor of shape `[batch_size, Y_dim]`
            the original tensor

        """
        natural_cov_mat = cov_mat*self.Y_std.unsqueeze(-1)*self.Y_std.unsqueeze(0)
        return natural_cov_mat

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
        sample = sample*self.Y_std.unsqueeze(1) + self.Y_mean.unsqueeze(1)
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
        F = F.repeat(n_samples, 1, 1) # [self.batch_size*n_samples, self.Y_dim, self.rank]
        mu = mu.repeat(n_samples, 1) # [self.batch_size*n_samples, self.Y_dim]
        logvar = logvar.repeat(n_samples, 1) # [self.batch_size*n_samples, self.Y_dim]
        eps_low_rank = torch.randn(self.batch_size*n_samples, self.rank, 1)
        eps_diag = torch.randn(self.batch_size*n_samples, self.Y_dim)
        half_var = torch.exp(0.5*logvar) # [self.batch_size*n_samples, self.Y_dim]
        samples = torch.bmm(F, eps_low_rank).squeeze() + mu + half_var*eps_diag
        samples = samples.reshape(n_samples, self.batch_size, self.Y_dim)
        samples = samples.transpose(0, 1)
        samples = self.unwhiten_back(samples)
        samples = samples.data.cpu().numpy()
        return samples

    def sample_full_rank(self, n_samples, mu, tril_elements, as_numpy=True):
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
        samples = torch.zeros([self.batch_size, n_samples, self.Y_dim])
        for b in range(self.batch_size):
            tril = torch.zeros([self.Y_dim, self.Y_dim], device=self.device, dtype=None)
            tril[self.tril_idx[0], self.tril_idx[1]] = tril_elements[b, :]
            log_diag_tril = torch.diagonal(tril, offset=0, dim1=0, dim2=1)
            tril[torch.eye(self.Y_dim, dtype=bool)] = torch.exp(log_diag_tril)
            prec_mat = torch.mm(tril, tril.T) # [Y_dim, Y_dim]
            mvn = torch.distributions.multivariate_normal.MultivariateNormal(loc=mu[b, :], precision_matrix=prec_mat)
            sample_b = mvn.sample([n_samples,])
            samples[b, :, :] = sample_b
        samples = self.unwhiten_back(samples)
        if as_numpy:
            return samples.cpu().numpy()
        else:
            return samples

class DiagonalGaussianBNNPosterior(BaseGaussianBNNPosterior):
    """The negative log likelihood (NLL) for a single Gaussian with diagonal covariance matrix
        
    `BaseGaussianNLL.__init__` docstring for the parameter description.

    """
    def __init__(self, Y_dim, device, Y_mean=None, Y_std=None):
        super(DiagonalGaussianBNNPosterior, self).__init__(Y_dim, device, Y_mean, Y_std)
        self.out_dim = self.Y_dim*2

    def set_sliced_pred(self, pred):
        d = self.Y_dim # for readability
        self.batch_size = pred.shape[0]
        self.mu = pred[:, :d]
        self.logvar = pred[:, d:]
        self.cov_diag = torch.exp(self.logvar)

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
        samples = self.unwhiten_back(samples)
        samples = samples.data.cpu().numpy()
        return samples

    def get_hpd_interval(self):
        return NotImplementedError

class LowRankGaussianBNNPosterior(BaseGaussianBNNPosterior):
    """The negative log likelihood (NLL) for a single Gaussian with diagonal covariance matrix
        
    `BaseGaussianNLL.__init__` docstring for the parameter description.

    """
    def __init__(self, Y_dim, device, Y_mean=None, Y_std=None):
        super(LowRankGaussianBNNPosterior, self).__init__(Y_dim, device, Y_mean, Y_std)
        self.out_dim = self.Y_dim*4
        self.rank = 2 # FIXME: hardcoded

    def set_sliced_pred(self, pred):
        d = self.Y_dim # for readability
        self.batch_size = pred.shape[0]
        self.mu = pred[:, :d]
        self.logvar = pred[:, d:2*d]
        self.F = pred[:, 2*d:].reshape([self.batch_size, self.Y_dim, self.rank])
        F_F_tran = torch.bmm(self.F, torch.transpose(self.F, 1, 2)) # [n_lenses, d, d]
        self.cov_diag = torch.exp(self.logvar) + torch.diagonal(F_F_tran, dim1=1, dim2=2) # [n_lenses, d]
        self.cov_mat = torch.diag_embed(self.logvar) + F_F_tran

    def sample(self, n_samples, sample_seed):
        self.seed_samples(sample_seed)
        return self.sample_low_rank(n_samples, self.mu, self.logvar, self.F)

    def get_hpd_interval(self):
        return NotImplementedError

class DoubleLowRankGaussianBNNPosterior(BaseGaussianBNNPosterior):
    """The negative log likelihood (NLL) for a single Gaussian with diagonal covariance matrix
        
    `BaseGaussianNLL.__init__` docstring for the parameter description.

    """
    def __init__(self, Y_dim, device, Y_mean=None, Y_std=None):
        super(DoubleLowRankGaussianBNNPosterior, self).__init__(Y_dim, device, Y_mean, Y_std)
        self.out_dim = self.Y_dim*8 + 1
        self.rank = 2 # FIXME: hardcoded

    def set_sliced_pred(self, pred):
        d = self.Y_dim # for readability
        self.batch_size = pred.shape[0]
        # First gaussian
        self.mu = pred[:, :d]
        self.logvar = pred[:, d:2*d]
        self.F = pred[:, 2*d:4*d].reshape([self.batch_size, self.Y_dim, self.rank])
        F_F_tran = torch.bmm(self.F, torch.transpose(self.F, 1, 2)) # [n_lenses, d, d]
        self.cov_diag = torch.exp(self.logvar) + torch.diagonal(F_F_tran, dim1=1, dim2=2) # [n_lenses, d]
        self.cov_mat = torch.diag_embed(self.logvar) + F_F_tran
        # Second gaussian
        self.mu2 = pred[:, 4*d:5*d]
        self.logvar2 = pred[:, 5*d:6*d]
        self.F2 = pred[:, 6*d:8*d].reshape([self.batch_size, self.Y_dim, self.rank])
        F_F_tran2 = torch.bmm(self.F2, torch.transpose(self.F2, 1, 2))
        self.cov_diag2 = torch.exp(self.logvar2) + torch.diagonal(F_F_tran2, dim1=1, dim2=2)
        self.cov_mat2 = torch.diag_embed(self.logvar2) + F_F_tran2
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
        samples = torch.zeros([self.batch_size, n_samples, self.Y_dim], device=self.device)
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

class FullRankGaussianBNNPosterior(BaseGaussianBNNPosterior):
    """The negative log likelihood (NLL) for a single Gaussian with diagonal covariance matrix
        
    `BaseGaussianNLL.__init__` docstring for the parameter description.

    """
    def __init__(self, Y_dim, device, Y_mean=None, Y_std=None):
        super(FullRankGaussianBNNPosterior, self).__init__(Y_dim, device, Y_mean, Y_std)
        self.tril_idx = torch.tril_indices(self.Y_dim, self.Y_dim, offset=0, device=device) # lower-triangular indices
        self.tril_len = len(self.tril_idx[0])
        self.out_dim = self.Y_dim + self.Y_dim*(self.Y_dim + 1)//2

    def set_sliced_pred(self, pred):
        d = self.Y_dim # for readability
        self.batch_size = pred.shape[0]
        self.mu = pred[:, :d]
        self.tril_elements = pred[:, d:self.out_dim]

    def sample(self, n_samples, sample_seed):
        self.seed_samples(sample_seed)
        return self.sample_full_rank(n_samples, self.mu, self.tril_elements)

    def get_hpd_interval(self):
        return NotImplementedError

class DoubleGaussianBNNPosterior(BaseGaussianBNNPosterior):
    """The negative log likelihood (NLL) for a single Gaussian with diagonal covariance matrix
        
    `BaseGaussianNLL.__init__` docstring for the parameter description.

    """
    def __init__(self, Y_dim, device, Y_mean=None, Y_std=None):
        super(DoubleGaussianBNNPosterior, self).__init__(Y_dim, device, Y_mean, Y_std)
        self.tril_idx = torch.tril_indices(self.Y_dim, self.Y_dim, offset=0, device=device) # lower-triangular indices
        self.tril_len = len(self.tril_idx[0])
        self.out_dim = self.Y_dim**2 + 3*self.Y_dim + 1

    def set_sliced_pred(self, pred):
        d = self.Y_dim # for readability
        self.batch_size = pred.shape[0]
        # First gaussian
        self.mu = pred[:, :d]
        self.tril_elements = pred[:, d:d+self.tril_len]
        self.mu2 = pred[:, d+self.tril_len:2*d+self.tril_len]
        self.tril_elements2 = pred[:, 2*d+self.tril_len:-1]
        #print(pred[:, -1])
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
        samples = torch.zeros([self.batch_size, n_samples, self.Y_dim], device=self.device)
        # Determine first vs. second Gaussian
        unif2 = torch.rand(self.batch_size, n_samples)
        second_gaussian = (self.w2 > unif2)
        # Sample from second Gaussian
        samples2 = self.sample_full_rank(n_samples, self.mu2, self.tril_elements2, as_numpy=False)
        samples[second_gaussian, :] = samples2[second_gaussian, :]
        # Sample from first Gaussian
        samples1 = self.sample_full_rank(n_samples, self.mu, self.tril_elements, as_numpy=False)
        samples[~second_gaussian, :] = samples1[~second_gaussian, :]
        samples = samples.data.cpu().numpy()
        return samples

    def get_hpd_interval(self):
        return NotImplementedError