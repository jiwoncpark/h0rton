from abc import ABC, abstractmethod
import numpy as np
from scipy.special import logsumexp
from h0rton.h0_inference.gaussian_bnn_posterior_cpu import sigmoid, logsigmoid
__all__ = ['BaseGaussianNLLCPU', 'DiagonalGaussianNLLCPU', 'FullRankGaussianNLLCPU', 'DoubleGaussianNLLCPU']

log_2_pi = 1.8378770664093453
log_2 = 0.6931471805599453

class BaseGaussianNLLCPU(ABC):
    """Abstract base class to represent the Gaussian negative log likelihood (NLL).

    Gaussian NLLs or mixtures thereof with various forms of the covariance matrix inherit from this class.

    """
    def __init__(self, Y_dim):
        """
        Parameters
        ----------
        Y_dim : int
            number of parameters to predict

        """
        self.Y_dim = Y_dim
        self.sigmoid = sigmoid
        self.logsigmoid = logsigmoid

    @abstractmethod
    def slice(self, pred):
        """Slice the raw network prediction into meaningful Gaussian parameters

        Parameters
        ----------
        pred : np.Tensor of shape `[batch_size, self.Y_dim]`
            the network prediction

        """
        return NotImplemented

    @abstractmethod
    def __call__(self, pred, target):
        """Evaluate the NLL. Must be overridden by subclasses.

        Parameters
        ----------
        pred : np.Tensor
            raw network output for the predictions
        target : np.Tensor
            Y labels

        """
        return NotImplemented

    def nll_diagonal(self, target, mu, logvar):
        """Evaluate the NLL for single Gaussian with diagonal covariance matrix

        Parameters
        ----------
        target : np.Tensor of shape [batch_size, Y_dim]
            Y labels
        mu : np.Tensor of shape [batch_size, Y_dim]
            network prediction of the mu (mean parameter) of the BNN posterior
        logvar : np.Tensor of shape [batch_size, Y_dim]
            network prediction of the log of the diagonal elements of the covariance matrix

        Returns
        -------
        np.Tensor of shape
            NLL values

        """
        precision = np.exp(-logvar)
        # Loss kernel
        loss = precision * (target - mu)**2.0 + logvar
        # Restore prefactors
        loss += np.log(2.0*np.pi)
        loss *= 0.5
        return np.mean(np.sum(loss, axis=1), axis=0)

    def nll_full_rank(self, target, mu, tril_elements, reduce=True):
        """Evaluate the NLL for a single Gaussian with a full-rank covariance matrix

        Parameters
        ----------
        target : np.Tensor of shape [batch_size, Y_dim]
            Y labels
        mu : np.Tensor of shape [batch_size, Y_dim]
            network prediction of the mu (mean parameter) of the BNN posterior
        tril_elements : np.Tensor of shape [batch_size, Y_dim*(Y_dim + 1)//2]
        reduce : bool
            whether to take the mean across the batch

        Returns
        -------
        np.Tensor of shape [batch_size,]
            NLL values

        """
        batch_size, _ = target.shape
        tril = np.zeros([batch_size, self.Y_dim, self.Y_dim])
        tril[:, self.tril_idx[0], self.tril_idx[1]] = tril_elements
        log_diag_tril = np.diagonal(tril, offset=0, axis1=1, axis2=2) # [batch_size, Y_dim]
        logdet_term = -np.sum(log_diag_tril, axis=1) # [batch_size,]
        tril[:, np.eye(self.Y_dim).astype(bool)] = np.exp(log_diag_tril)
        prec_mat = np.matmul(tril, np.transpose(tril, [0, 2, 1])) # [batch_size, Y_dim, Y_dim]
        y_diff = mu - target # [batch_size, Y_dim]
        mahalanobis_term = 0.5*np.sum(
            y_diff*np.sum(prec_mat*np.expand_dims(y_diff, -1), axis=-2), axis=-1) # [batch_size,]
        loss = logdet_term + mahalanobis_term + 0.5*self.Y_dim*log_2_pi
        if reduce:
            return np.mean(loss, axis=0) # float
        else:
            return loss # [batch_size,]

    def nll_mixture(self, target, mu, tril_elements, mu2, tril_elements2, alpha):
        """Evaluate the NLL for a single Gaussian with a full but low-rank plus diagonal covariance matrix

        Parameters
        ----------
        target : np.Tensor of shape [batch_size, Y_dim]
            Y labels
        mu : np.Tensor of shape [batch_size, Y_dim]
            network prediction of the mu (mean parameter) of the BNN posterior for the first Gaussian
        tril_elements : np.Tensor of shape [batch_size, self.tril_len]
            network prediction of the elements in the precision matrix
        mu2 : np.Tensor of shape [batch_size, Y_dim]
            network prediction of the mu (mean parameter) of the BNN posterior for the second Gaussian
        tril_elements2 : np.Tensor of shape [batch_size, self.tril_len]
            network prediction of the elements in the precision matrix for the second Gaussian
        alpha : np.Tensor of shape [batch_size, 1]
            network prediction of the logit of twice the weight on the second Gaussian 

        Note
        ----
        The weight on the second Gaussian is required to be less than 0.5, to make the two Gaussians well-defined.

        Returns
        -------
        np.Tensor of shape [batch_size,]
            NLL values

        """
        batch_size, _ = target.shape
        log_ll = np.empty([batch_size, 2], dtype=None)
        alpha = alpha.reshape(-1)
        log_ll[:, 0] = np.log1p(2.0*np.exp(-alpha)) - log_2 - np.log1p(np.exp(-alpha)) - self.nll_full_rank(target, mu, tril_elements, reduce=False) # [batch_size]
        # np.log(np.tensor([0.5])).double()
        log_ll[:, 1] = -log_2 + self.logsigmoid(alpha) - self.nll_full_rank(target, mu2, tril_elements2, reduce=False) # [batch_size], 0.6931471 = np.log(2)
        log_nll = -logsumexp(log_ll, axis=1)
        return np.mean(log_nll)

class DiagonalGaussianNLLCPU(BaseGaussianNLLCPU):
    """The negative log likelihood (NLL) for a single Gaussian with diagonal covariance matrix
        
    `BaseGaussianNLLCPU.__init__` docstring for the parameter description.

    """
    posterior_name = 'DiagonalGaussianBNNPosterior'

    def __init__(self, Y_dim):
        super(DiagonalGaussianNLLCPU, self).__init__(Y_dim)
        self.out_dim = Y_dim*2

    def __call__(self, pred, target):
        sliced = self.slice(pred)
        return self.nll_diagonal(target, **sliced)

    def slice(self, pred):
        d = self.Y_dim # for readability
        sliced = dict(
                      mu=pred[:, :d],
                      logvar=pred[:, d:]
                      )
        return sliced

class FullRankGaussianNLLCPU(BaseGaussianNLLCPU):
    """The negative log likelihood (NLL) for a single Gaussian with a full-rank covariance matrix
        
    See `BaseGaussianNLLCPU.__init__` docstring for the parameter description.

    """
    posterior_name = 'FullRankGaussianBNNPosterior'

    def __init__(self, Y_dim):
        super(FullRankGaussianNLLCPU, self).__init__(Y_dim)
        self.tril_idx = np.tril_indices(self.Y_dim) # lower-triangular indices
        self.tril_len = len(self.tril_idx[0])
        self.out_dim = self.Y_dim + self.Y_dim*(self.Y_dim + 1)//2

    def __call__(self, pred, target):
        sliced = self.slice(pred)
        return self.nll_full_rank(target, **sliced, reduce=True)

    def slice(self, pred):
        d = self.Y_dim # for readability
        sliced = dict(
                      mu=pred[:, :d],
                      tril_elements=pred[:, d:d+self.tril_len]
                      )
        return sliced

class DoubleGaussianNLLCPU(BaseGaussianNLLCPU):
    """The negative log likelihood (NLL) for a mixture of two Gaussians, each with a full but constrained as low-rank plus diagonal covariance 
        
    Only rank 2 is currently supported. `BaseGaussianNLLCPU.__init__` docstring for the parameter description.

    """
    posterior_name = 'DoubleGaussianBNNPosterior'

    def __init__(self, Y_dim):
        super(DoubleGaussianNLLCPU, self).__init__(Y_dim)
        self.tril_idx = np.tril_indices(self.Y_dim) # lower-triangular indices
        self.tril_len = len(self.tril_idx[0])
        self.out_dim = self.Y_dim**2 + 3*self.Y_dim + 1

    def __call__(self, pred, target):
        sliced = self.slice(pred)
        return self.nll_mixture(target, **sliced)

    def slice(self, pred):
        d = self.Y_dim # for readability
        sliced = dict(
                      mu=pred[:, :d],
                      tril_elements=pred[:, d:d+self.tril_len],
                      mu2=pred[:, d+self.tril_len:2*d+self.tril_len],
                      tril_elements2=pred[:, 2*d+self.tril_len:-1],
                      alpha=pred[:, -1]
                      )
        return sliced