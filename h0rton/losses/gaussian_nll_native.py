from abc import ABC, abstractmethod
import numpy as np
import torch
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.lowrank_multivariate_normal import LowRankMultivariateNormal
__all__ = ['BaseGaussianNLLNative', 'DiagonalGaussianNLLNative', 'LowRankGaussianNLLNative', 'DoubleLowRankGaussianNLLNative', 'FullRankGaussianNLLNative', 'DoubleGaussianNLLNative']

log_2_pi = 1.8378770664093453
log_2 = 0.6931471805599453

class BaseGaussianNLLNative(ABC):
    """Abstract base class to represent the Gaussian negative log likelihood (NLLNative).

    Gaussian NLLNatives or mixtures thereof with various forms of the covariance matrix inherit from this class.

    """
    def __init__(self, Y_dim, device):
        """
        Parameters
        ----------
        Y_dim : int
            number of parameters to predict
        device : torch.device object

        """
        self.Y_dim = Y_dim
        self.device = device
        self.sigmoid = torch.nn.Sigmoid()
        self.logsigmoid = torch.nn.LogSigmoid()

    @abstractmethod
    def slice(self, pred):
        """Slice the raw network prediction into meaningful Gaussian parameters

        Parameters
        ----------
        pred : torch.Tensor of shape `[batch_size, self.Y_dim]`
            the network prediction

        """
        return NotImplemented

    @abstractmethod
    def __call__(self, pred, target):
        """Evaluate the NLLNative. Must be overridden by subclasses.

        Parameters
        ----------
        pred : torch.Tensor
            raw network output for the predictions
        target : torch.Tensor
            Y labels

        """
        return NotImplemented

    def nll_diagonal(self, target, mu, logvar):
        """Evaluate the NLLNative for single Gaussian with diagonal covariance matrix

        Parameters
        ----------
        target : torch.Tensor of shape [batch_size, Y_dim]
            Y labels
        mu : torch.Tensor of shape [batch_size, Y_dim]
            network prediction of the mu (mean parameter) of the BNN posterior
        logvar : torch.Tensor of shape [batch_size, Y_dim]
            network prediction of the log of the diagonal elements of the covariance matrix

        Returns
        -------
        torch.Tensor of shape
            NLL values

        """
        precision = torch.exp(-logvar)
        # Loss kernel
        loss = precision * (target - mu)**2.0 + logvar
        # Restore prefactors
        loss += np.log(2.0*np.pi)
        loss *= 0.5
        return torch.mean(torch.sum(loss, dim=1), dim=0)

    def nll_low_rank(self, target, mu, logvar, F, reduce=True):
        """Evaluate the NLLNative for a single Gaussian with a full but low-rank plus diagonal covariance matrix

        Parameters
        ----------
        target : torch.Tensor of shape [batch_size, Y_dim]
            Y labels
        mu : torch.Tensor of shape [batch_size, Y_dim]
            network prediction of the mu (mean parameter) of the BNN posterior
        logvar : torch.Tensor of shape [batch_size, Y_dim]
            network prediction of the log of the diagonal elements of the covariance matrix
        F : torch.Tensor of shape [batch_size, rank*Y_dim]
            network prediction of the low rank portion of the covariance matrix
        reduce : bool
            whether to take the mean across the batch

        Returns
        -------
        torch.Tensor of shape [batch_size,]
            NLL values

        """
        # 1/(Y_dim - 1) * (sq_mahalanobis + log(det of \Sigma))
        batch_size, _ = target.shape # self.Y_dim = Y_dim - 1
        rank = 2 # FIXME: hardcoded for rank 2
        F = F.reshape([batch_size, self.Y_dim, rank]) 
        lr_mvn = LowRankMultivariateNormal(loc=mu, cov_factor=F, cov_diag=torch.exp(logvar))
        loss = -lr_mvn.log_prob(target)

        if reduce==True:
            return torch.mean(loss, dim=0) # float
        else:
            return loss # [batch_size,]

    def nll_mixture_low_rank(self, target, mu, logvar, F, mu2, logvar2, F2, alpha):
        """Evaluate the NLLNative for a single Gaussian with a full but low-rank plus diagonal covariance matrix
        Parameters
        ----------
        target : torch.Tensor of shape [batch_size, Y_dim]
            Y labels
        mu : torch.Tensor of shape [batch_size, Y_dim]
            network prediction of the mu (mean parameter) of the BNN posterior for the first Gaussian
        logvar : torch.Tensor of shape [batch_size, Y_dim]
            network prediction of the log of the diagonal elements of the covariance matrix for the first Gaussian
        F : torch.Tensor of shape [batch_size, rank*Y_dim]
            network prediction of the low rank portion of the covariance matrix for the first Gaussian
        mu2 : torch.Tensor of shape [batch_size, Y_dim]
            network prediction of the mu (mean parameter) of the BNN posterior for the second Gaussian
        logvar2 : torch.Tensor of shape [batch_size, Y_dim]
            network prediction of the log of the diagonal elements of the covariance matrix for the second Gaussian
        F2 : torch.Tensor of shape [batch_size, rank*Y_dim]
            network prediction of the low rank portion of the covariance matrix for the second Gaussian
        alpha : torch.Tensor of shape [batch_size, 1]
            network prediction of the logit of twice the weight on the second Gaussian 
        reduce : bool
            whether to take the mean across the batch
        Note
        ----
        The weight on the second Gaussian is required to be less than 0.5, to make the two Gaussians well-defined.
        Returns
        -------
        torch.Tensor of shape [batch_size,]
            NLL values
        """
        batch_size, _ = target.shape
        alpha = alpha.reshape(-1)
        #log_w1p1 = -alpha -torch.log1p(torch.exp(-alpha)) - self.nll_low_rank(target, mu, logvar, F=F, reduce=False) # [batch_size]
        #log_w2p2 = self.logsigmoid(alpha) - self.nll_low_rank(target, mu2, logvar2, F=F2, reduce=False) # [batch_size], 0.6931471 = np.log(2)
        log_w1p1 = torch.log1p(2.0*torch.exp(-alpha)) - log_2 - torch.log1p(torch.exp(-alpha)) - self.nll_low_rank(target, mu, logvar, F=F, reduce=False)
        log_w2p2 = -log_2 + self.logsigmoid(alpha) - self.nll_low_rank(target, mu2, logvar2, F=F2, reduce=False)
        stacked = torch.stack([log_w1p1, log_w2p2], dim=1)
        log_nll = -torch.logsumexp(stacked, dim=1)
        return torch.mean(log_nll)

    def nll_full_rank(self, target, mu, tril_elements, reduce=True):
        """Evaluate the NLLNative for a single Gaussian with a full-rank covariance matrix

        Parameters
        ----------
        target : torch.Tensor of shape [batch_size, Y_dim]
            Y labels
        mu : torch.Tensor of shape [batch_size, Y_dim]
            network prediction of the mu (mean parameter) of the BNN posterior
        tril_elements : torch.Tensor of shape [batch_size, Y_dim*(Y_dim + 1)//2]
        reduce : bool
            whether to take the mean across the batch

        Returns
        -------
        torch.Tensor of shape [batch_size,]
            NLL values

        """
        batch_size, _ = target.shape
        tril = torch.zeros([batch_size, self.Y_dim, self.Y_dim], device=self.device, dtype=None)
        tril[:, self.tril_idx[0], self.tril_idx[1]] = tril_elements
        log_diag_tril = torch.diagonal(tril, offset=0, dim1=1, dim2=2) # [batch_size, Y_dim]
        tril[:, torch.eye(self.Y_dim, dtype=bool)] = torch.exp(log_diag_tril)
        prec_mat = torch.bmm(tril, torch.transpose(tril, 1, 2))
        mvn = MultivariateNormal(loc=mu, precision_matrix=prec_mat)
        loss = -mvn.log_prob(target)
        if reduce:
            return torch.mean(loss, dim=0) # float
        else:
            return loss # [batch_size,]

    def nll_mixture(self, target, mu, tril_elements, mu2, tril_elements2, alpha):
        """Evaluate the NLLNative for a single Gaussian with a full but low-rank plus diagonal covariance matrix

        Parameters
        ----------
        target : torch.Tensor of shape [batch_size, Y_dim]
            Y labels
        mu : torch.Tensor of shape [batch_size, Y_dim]
            network prediction of the mu (mean parameter) of the BNN posterior for the first Gaussian
        tril_elements : torch.Tensor of shape [batch_size, self.tril_len]
            network prediction of the elements in the precision matrix
        mu2 : torch.Tensor of shape [batch_size, Y_dim]
            network prediction of the mu (mean parameter) of the BNN posterior for the second Gaussian
        tril_elements2 : torch.Tensor of shape [batch_size, self.tril_len]
            network prediction of the elements in the precision matrix for the second Gaussian
        alpha : torch.Tensor of shape [batch_size, 1]
            network prediction of the logit of twice the weight on the second Gaussian 

        Note
        ----
        The weight on the second Gaussian is required to be less than 0.5, to make the two Gaussians well-defined.

        Returns
        -------
        torch.Tensor of shape [batch_size,]
            NLL values

        """
        batch_size, _ = target.shape
        alpha = alpha.reshape(-1)
        log_w1p1 = torch.log1p(2.0*torch.exp(-alpha)) - log_2 - torch.log1p(torch.exp(-alpha)) - self.nll_full_rank(target, mu, tril_elements, reduce=False) # [batch_size]
        log_w2p2 = -log_2 + self.logsigmoid(alpha) - self.nll_full_rank(target, mu2, tril_elements2, reduce=False) # [batch_size]
        stacked = torch.stack([log_w1p1, log_w2p2], dim=1)
        log_nll = -torch.logsumexp(stacked, dim=1)
        return torch.mean(log_nll)

class DiagonalGaussianNLLNative(BaseGaussianNLLNative):
    """The negative log likelihood (NLLNative) for a single Gaussian with diagonal covariance matrix
        
    `BaseGaussianNLLNative.__init__` docstring for the parameter description.

    """
    posterior_name = 'DiagonalGaussianBNNPosterior'

    def __init__(self, Y_dim, device):
        super(DiagonalGaussianNLLNative, self).__init__(Y_dim, device)
        self.out_dim = Y_dim*2

    def __call__(self, pred, target):
        return self.nll_diagonal(target, *self.slice(pred))

    def slice(self, pred):
        d = self.Y_dim # for readability
        return torch.split(pred, [d, d], dim=1)

class LowRankGaussianNLLNative(BaseGaussianNLLNative):
    """The negative log likelihood (NLLNative) for a single Gaussian with a full but constrained as low-rank plus diagonal covariance matrix
        
    Only rank 2 is currently supported. `BaseGaussianNLLNative.__init__` docstring for the parameter description.

    """
    posterior_name = 'LowRankGaussianBNNPosterior'

    def __init__(self, Y_dim, device):
        super(LowRankGaussianNLLNative, self).__init__(Y_dim, device)
        self.out_dim = Y_dim*4

    def __call__(self, pred, target):
        return self.nll_low_rank(target, *self.slice(pred), reduce=True)

    def slice(self, pred):
        d = self.Y_dim # for readability
        return torch.split(pred, [d, d, 2*d], dim=1)

class FullRankGaussianNLLNative(BaseGaussianNLLNative):
    """The negative log likelihood (NLLNative) for a single Gaussian with a full-rank covariance matrix
        
    See `BaseGaussianNLLNative.__init__` docstring for the parameter description.

    """
    posterior_name = 'FullRankGaussianBNNPosterior'

    def __init__(self, Y_dim, device):
        super(FullRankGaussianNLLNative, self).__init__(Y_dim, device)
        self.tril_idx = torch.tril_indices(self.Y_dim, self.Y_dim, offset=0, device=device) # lower-triangular indices
        self.tril_len = len(self.tril_idx[0])
        self.out_dim = self.Y_dim + self.Y_dim*(self.Y_dim + 1)//2

    def __call__(self, pred, target):
        return self.nll_full_rank(target, *self.slice(pred), reduce=True)

    def slice(self, pred):
        d = self.Y_dim # for readability
        return torch.split(pred, [d, self.tril_len], dim=1)

class DoubleLowRankGaussianNLLNative(BaseGaussianNLLNative):
    """The negative log likelihood (NLLNative) for a mixture of two Gaussians, each with a full but constrained as low-rank plus diagonal covariance 
        
    Only rank 2 is currently supported. `BaseGaussianNLLNative.__init__` docstring for the parameter description.

    """
    posterior_name = 'DoubleLowRankGaussianBNNPosterior'

    def __init__(self, Y_dim, device):
        super(DoubleLowRankGaussianNLLNative, self).__init__(Y_dim, device)
        self.out_dim = Y_dim*8 + 1

    def __call__(self, pred, target):
        return self.nll_mixture_low_rank(target, *self.slice(pred))

    def slice(self, pred):
        d = self.Y_dim # for readability
        #mu, logvar, F, mu2, logvar2, F2, alpha
        return torch.split(pred, [d, d, 2*d, d, d, 2*d, 1], dim=1)

class DoubleGaussianNLLNative(BaseGaussianNLLNative):
    """The negative log likelihood (NLLNative) for a mixture of two Gaussians, each with a full but constrained as low-rank plus diagonal covariance 
        
    Only rank 2 is currently supported. `BaseGaussianNLLNative.__init__` docstring for the parameter description.

    """
    posterior_name = 'DoubleGaussianBNNPosterior'

    def __init__(self, Y_dim, device):
        super(DoubleGaussianNLLNative, self).__init__(Y_dim, device)
        self.tril_idx = torch.tril_indices(self.Y_dim, self.Y_dim, offset=0, device=device) # lower-triangular indices
        self.tril_len = len(self.tril_idx[0])
        self.out_dim = self.Y_dim**2 + 3*self.Y_dim + 1

    def __call__(self, pred, target):
        return self.nll_mixture(target, *self.slice(pred))

    def slice(self, pred):
        d = self.Y_dim # for readability
        return torch.split(pred, [d, self.tril_len, d, self.tril_len, 1], dim=1)