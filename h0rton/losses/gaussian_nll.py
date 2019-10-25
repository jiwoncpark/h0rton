from abc import ABC, abstractmethod
import torch
__all__ = ['BaseGaussianNLL', 'DiagonalGaussianNLL', 'LowRankGaussianNLL', 'DoubleGaussianNLL']

class BaseGaussianNLL(ABC):
    """Abstract base class to represent the Gaussian negative log likelihood (NLL).

    Gaussian NLLs or mixtures thereof with various forms of the covariance matrix inherit from this class.

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

    @abstractmethod
    def __call__(self, pred, target):
        """Evaluate the NLL. Must be overridden by subclasses.

        Parameters
        ----------
        pred : torch.Tensor
            raw network output for the predictions
        target : torch.Tensor
            Y labels

        """
        return NotImplemented

    def nll_diagonal(self, target, mu, logvar):
        """Evaluate the NLL for single Gaussian with diagonal covariance matrix

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
        print(torch.sum(precision * (target - mu)**2.0 + logvar, dim=1))
        return torch.mean(torch.sum(precision * (target - mu)**2.0 + logvar, dim=1), dim=0)

    def nll_lowrank(self, target, mu, logvar, F, reduce=True):
        """Evaluate the NLL for a single Gaussian with a full but low-rank plus diagonal covariance matrix

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
        inv_var = torch.exp(-logvar) # [batch_size, self.Y_dim]
        diag_inv_var = torch.diag_embed(inv_var)  # [batch_size, self.Y_dim, self.Y_dim]
        diag_prod = F**2.0 * inv_var.reshape([batch_size, self.Y_dim, 1]) # [batch_size, self.Y_dim, rank] after broadcasting
        off_diag_prod = torch.prod(F, dim=2)*inv_var # [batch_size, self.Y_dim]
        #batchdiag = torch.diag_embed(torch.exp(logvar)) # [batch_size, self.Y_dim, self.Y_dim]
        #batch_eye = torch.eye(rank).reshape(1, rank, rank).repeat(batch_size, 1, 1) # [batch_size, rank, rank]
        # (25), (26) in Miller et al 2016
        log_det = torch.sum(logvar, dim=1) # [batch_size]
        M00 = torch.sum(diag_prod[:, :, 0], dim=1) + 1.0 # [batch_size]
        M11 = torch.sum(diag_prod[:, :, 1], dim=1) + 1.0 # [batch_size]
        M12 = torch.sum(off_diag_prod, dim=1) # [batch_size]
        det_M = M00*M11 - M12**2.0 # [batch_size]
        log_det += torch.log(det_M) # [batch_size,]

        inv_M = torch.ones([batch_size, rank, rank], device=self.device).double()
        inv_M[:, 0, 0] = M11
        inv_M[:, 1, 1] = M00
        inv_M[:, 1, 0] = -M12
        inv_M[:, 0, 1] = -M12
        inv_M /= det_M.reshape(batch_size, 1, 1)

        # (27) in Miller et al 2016
        inv_cov = diag_inv_var - torch.bmm(torch.bmm(torch.bmm(torch.bmm(diag_inv_var, F), inv_M), torch.transpose(F, 1, 2)), diag_inv_var) # [batch_size, self.Y_dim, self.Y_dim]
        sq_mahalanobis = torch.bmm(torch.bmm((mu - target).reshape(batch_size, 1, self.Y_dim), inv_cov), (mu - target).reshape(batch_size, self.Y_dim, 1)).reshape(-1) # [batch_size,]
        
        if reduce==True:
            return torch.mean(sq_mahalanobis + log_det, dim=0) # float
        else:
            return sq_mahalanobis + log_det # [batch_size,]

    def nll_mixture(self, target, mu, logvar, F, mu2, logvar2, F2, alpha):
        """Evaluate the NLL for a single Gaussian with a full but low-rank plus diagonal covariance matrix

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
        rank = 2
        log_ll = torch.empty([batch_size, 2], device=self.device)
        sigmoid = torch.nn.Sigmoid()
        logsigmoid = torch.nn.LogSigmoid()
        alpha = alpha.reshape(-1)
        log_ll[:, 0] = torch.log(1.0 - 0.5*sigmoid(alpha)) - self.nll_lowrank(target, mu, logvar, F=F, reduce=False) # [batch_size]
        # torch.log(torch.tensor([0.5], device=self.device)).double()
        log_ll[:, 1] = -0.6931471 + logsigmoid(alpha) - self.nll_lowrank(target, mu2, logvar2, F=F2, reduce=False) # [batch_size], 0.6931471 = np.log(2)
        log_nll = -torch.logsumexp(log_ll, dim=1) 
        return torch.mean(log_nll)

class DiagonalGaussianNLL(BaseGaussianNLL):
    """The negative log likelihood (NLL) for a single Gaussian with diagonal covariance matrix
        
    `BaseGaussianNLL.__init__` docstring for the parameter description.

    """
    def __init__(self, Y_dim, device):
        super(DiagonalGaussianNLL, self).__init__(Y_dim, device)
        self.out_dim = Y_dim*2

    def __call__(self, pred, target):
        d = self.Y_dim # for readability
        mu = pred[:, :d]
        logvar = pred[:, d:]
        return self.nll_diagonal(target, mu, logvar)

class LowRankGaussianNLL(BaseGaussianNLL):
    """The negative log likelihood (NLL) for a single Gaussian with a full but constrained as low-rank plus diagonal covariance matrix
        
    Only rank 2 is currently supported. `BaseGaussianNLL.__init__` docstring for the parameter description.

    """
    def __init__(self, Y_dim, device):
        super(LowRankGaussianNLL, self).__init__(Y_dim, device)
        self.out_dim = Y_dim*4

    def __call__(self, pred, target):
        d = self.Y_dim # for readability
        mu = pred[:, :d]
        logvar = pred[:, d:2*d]
        F = pred[:, 2*d:]
        return self.nll_lowrank(target, mu, logvar, F, reduce=True)

class DoubleGaussianNLL(BaseGaussianNLL):
    """The negative log likelihood (NLL) for a mixture of two Gaussians, each with a full but constrained as low-rank plus diagonal covariance 
        
    Only rank 2 is currently supported. `BaseGaussianNLL.__init__` docstring for the parameter description.

    """
    def __init__(self, Y_dim, device):
        super(DoubleGaussianNLL, self).__init__(Y_dim, device)
        self.out_dim = Y_dim*8 + 1

    def __call__(self, pred, target):
        d = self.Y_dim # for readability
        mu = pred[:, :d]
        logvar = pred[:, d:2*d]
        F = pred[:, 2*d:4*d]
        mu2 = pred[:, 4*d:5*d]
        logvar2 = pred[:, 5*d:6*d]
        F2 = pred[:, 6*d:8*d]
        alpha = pred[:, -1]
        return self.nll_mixture(target, mu, logvar, F, mu2, logvar2, F2, alpha)