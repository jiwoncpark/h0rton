import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from addict import Dict

class Plotter:
    """
    Interprets the network output for making plots

    """
    def __init__(self, cov_mat, Y_dim, device):
        self.cov_mat = cov_mat
        self.Y_dim = Y_dim
        self.device = device
        self.mixture_cmap = cm.tab20
        self.mixture_norm = colors.Normalize(vmin=0, vmax=20) # FIXME: hardcode max num of normals

    def _get_normal_params(self, pred):
        """Compute the mean and cov mat of a normal from the network output

        Parameters
        ----------
        pred_params : dict
            the params sliced from pred that are meaningful components
            of normal mean and cov mat

        Returns
        -------
        dict
            the normal parameters
        """
        batch_size, self.out_dim = pred.shape

        normal = self._slice_pred_params(pred)
        F_tran_F = np.matmul(normal.F, np.swapaxes(normal.F, 1, 2))
        cov_mat = np.apply_along_axis(np.diag, -1, np.exp(normal.logvar)) + F_tran_F
        cov_diag = np.exp(normal.logvar) + np.diagonal(F_tran_F, axis1=1, axis2=2)
        assert np.array_equal(cov_mat.shape, [batch_size, self.Y_dim, self.Y_dim])
        assert np.array_equal(cov_diag.shape, [batch_size, self.Y_dim])
        normal.update(
                      cov_mat=cov_mat,
                      cov_diag=cov_diag,
                      )
        return normal

    def _slice_pred_params(self, pred):
        """Slices the portion of pred corresponding to a normal
        into meaningful components of normal mean and cov mat

        """
        d = self.Y_dim # for readability
        batch_size, self.out_dim = pred.shape
        mu = pred[:, :d] # [, Y_dim]
        logvar = pred[:, d:2*d] # [, Y_dim])
        F = pred[:, 2*d:].reshape(batch_size, d, 2) # [, 2*Y_dim]
        pred_params = Dict(
                           mu=mu,
                           logvar=logvar,
                           F=F,
                           alpha=np.ones(batch_size))
        return pred_params

    def set_normal_mixture_params(self, pred):
        d = self.Y_dim # for readability
        batch_size, self.out_dim = pred.shape

        if self.cov_mat == 'diagonal':
            mu = pred[:, :d]
            logvar = pred[:, d:2*d]
            normal = Dict(
                          mu=mu,
                          logvar=logvar, 
                          F=None,
                          cov_mat=np.apply_along_axis(np.diag, -1, np.exp(logvar)),
                          cov_diag=np.exp(logvar),
                          alpha=np.ones(batch_size),
                          )                          
            normal_mixture = [normal]

        elif self.cov_mat == 'low_rank':
            normal = self._get_normal_params(pred)
            normal_mixture = [normal]

        elif self.cov_mat == 'double':
            # Boring and arbitrary slicing... bear with me...
            alpha = pred[:, -1]
            normal_mixture = [] # each element is a dict of params for a normal
            for i in range(2): 
                normal_comp = self._get_normal_params(pred[:, i*4*d: (i+1)*4*d])
                normal_mixture.append(normal_comp)

            # FIXME: only works for 2 b/c of alphas, if we want to ensure alpha_1 > alpha_2
            # FIXME: rename network output alpha to alpha_logit
            alpha2 = 0.5*self._sigmoid(alpha)
            normal_mixture[0].update(alpha=1.0 - alpha2)
            normal_mixture[1].update(alpha=alpha2)
            normal_mixture = normal_mixture

        else:
            raise NotImplementedError

        self.normal_mixture = normal_mixture

    def get_1d_mapping_fig(self, name, idx, Y):
        """Plots the marginal 1D mapping

        Parameters
        ----------
        name : str
            name of the parameter
        idx : int
            parameter index
        Y : 1D array-like
            truth label

        Returns
        -------
        matplotlib.FigureCanvas object
            plot of network predictions against truth

        """
        n_data = len(Y)
        my_dpi = 72.0
        fig = Figure(figsize=(720/my_dpi, 360/my_dpi), dpi=my_dpi, tight_layout=True)
        ax = fig.gca()

        # Reference (perfect) mapping
        perfect = np.linspace(np.min(Y), np.max(Y), 20)
        ax.plot(perfect, np.zeros_like(perfect), linestyle='--', color='b', label="Perfect mapping")

        # Network predictions
        for normal_idx, normal in enumerate(self.normal_mixture):
            offset = normal.mu[:, idx] - Y
            rgba_colors = np.zeros((n_data, 4))
            rgba_colors[:, 0] = 1.0
            rgba_colors[:, 3] = normal.alpha
            ax.errorbar(Y, offset, 
                        ecolor=rgba_colors, color=self.mixture_cmap(self.mixture_norm(normal_idx)),
                        marker='o', linewidth=0.0,
                        yerr=normal.cov_diag[:, idx], elinewidth=0.5)
        ax.set_title('offset vs. truth ({:s})'.format(name))
        ax.set_ylabel('pred - truth')
        ax.set_xlabel('truth')
        return fig

    def _sigmoid(self, x):
        return 1.0/(np.exp(-x) + 1.0)

if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from torch.utils.data.sampler import SubsetRandomSampler
    import torchvision.transforms as transforms
    import torch

    pred_Y = np.ones((17, 2))
    pred_plt = np.ones((17, 9))
    plotter = Plotter('double', 2,  torch.device('cuda'))
    plotter.set_normal_mixture_params(pred_plt)
    _ = plotter.get_1d_mapping_fig('some param', 1, pred_Y[:, 1])



