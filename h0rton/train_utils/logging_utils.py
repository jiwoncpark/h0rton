import numpy as np
from matplotlib.figure import Figure
__all__ = ['get_1d_mapping_fig', 'get_mae', 'interpret_pred', 'get_logdet']

def get_logdet(tril_elements, Y_dim):
    """Returns the log determinant of the covariance matrix

    """
    batch_size = tril_elements.shape[0]
    tril = np.zeros([batch_size, Y_dim, Y_dim])
    tril_idx = np.tril_indices(Y_dim)
    tril_len = len(tril_idx[0])
    tril[:, tril_idx[0], tril_idx[1]] = tril_elements[:, :tril_len] # safeguarding measure
    log_diag_tril = np.diagonal(tril, offset=0, axis1=1, axis2=2) # [batch_size, Y_dim]
    return -np.sum(log_diag_tril, axis=1) # [batch_size,]

def get_1d_mapping_fig(name, mu, Y):
    """Plots the marginal 1D mapping of the mean predictions

    Parameters
    ----------
    name : str
        name of the parameter
    mu : np.array of shape [batch_size,]
        network prediction of the Gaussian mean
    Y : np.array of shape [batch_size,]
        truth label
    which_normal_i : int
        which Gaussian (0 for first, 1 for second)

    Returns
    -------
    matplotlib.FigureCanvas object
        plot of network predictions against truth

    """
    my_dpi = 72.0
    fig = Figure(figsize=(720/my_dpi, 360/my_dpi), dpi=my_dpi, tight_layout=True)
    ax = fig.gca()

    # Reference (perfect) mapping
    perfect = np.linspace(np.min(Y), np.max(Y), 20)
    ax.plot(perfect, np.zeros_like(perfect), linestyle='--', color='b', label="Perfect mapping")

    # For this param
    offset = mu - Y
    ax.scatter(Y, offset, color='tab:blue', marker='o')
    ax.set_title('offset vs. truth ({:s})'.format(name))
    ax.set_ylabel('pred - truth')
    ax.set_xlabel('truth')
    return fig

def _sigmoid(self, x):
    return 1.0/(np.exp(-x) + 1.0)

def get_mae(pred_mu, true_mu, Y_cols):
    """Get the total RMSE of predicted mu of the primary Gaussian wrt the transformed labels mu in a batch of validation data

    Parameters
    ----------
    pred_mu : np.array of shape `[batch_size, Y_dim]`
        predicted means of the primary Gaussian
    true_mu : np.array of shape `[batch_size, Y_dim]`
        true (label) Gaussian means
    Y_cols : np.array of shape `[Y_dim,]`
        the column names

    Returns
    -------
    dict
        total mean of the RMSE for that batch

    """
    abs_error = np.abs(pred_mu - true_mu) # [batch_size, Y_dim]
    mae_dict = {}
    # Total sq error, averaged across examples
    mae_dict['mae'] = np.median(np.sum(abs_error, axis=1)) # float
    # Parameter-wise sq error, averaged across examples
    mae_param = np.median(abs_error, axis=0) # [Y_dim,]
    param_dict = dict(zip(Y_cols, range(len(Y_cols))))
    for k, i in param_dict.items():
        mae_dict[k] = mae_param[i]
    return mae_dict

def interpret_pred(pred, Y_dim):
    """Slice the network prediction into means and cov matrix elements

    Parameters
    ----------
    pred : np.array of shape `[batch_size, out_dim]`
    Y_dim : int
        number of parameters to predict

    Note
    ----
    Currently hardcoded for `DoubleGaussianNLL`. (Update: no longer used; slicing function replaced by the BNNPosterior class.)

    Returns
    -------
    dict
        pred sliced into parameters of the Gaussians to predict

    """
    d = Y_dim # for readability
    mu = pred[:, :d]
    logvar = pred[:, d:2*d]
    F = pred[:, 2*d:4*d]
    mu2 = pred[:, 4*d:5*d]
    logvar2 = pred[:, 5*d:6*d]
    F2 = pred[:, 6*d:8*d]
    alpha = pred[:, -1]
    w2 = 0.5/(np.exp(-alpha) + 1.0)
    pred_dict = dict(
                     mu=mu,
                     logvar=logvar,
                     F=F,
                     mu2=mu2,
                     logvar2=logvar2,
                     F2=F2,
                     w2=w2,
                     )
    return pred_dict

