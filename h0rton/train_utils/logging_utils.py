import numpy as np
__all__ = ['get_transformed_rmse', 'interpret_pred']

def get_transformed_rmse(pred_mu, true_mu):
    """Get the total RMSE of predicted mu of the primary Gaussian wrt the transformed labels mu in a batch of validation data

    Parameters
    ----------
    pred_mu : np.array of shape `[batch_size, Y_dim]`
        predicted means of the primary Gaussian
    true_mu : np.array of shape `[batch_size, Y_dim]`
        true (label) Gaussian means

    Returns
    -------
    float
        total sum of the RMSE for that batch

    """
    rmse = np.sum(np.mean((pred_mu - true_mu)**2.0, axis=1))
    return rmse

def interpret_pred(pred, Y_dim):
    """Slice the network prediction into means and cov matrix elements

    Parameters
    ----------
    pred : np.array of shape `[batch_size, out_dim]`
    Y_dim : int
        number of parameters to predict

    Note
    ----
    Currently hardcoded for `DoubleGaussianNLL`

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

