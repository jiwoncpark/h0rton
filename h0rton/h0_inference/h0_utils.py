import numpy as np
from astropy.cosmology import FlatLambdaCDM
from lenstronomy.Cosmo.lens_cosmo import LensCosmo

__all__ = ["reorder_to_tdlmc", "pred_to_natural_gaussian", "H0Converter"]

def reorder_to_tdlmc(img_array, increasing_dec_i, abcd_ordering_i):
    """Apply the permutation scheme for reordering the list of ra, dec, and time delays to conform to the order in the TDLMC challenge

    Parameters
    ----------
    img_array : array-like
        array of properties corresponding to the AGN images

    Returns
    -------
    array-like
        `img_array` reordered to the TDLMC order
    """
    #print(img_array.shape, self.increasing_dec_i.shape, self.abcd_ordering_i.shape)
    img_array = np.array(img_array)[increasing_dec_i][abcd_ordering_i]
    return img_array

def pred_to_natural_gaussian(pred_mu, pred_cov_mat, shift, scale):
    """Convert the BNN-predicted multivariate Gaussian parameters into the natural space counterparts by reverse transformation

    Parameters
    ----------
    pred_mu : np.array of shape `[Y_dim,]`
    pred_cov_mat : np.array of shape `[Y_dim, Y_dim]`
    scale : np.array of shape `[Y_dim,]`
        vector by which the features were scaled, e.g. the training-set feature standard deviations
    shift : np.array of shape `[Y_dim,]`
        vector by which the features were shifted, e.g. the training-set feature means

    Returns
    -------
    mu : np.array of shape `[Y_dim,]`
        mu in natural space
    cov_mat : np.array of shape `[Y_dim, Y_dim]`
        covariance matrix in natural space

    """
    mu = pred_mu*scale + shift
    A = np.diagflat(scale)
    cov_mat = np.matmul(np.matmul(A, pred_cov_mat), A.T) # There is a better way to do this...
    return mu, cov_mat

class H0Converter:
    """Convert the time-delay distance to H0

    Note
    ----
    This was modified from lenstronomy.Cosmo.cosmo_solver.

    """
    def __init__(self, z_lens, z_src):
        self.cosmo_fiducial = FlatLambdaCDM(H0=70.0, Om0=0.3, Ob0=0.0) # arbitrary
        self.h0_fiducial = self.cosmo_fiducial.H0.value
        self.lens_cosmo = LensCosmo(z_lens=z_lens, z_source=z_src, cosmo=self.cosmo_fiducial)
        self.ddt_fiducial = self.lens_cosmo.ddt

    def get_H0(self, D_dt):
        h0 = self.h0_fiducial * self.ddt_fiducial / D_dt
        return h0