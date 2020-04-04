import numpy as np
from astropy.cosmology import FlatLambdaCDM
from lenstronomy.Cosmo.lens_cosmo import LensCosmo
from hierarc.Sampling.mcmc_sampling import MCMCSampler
import corner
import matplotlib.pyplot as plt

__all__ = ["reorder_to_tdlmc", "pred_to_natural_gaussian", "CosmoConverter", "get_lognormal_stats", "remove_outliers_from_lognormal", "combine_lenses"]

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

class CosmoConverter:
    """Convert the time-delay distance to H0 and vice versa

    Note
    ----
    This was modified from lenstronomy.Cosmo.cosmo_solver to handle array types.

    """
    def __init__(self, z_lens, z_src):
        self.cosmo_fiducial = FlatLambdaCDM(H0=70.0, Om0=0.3, Ob0=0.0) # arbitrary
        self.h0_fiducial = self.cosmo_fiducial.H0.value
        self.lens_cosmo = LensCosmo(z_lens=z_lens, z_source=z_src, cosmo=self.cosmo_fiducial)
        self.ddt_fiducial = self.lens_cosmo.ddt

    def get_H0(self, D_dt):
        H0 = self.h0_fiducial * self.ddt_fiducial / D_dt
        return H0

    def get_D_dt(self, H0):
        D_dt = self.h0_fiducial * self.ddt_fiducial / H0
        return D_dt

def get_lognormal_stats(samples, weights=None):
    """Compute lognormal stats assuming the samples are drawn from a lognormal distribution

    """
    if weights is None:
        weights = np.ones_like(samples)
    log_samples = np.log(samples)
    mu = np.average(log_samples, weights=weights)
    sig2 = np.average((log_samples - mu)**2.0, weights=weights)
    mode = np.exp(mu - sig2)
    std = ((np.exp(sig2) - 1.0)*(np.exp(2*mu - sig2)))**0.5
    stats = dict(
                 mu=mu,
                 sigma=sig2**0.5,
                 mode=mode,
                 std=std
                 )
    return stats

def remove_outliers_from_lognormal(data, level=3):
    """Remove extreme outliers corresponding to level-STD away from the mean
    
    Parameters
    ----------
    data : np.array
        data expected to follow a lognormal distribution

    """
    # Quantiles are preserved under monotonic transformations
    log_data = np.log(data)
    return data[abs(log_data - np.mean(log_data)) < level*np.std(log_data)]

def combine_lenses(true_cosmo, D_dt_mu, D_dt_sigma, z_lens, z_src, samples_save_path, corner_save_path=None):
    """Combine lenses in the D_dt space

    """
    n_test = len(D_dt_mu)

    kwargs_posterior_list = []
    for i in range(n_test):
        kwargs_posterior = {'z_lens': z_lens[i], 'z_source': z_src[i], 
                            'ddt_mu': D_dt_mu[i], 'ddt_sigma': D_dt_sigma[i],
                           'likelihood_type': 'TDLogNorm'}
        kwargs_posterior_list.append(kwargs_posterior)

    kwargs_lower_cosmo = {'h0': 50.0}
    kwargs_lower_lens = {}
    kwargs_lower_kin = {}

    kwargs_upper_cosmo = {'h0': 90.0}
    kwargs_upper_lens = {}
    kwargs_upper_kin = {}

    kwargs_fixed_cosmo = {'om': true_cosmo.Om0}
    kwargs_fixed_lens = {}
    kwargs_fixed_kin = {}

    kwargs_mean_start = {'kwargs_cosmo': {'h0': 70.0},
                         'kwargs_lens': {},
                         'kwargs_kin': {}}

    kwargs_sigma_start = {'kwargs_cosmo': {'h0': 10.0},
                         'kwargs_lens': {},
                         'kwargs_kin': {}}

    n_walkers = 10
    n_run = 100
    n_burn = 400

    kwargs_bounds = {'kwargs_lower_cosmo': kwargs_lower_cosmo,
                'kwargs_lower_lens': kwargs_lower_lens,
                'kwargs_lower_kin': kwargs_lower_kin,
                'kwargs_upper_cosmo': kwargs_upper_cosmo,
                'kwargs_upper_lens': kwargs_upper_lens,
                'kwargs_upper_kin': kwargs_upper_kin,
                'kwargs_fixed_cosmo': kwargs_fixed_cosmo,
                'kwargs_fixed_lens': kwargs_fixed_lens,
                'kwargs_fixed_kin': kwargs_fixed_kin}

    cosmology = 'FLCDM'  # available models: 'FLCDM', "FwCDM", "w0waCDM", "oLCDM"
    mcmc_sampler = MCMCSampler(kwargs_posterior_list, cosmology, kwargs_bounds, ppn_sampling=False,
                     lambda_mst_sampling=False, lambda_mst_distribution='NONE', anisotropy_sampling=False,
                               kappa_ext_sampling=False, kappa_ext_distribution='NONE',
                     anisotropy_model='NONE', anisotropy_distribution='NONE', custom_prior=None, interpolate_cosmo=True, num_redshift_interp=100,
                     cosmo_fixed=None)

    mcmc_samples, log_prob_cosmo = mcmc_sampler.mcmc_emcee(n_walkers, n_run, n_burn, kwargs_mean_start, kwargs_sigma_start)
    np.save(samples_save_path, mcmc_samples)

    if corner_save_path is not None:
        corner.corner(mcmc_samples, show_titles=True, labels=mcmc_sampler.param_names(latex_style=True))
        plt.show()
        plt.savefig(corner_save_path)
        plt.close()

    return mcmc_samples, log_prob_cosmo