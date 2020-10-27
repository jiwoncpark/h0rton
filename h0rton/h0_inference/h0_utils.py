import os
import numpy as np
from astropy.cosmology import FlatLambdaCDM
from lenstronomy.Cosmo.lens_cosmo import LensCosmo
from hierarc.Sampling.mcmc_sampling import MCMCSampler
import corner
import matplotlib.pyplot as plt
from scipy.stats import norm, median_abs_deviation

__all__ = ["reorder_to_tdlmc", "pred_to_natural_gaussian", "CosmoConverter", "get_lognormal_stats", "get_lognormal_stats_naive", "get_normal_stats", "get_normal_stats_naive", "remove_outliers_from_lognormal", "combine_lenses", "gaussian_ll_pdf"]

MAD_to_sig = 1.0/norm.ppf(0.75) # 1.4826 built into scipy, so not used.

class DeltaFunction:
    def __init__(self, true_value=0.0):
        self.true_value = true_value
    def rvs(self, random_state=None):
        return self.true_value

def gaussian_ll_pdf(x, mu, sigma):
    """Evaluates the (unnormalized) log of the normal PDF at point x
    
    Parameters
    ----------
    x : float or array-like
        point at which to evaluate the log pdf
    mu : float or array-like
        mean of the normal on a linear scale
    sigma : float or array-like
        standard deviation of the normal on a linear scale
        
    """
    log_pdf = -0.5*(x - mu)**2.0/sigma**2.0 #- np.log(sigma) - 0.5*np.log(2.0*np.pi)
    return log_pdf

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
    #print(img_array, increasing_dec_i.shape, abcd_ordering_i.shape)
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

    Note
    ----
    Derive it or go here: https://math.stackexchange.com/questions/332441/affine-transformation-applied-to-a-multivariate-gaussian-random-variable-what

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
    def __init__(self, z_lens, z_src, H0=70.0, Om0=0.3):
        self.cosmo_fiducial = FlatLambdaCDM(H0=H0, Om0=Om0) # arbitrary
        self.h0_fiducial = self.cosmo_fiducial.H0.value
        self.lens_cosmo = LensCosmo(z_lens=z_lens, z_source=z_src, cosmo=self.cosmo_fiducial)
        self.ddt_fiducial = self.lens_cosmo.ddt

    def get_H0(self, D_dt):
        H0 = self.h0_fiducial * self.ddt_fiducial / D_dt
        return H0

    def get_D_dt(self, H0):
        D_dt = self.h0_fiducial * self.ddt_fiducial / H0
        return D_dt

def get_lognormal_stats(all_samples):
    """Compute lognormal stats robustly, using median stats, assuming the samples are drawn from a lognormal distribution

    """
    is_nan_mask = np.logical_or(np.isnan(all_samples), ~np.isfinite(all_samples))
    samples = all_samples[~is_nan_mask]
    log_samples = np.log(samples)
    mu = np.median(log_samples)
    sig2 = median_abs_deviation(log_samples, axis=None, scale='normal')**2.0
    mode = np.exp(mu - sig2)
    std = ((np.exp(sig2) - 1.0)*(np.exp(2*mu - sig2)))**0.5
    stats = dict(
                 mu=mu,
                 sigma=sig2**0.5,
                 mode=mode,
                 std=std
                 )
    return stats

def get_lognormal_stats_naive(all_samples, all_weights=None):
    """Compute lognormal stats assuming the samples are drawn from a lognormal distribution

    """
    if all_weights is None:
        all_weights = np.ones_like(all_samples)
    is_nan_mask = np.logical_or(np.logical_or(np.isnan(all_weights), ~np.isfinite(all_weights)), np.isnan(all_samples))
    all_weights[~is_nan_mask] = all_weights[~is_nan_mask]/np.sum(all_weights[~is_nan_mask])
    samples = all_samples[~is_nan_mask]
    weights = all_weights[~is_nan_mask]
    n_samples = len(samples)
    log_samples = np.log(samples)
    mu = np.average(log_samples, weights=weights)
    sig2 = np.average((log_samples - mu)**2.0, weights=weights)*(n_samples/(n_samples - 1))
    mode = np.exp(mu - sig2)
    std = ((np.exp(sig2) - 1.0)*(np.exp(2*mu - sig2)))**0.5
    stats = dict(
                 mu=mu,
                 sigma=sig2**0.5,
                 mode=mode,
                 std=std
                 )
    return stats

def get_normal_stats(all_samples):
    is_nan_mask = np.logical_or(np.isnan(all_samples), ~np.isfinite(all_samples))
    samples = all_samples[~is_nan_mask]
    mean = np.median(samples)
    std = median_abs_deviation(samples, axis=None, scale='normal')
    stats = dict(
                 mean=mean,
                 std=std
                 )
    return stats

def get_normal_stats_naive(all_samples, all_weights):
    is_nan_mask = np.logical_or(np.logical_or(np.isnan(all_weights), ~np.isfinite(all_weights)), np.isnan(all_samples))
    all_weights[~is_nan_mask] = all_weights[~is_nan_mask]/np.sum(all_weights[~is_nan_mask])
    samples = all_samples[~is_nan_mask]
    weights = all_weights[~is_nan_mask]
    mean = np.average(samples, weights=weights)
    std = np.average((samples - mean)**2.0, weights=weights)**0.5
    #print(mean, std)
    stats = dict(
                 mean=mean,
                 std=std,
                 samples=samples,
                 weights=weights
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
    robust_mean = np.median(log_data)
    robust_std = median_abs_deviation(log_data, scale='normal')
    return data[abs(log_data - robust_mean) < level*robust_std]

def combine_lenses(likelihood_type, z_lens, z_src, true_Om0, samples_save_path=None, corner_save_path=None, n_run=100, n_burn=400, n_walkers=10, **posterior_parameters):
    """Combine lenses in the D_dt space

    Parameters
    ----------
    true_Om0 : float
        true Om0, not inferred
    likelihood_type : str
        'DdtGaussian', 'DdtLogNorm', 'DdtHistKDE' supported. 'DdtGaussian' must have 'ddt_mean', 'ddt_sigma'. 'DdtLogNorm' must have 'ddt_mu' and 'ddt_sigma'. 'DdtHistKDE' must have 'lens_ids' and 'samples_dir'.

    """
    n_test = len(z_lens)
    kwargs_posterior_list = []
    if likelihood_type in ['DdtLogNorm', 'DdtGaussian']:
        for i in range(n_test):
            kwargs_posterior = {'z_lens': z_lens[i], 'z_source': z_src[i],
                               'likelihood_type': likelihood_type}
            for param_name, param_value in posterior_parameters.items():
                kwargs_posterior.update({param_name: param_value[i]})
            kwargs_posterior_list.append(kwargs_posterior)
    elif likelihood_type == 'DdtHistKDE':
        lens_ids = posterior_parameters['lens_ids']
        samples_dir = posterior_parameters['samples_dir']
        binning_method = posterior_parameters['binning_method']
        for i, lens_i in enumerate(lens_ids):
            h0_dict_path = os.path.join(samples_dir, 'D_dt_dict_{:04d}.npy'.format(lens_i))
            h0_dict = np.load(h0_dict_path, allow_pickle=True).item() # TODO: Use context manager to prevent memory overload
            D_dt_samples = h0_dict['D_dt_samples']
            remove = np.isnan(D_dt_samples)
            D_dt_samples = D_dt_samples[~remove]
            #cosmo_converter = CosmoConverter(z_lens[i], z_src[i])
            #D_dt_samples = cosmo_converter.get_D_dt(H0_samples)
            kwargs_posterior = {'z_lens': z_lens[i], 'z_source': z_src[i], 
                                'ddt_samples': D_dt_samples, 'ddt_weights': None,
                               'likelihood_type': 'DdtHist', 'binning_method': binning_method}
            kwargs_posterior_list.append(kwargs_posterior)
    else:
        raise NotImplementedError("This likelihood type is not supported. Please choose from 'DdtGaussian', 'DdtLogNorm', and 'DdtHistKDE'.")

    kwargs_lower_cosmo = {'h0': 50.0}
    kwargs_lower_lens = {}
    kwargs_lower_kin = {}

    kwargs_upper_cosmo = {'h0': 90.0}
    kwargs_upper_lens = {}
    kwargs_upper_kin = {}

    kwargs_fixed_cosmo = {'om': true_Om0}
    kwargs_fixed_lens = {}
    kwargs_fixed_kin = {}

    kwargs_mean_start = {'kwargs_cosmo': {'h0': 70.0},
                         'kwargs_lens': {},
                         'kwargs_kin': {}}

    kwargs_sigma_start = {'kwargs_cosmo': {'h0': 10.0},
                         'kwargs_lens': {},
                         'kwargs_kin': {}}

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
    mcmc_sampler = MCMCSampler(kwargs_likelihood_list=kwargs_posterior_list, 
                               cosmology=cosmology, 
                               kwargs_bounds=kwargs_bounds, 
                               ppn_sampling=False,
                               lambda_mst_sampling=False, 
                               lambda_mst_distribution='NONE', 
                               anisotropy_sampling=False,
                               kappa_ext_sampling=False, 
                               kappa_ext_distribution='NONE',
                               anisotropy_model='NONE', 
                               anisotropy_distribution='NONE', 
                               custom_prior=None, 
                               interpolate_cosmo=True, 
                               num_redshift_interp=100,
                               cosmo_fixed=None)

    mcmc_samples, log_prob_cosmo = mcmc_sampler.mcmc_emcee(n_walkers, n_run, n_burn, kwargs_mean_start, kwargs_sigma_start)
    if samples_save_path is not None:
        np.save(samples_save_path, mcmc_samples)

    if corner_save_path is not None:
        corner.corner(mcmc_samples, show_titles=True, labels=mcmc_sampler.param_names(latex_style=True))
        plt.show()
        plt.savefig(corner_save_path)
        plt.close()

    return mcmc_samples, log_prob_cosmo
