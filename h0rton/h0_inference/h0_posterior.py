import random
import warnings
import numpy as np
import mpmath as mp
from tqdm import tqdm
from astropy.cosmology import FlatLambdaCDM
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.PointSource.point_source import PointSource
from lenstronomy.Analysis.lens_properties import LensProp
import lenstronomy.Util.param_util as param_util
import h0rton.tdlmc_utils

__all__ = ['log_normal_pdf', 'H0Posterior']

def log_normal_pdf(x, mu, sigma):
    """Evaluates the log of the normal (not lognormal) PDF at point x
    
    Parameters
    ----------
    x : float or array-like
        point at which to evaluate the log pdf
    mu : float or array-like
        mean of the normal
    sigma : float or array-like
        standard deviation of the normal
        
    """
    log_pdf = -0.5*(x - mu)**2.0/sigma**2.0 - np.log(sigma) - 0.5*np.log(2.0*np.pi)
    return log_pdf

class H0Posterior:
    """Represents the posterior over H0

    """
    def __init__(self, z_lens, z_src, lens_mass_dict, ext_shear_dict, ps_dict, measured_vd, measured_vd_err, measured_td, measured_td_err, lens_light_R_sersic, H0_prior, kappa_ext_prior, aniso_param_prior, kwargs_model, abcd_ordering_i):
        """

        Parameters
        ----------
        z_lens : float
        z_src : float
        lens_mass_dict : dict
            dict of lens mass kwargs
        ext_shear_dict : dict
            dict of external shear kwargs
        ps_dict : dict
            dict of point source kwargs
        measured_vd : float
            measured velocity dispersion
        measured_vd_err : float
            measurement error of velocity dispersion
        measured_td : array-like
            measured time delays
        measured_td_err : array-like
            measurement error of time delays
        lens_light_R_sersic : float
            effective radius of lens light in arcsec
        H0_prior : scipy rv_continuous object
        kappa_ext_prior : scipy rv_continuous object
            prior over the external convergence
        aniso_param_prior : scipy rv_continuous object
            prior over the anisotropy radius, r_ani
        kwargs_model : dict
            dictionary defining which models (parameterizations) are used to define `lens_mass_dict`, `ext_shear_dict`, `ps_dict`
        abcd_ordering_i : list
            ABCD in an increasing dec order if the keys ABCD mapped to values 0123, respectively, e.g. [3, 1, 0, 2] if D (value 3) is lowest, B (value 1) is second lowest
        
        """
        self.z_lens = z_lens
        self.z_src = z_src
        self.measured_vd = measured_vd
        self.measured_vd_err = measured_vd_err
        self.measured_td = np.array(measured_td)
        self.measured_td_err = np.array(measured_td_err)
        self.lens_light_R_sersic = lens_light_R_sersic
        self.H0_prior = H0_prior
        self.kappa_ext_prior = kappa_ext_prior
        self.aniso_param_prior = aniso_param_prior
        self.kwargs_model = kwargs_model
        self.abcd_ordering_i = abcd_ordering_i

        self.R_slit = 1.0 # arcsec
        self.dR_slit = 1.0 # arcsec
        self.psf_fwhm = 0.6 # arcsec
        self.Om0 = 0.27 # Omega matter
        self.Ob0 = 0.0 # Omega baryons
        # Number of AGN images
        self.n_img = len(measured_td)
        # TODO: key checking depending on kwargs_model
        self.kwargs_lens = [lens_mass_dict, ext_shear_dict]
        self.set_kwargs_ps(ps_dict)
        # Always define point source in terms of `LENSED_POSITION` for speed
        self.kwargs_model.update(dict(point_source_model_list=['LENSED_POSITION']))

        # Pre-store for reordering image arrays
        dec_image = self.kwargs_ps[0]['dec_image']
        self.increasing_dec_i = np.argsort(dec_image)

    @classmethod
    def from_dict(cls, lens_dict):
        """Initialize H0Posterior from a dictionary

        Parameters
        ----------
        lens_dict : dict
            contains properties required to initialize H0Posterior. See `__init__` method above for the required parameters and their formats.

        """
        return cls(lens_dict.items())

    def set_kwargs_ps(self, ps_dict):
        """Sets the kwargs_ps class attribute as those coresponding to the point source model `LENSED_POSITION`

        Parameters
        ----------
        ps_dict : dict
            point source parameters definitions, either of `SOURCE_POSITION` or `LENSED_POSITION`

        """
        if 'ra_source' in ps_dict:
            # If the user provided ps_dict in terms of the source position, we precompute the corresponding image positions before we enter the sampling loop.
            lens_model_class = LensModel(self.kwargs_model['lens_model_list'])
            ps_class = PointSource(['SOURCE_POSITION'], lens_model_class)
            kwargs_ps_source = [ps_dict]
            ra_image, dec_image = ps_class.image_position(kwargs_ps_source, self.kwargs_lens)
            self.kwargs_ps = [dict(ra_image=ra_image[0],
                                   dec_image=dec_image[0],
                                   point_amp=ps_dict['point_amp'])]
            self.requires_reordering = True # Since the ra_image is coming out of lenstronomy, we need to reorder it to agree with TDLMC
        else:
            self.kwargs_ps = [ps_dict]
            self.requires_reordering = False # If the user is providing `ra_image` inside `ps_dict`, the order is required to agree with `measured_time_delays`.

    def sample_H0(self):
        return self.H0_prior.rvs()

    def sample_kappa_ext(self):
        return self.kappa_ext_prior.rvs()

    def sample_aniso_param(self):
        return self.aniso_param_prior.rvs()

    def calculate_offset_from_true_image_positions(self, true_img_ra, true_img_dec):
        """Calculates the difference in arcsec between the (inferred or fed-in) image positions known to `H0Posterior` and the provided true image positions

        Parameters
        ----------
        true_img_ra : array-like, of length self.n_img
            ra of true image positions in TDLMC order
        true_img_dec : array-like, of length self.n_img
            dec of true image positions in TDLMC order
        
        Returns
        -------
        array-like
            offset in arcsec for each image

        """
        model_ra = self.kwargs_ps[0]['ra_image']
        model_dec = self.kwargs_ps[0]['dec_image']
        if self.requires_reordering:
            model_ra = self.reorder_to_tdlmc(model_ra)
            model_dec = self.reorder_to_tdlmc(model_dec)
        
        ra_offset = model_ra - true_img_ra
        dec_offset = model_dec - true_img_dec

        ra_offset = ra_offset.reshape(-1, 1)
        dec_offset = dec_offset.reshape(-1, 1)

        offset = np.concatenate([ra_offset, dec_offset], axis=1)
        return np.linalg.norm(offset, axis=1)

    def reorder_to_tdlmc(self, img_array):
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
        img_array = np.array(img_array)[self.increasing_dec_i][self.abcd_ordering_i]
        return img_array

    def get_samples(self, n_samples, seed, exclude_vel_disp=False):
        """Get samples from the H0Posterior

        Parameters
        ----------
        n_samples : int
        seed : int
            random seed to use
        exclude_vel_disp : bool
            whether to exclude the velocity dispersion likelihood. Default: False

        Returns
        -------
        dict
            keys are `H0_samples` (array of shape (n_samples,)), `H0_weights` (array of shape (n_samples,)), `inferred_vd` (inferred velocity dispersions, array of shape (n_samples,)), `ll_vd` (log likelihood of the inferred velocity dispersions, array of shape (n_samples,)), `inferred_td` (inferred time delays, array of shape (n_samples, n_images)), `ll_td` (log likelihood of the inferred time delays, array of shape (n_samples, n_images))

        """
        # Seed for reproducibility
        random.seed(seed)
        np.random.seed(seed)
        # Initialize output dict
        samples_dict = dict(
                            H0_samples=np.zeros(n_samples),
                            H0_weights=np.zeros(n_samples),
                            inferred_vd=np.zeros(n_samples),
                            ll_vd=np.zeros(n_samples),
                            inferred_td=np.zeros((n_samples, self.n_img)),
                            ll_td=np.zeros((n_samples, self.n_img)),
                            )

        for n in tqdm(range(n_samples)):
            H0_candidate = self.sample_H0()
            k_ext = self.sample_kappa_ext()
            aniso_param = self.sample_aniso_param()
            # Define cosmology
            cosmo = FlatLambdaCDM(H0=H0_candidate, Om0=self.Om0, Ob0=self.Ob0)
            # Tool for getting time delays and velocity dispersions
            lens_prop = LensProp(self.z_lens, self.z_src, self.kwargs_model, cosmo=cosmo)
            # Velocity dispersion
            if exclude_vel_disp:
                ll_vd = 0.0
            else:
                inferred_vd = lens_prop.velocity_dispersion(self.kwargs_lens, r_eff=self.lens_light_R_sersic, R_slit=self.R_slit, dR_slit=self.dR_slit, psf_fwhm=self.psf_fwhm, aniso_param=aniso_param, num_evaluate=5000, kappa_ext=k_ext)
                ll_vd = log_normal_pdf(inferred_vd,
                                       self.measured_vd,
                                       self.measured_vd_err)
            # Time delays
            inferred_td = lens_prop.time_delays(self.kwargs_lens, self.kwargs_ps, kappa_ext=k_ext)
            if self.requires_reordering:
                inferred_td = self.reorder_to_tdlmc(inferred_td)
            else:
                inferred_td = np.array(inferred_td)
            inferred_td = inferred_td[1:] - inferred_td[0]
            ll_td = np.sum(log_normal_pdf(inferred_td, self.measured_td, self.measured_td_err))
            log_w = ll_vd + ll_td
            w = mp.exp(log_w)

            if n == 0 and np.isnan(float(w)):
                warnings.warn("Weight is nan. Double check `abcd_ordering`.")

            samples_dict['H0_samples'][n] = H0_candidate
            samples_dict['H0_weights'][n] = w
            if not exclude_vel_disp:
                samples_dict['inferred_vd'][n] = inferred_vd
                samples_dict['ll_vd'][n] = ll_vd
            samples_dict['inferred_td'][n, :] = inferred_td
            samples_dict['ll_td'][n] = ll_td

        # Normalize weights to unity
        samples_dict['H0_weights'] /= np.sum(samples_dict['H0_weights'])

        return samples_dict








