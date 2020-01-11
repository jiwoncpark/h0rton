import numpy as np
import mpmath as mp
from astropy.cosmology import FlatLambdaCDM
import baobab.sim_utils.kinematics_utils as kinematics_utils
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.PointSource.point_source import PointSource
from lenstronomy.Analysis.td_cosmography import TDCosmography

__all__ = ['gaussian_ll_pdf', 'H0Posterior']

def gaussian_ll_pdf(x, mu, sigma):
    """Evaluates the log of the normal PDF at point x
    
    Parameters
    ----------
    x : float or array-like
        point at which to evaluate the log pdf
    mu : float or array-like
        mean of the normal on a linear scale
    sigma : float or array-like
        standard deviation of the normal on a linear scale
        
    """
    log_pdf = -0.5*(x - mu)**2.0/sigma**2.0 - np.log(sigma) - 0.5*np.log(2.0*np.pi)
    return log_pdf

class H0Posterior:
    """Represents the posterior over H0

    """
    required_params = ['lens_mass_gamma', 'lens_mass_theta_E', 'lens_mass_e1', 'lens_mass_e2', 'external_shear_gamma_ext', 'external_shear_psi_ext', 'lens_light_R_sersic', 'src_light_center_x', 'src_light_center_y',]
    def __init__(self, H0_prior, kappa_ext_prior, aniso_param_prior, exclude_vel_disp, kwargs_model, baobab_time_delays, Om0, kinematics=None):
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
        exclude_vel_disp : bool
            whether to exclude the velocity dispersion likelihood. Default: False
        
        """
        self.H0_prior = H0_prior
        self.kappa_ext_prior = kappa_ext_prior
        self.aniso_param_prior = aniso_param_prior
        self.exclude_vel_disp = exclude_vel_disp
        self.kwargs_model = kwargs_model
        self.baobab_time_delays = baobab_time_delays
        self.kinematics = kinematics
        self.Om0 = Om0 # Omega matter
        # Always define point source in terms of `LENSED_POSITION` for speed
        self.kwargs_model.update(dict(point_source_model_list=['LENSED_POSITION']))

        if self.kinematics.anisotropy_model == 'analytic':
            self.get_velocity_dispersion = getattr(kinematics_utils, 'velocity_dispersion_analytic')
        else:
            # TODO: currently not available, as BNN does not predict lens light profile
            self.get_velocity_disperison = getattr(kinematics_utils, 'velocity_dispersion_numerical')

        if not self.exclude_vel_disp:
            if self.kinematics is None:
                raise ValueError("kinematics is required to calculate velocity dispersion.")

    @classmethod
    def from_dict(cls, lens_dict):
        """Initialize H0Posterior from a dictionary

        Parameters
        ----------
        lens_dict : dict
            contains properties required to initialize H0Posterior. See `__init__` method above for the required parameters and their formats.

        """
        return cls(lens_dict.items())

    def set_cosmology_observables(self, z_lens, z_src, measured_vd, measured_vd_err, measured_td, measured_td_err, abcd_ordering_i):
        """Set the cosmology observables for a given lens system

        Parameters
        ----------
        measured_td : np.array of shape `[n_images - 1,]`
            the measured time delays in days (offset from the image with the smallest dec)
        measured_td_err : float
            the time delay measurement error in days
        abcd_ordering_i : np.array of shape `[n_images,]`
            the image ordering followed by `measured_td` in increasing dec. Example: if the `measured_td` are [a, b, c, d] and the corresponding image dec are [0.3, -0.1, 0.8, 0.4], then `abcd_ordering_i` are [1, 0, 3, 2].

        """
        self.z_lens = z_lens
        self.z_src = z_src
        self.measured_vd = measured_vd
        self.measured_vd_err = measured_vd_err
        self.measured_td = np.array(measured_td)
        self.measured_td_err = np.array(measured_td_err)
        self.abcd_ordering_i = abcd_ordering_i
        # Number of AGN images
        self.n_img = len(measured_td) + 1

    def set_lens_model(self, bnn_sample):
        """Set the lens model parameters for a given lens mass model

        """
        # Lens mass
        # FIXME: hardcoded for SPEMD
        kwargs_spemd = {'theta_E': bnn_sample['lens_mass_theta_E'],
                        'center_x': 0, 
                        'center_y': 0,
                        'e1': bnn_sample['lens_mass_e1'], 
                        'e2': bnn_sample['lens_mass_e2'], 
                        'gamma': bnn_sample['lens_mass_gamma'],}
        # External shear
        kwargs_shear = {'gamma_ext': bnn_sample['external_shear_gamma_ext'],
                        'psi_ext': bnn_sample['external_shear_psi_ext'],
                        'ra_0': 0.0,
                        'dec_0': 0.0}
        # AGN point source
        kwargs_ps = {'ra_source': bnn_sample['src_light_center_x'],
                      'dec_source': bnn_sample['src_light_center_y'],}
        
        self.lens_light_R_sersic = bnn_sample['lens_light_R_sersic']
        # TODO: key checking depending on kwargs_model
        self.kwargs_lens = [kwargs_spemd, kwargs_shear]
        self.set_kwargs_ps(kwargs_ps)
        # Pre-store for reordering image arrays
        dec_image = self.kwargs_ps[0]['dec_image']
        self.increasing_dec_i = np.argsort(dec_image)

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
                                   dec_image=dec_image[0])]
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

    def get_sample(self):
        """Get samples from the H0Posterior

        Parameters
        ----------
        n_samples : int

        Returns
        -------
        tuple of floats
            the candidate H0 and its weight

        """
        h0_candidate = self.sample_H0()
        k_ext = self.sample_kappa_ext()
        aniso_param = self.sample_aniso_param()
        # Define cosmology
        cosmo = FlatLambdaCDM(H0=h0_candidate, Om0=self.Om0)
        # Tool for getting time delays and velocity dispersions
        td_cosmo = TDCosmography(self.z_lens, self.z_src, self.kwargs_model, cosmo_fiducial=cosmo)
        # Velocity dispersion
        if self.exclude_vel_disp:
            ll_vd = 0.0
        else:
            inferred_vd = self.get_velocity_dispersion(
                                                       td_cosmo, 
                                                       self.kwargs_lens, 
                                                       None, #FIXME: only analytic
                                                       aniso_param*self.lens_light_R_sersic, 
                                                       self.kinematics.kwargs_aperture, 
                                                       self.kinematics.kwargs_psf, 
                                                       self.kinematics.anisotropy_model, 
                                                       self.lens_light_R_sersic,
                                                       self.kinematics.kwargs_numerics,
                                                       k_ext
                                                       )
            ll_vd = gaussian_ll_pdf(inferred_vd, self.measured_vd, self.measured_vd_err)
        # Time delays
        inferred_td = td_cosmo.time_delays(self.kwargs_lens, self.kwargs_ps, kappa_ext=k_ext)
        if self.requires_reordering:
            inferred_td = self.reorder_to_tdlmc(inferred_td)
        else:
            inferred_td = np.array(inferred_td)
        inferred_td = inferred_td[1:] - inferred_td[0]
        ll_td = np.sum(gaussian_ll_pdf(inferred_td, self.measured_td, self.measured_td_err))
        log_w = ll_vd + ll_td
        weight = mp.exp(log_w)
        return h0_candidate, weight 








