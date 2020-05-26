import numpy as np
import mpmath as mp
from astropy.cosmology import FlatLambdaCDM
import baobab.sim_utils.kinematics_utils as kinematics_utils
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.PointSource.point_source import PointSource
from lenstronomy.Analysis.td_cosmography import TDCosmography
from h0rton.h0_inference import h0_utils

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
    required_params = ["lens_mass_center_x", "src_light_center_x","lens_mass_center_y", "src_light_center_y", "lens_mass_gamma", "lens_mass_theta_E", "lens_mass_e1", "lens_mass_e2", "external_shear_gamma1", "external_shear_gamma2", "lens_light_R_sersic", "src_light_R_sersic"]
    def __init__(self, H0_prior, kappa_ext_prior, aniso_param_prior, exclude_vel_disp, kwargs_model, baobab_time_delays, Om0, define_src_pos_wrt_lens, kinematics=None):
        """

        Parameters
        ----------
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
        self.define_src_pos_wrt_lens = define_src_pos_wrt_lens
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

    def set_cosmology_observables(self, z_lens, z_src, measured_vd, measured_vd_err, measured_td, measured_td_err, abcd_ordering_i, true_img_dec, true_img_ra):
        """Set the cosmology observables for a given lens system, persistent across all the samples for that system

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
        lens_light_R_sersic : float
            effective radius of lens light in arcsec
        measured_td : np.array of shape `[n_images,]`
            the measured time delays in days 
        measured_td_err : float
            the time delay measurement error in days
        abcd_ordering_i : np.array of shape `[n_images,]`
            the image ordering followed by `measured_td` in increasing dec. Example: if the `measured_td` are [a, b, c, d] and the corresponding image dec are [0.3, -0.1, 0.8, 0.4], then `abcd_ordering_i` are [1, 0, 3, 2].
        true_img_dec : np.array of shape `[n_images, ]`
            dec of the true image positions in arcsec
        true_img_ra : np.array of shape `[n_images, ]`
            ra of the true image positions in arcsec

        """
        self.z_lens = z_lens
        self.z_src = z_src
        self.measured_vd = measured_vd
        self.measured_vd_err = measured_vd_err
        self.measured_td = np.array(measured_td)
        self.measured_td_err = np.array(measured_td_err)
        self.abcd_ordering_i = abcd_ordering_i
        self.true_img_dec = true_img_dec
        self.true_img_ra = true_img_ra
        self._reorder_measured_td_to_tdlmc()
        # Number of AGN images
        self.n_img = len(measured_td)

    def _reorder_measured_td_to_tdlmc(self):
        """Reorder the measured time delays (same for all lens model samples)

        """
        #print(self.measured_td, self.true_img_dec, self.abcd_ordering_i)
        reordered_measured_td = h0_utils.reorder_to_tdlmc(self.measured_td, np.argsort(self.true_img_dec), self.abcd_ordering_i)
        # Measured time in days (offset from the image with the smallest dec)
        self.measured_td_wrt0 = reordered_measured_td[1:] - reordered_measured_td[0]

    def format_lens_model(self, sample):
        """Set the lens model parameters for a given lens mass model

        Parameters
        ----------
        sample : dict
            a sampled set of lens model parameters

        """
        # Lens mass
        # FIXME: hardcoded for SPEMD
        kwargs_spemd = {'theta_E': sample['lens_mass_theta_E'],
                        'center_x': sample['lens_mass_center_x'], 
                        'center_y': sample['lens_mass_center_y'],
                        'e1': sample['lens_mass_e1'], 
                        'e2': sample['lens_mass_e2'], 
                        'gamma': sample['lens_mass_gamma'],}
        # External shear
        kwargs_shear = {'gamma1': sample['external_shear_gamma1'],
                        'gamma2': sample['external_shear_gamma2'],
                        'ra_0': sample['lens_mass_center_x'],
                        'dec_0': sample['lens_mass_center_y']}
        # AGN point source
        if self.define_src_pos_wrt_lens:
            kwargs_ps = {
            'ra_source': sample['src_light_center_x'] + sample['lens_mass_center_x'],
            'dec_source': sample['src_light_center_y'] + sample['lens_mass_center_y'],
            }
        else:
            kwargs_ps = {
            'ra_source': sample['src_light_center_x'],
            'dec_source': sample['src_light_center_y'],
            }
        kwargs_lens = [kwargs_spemd, kwargs_shear]
        # Raytrace to get point source kwargs in image plane
        kwargs_img, requires_reordering = self.get_img_pos(kwargs_ps, kwargs_lens)
        # Pre-store for reordering image arrays
        dec_image = kwargs_img[0]['dec_image']
        increasing_dec_i = np.argsort(dec_image)

        formatted_lens_model = dict(
                          # TODO: key checking depending on kwargs_model
                          kwargs_lens=kwargs_lens,
                          kwargs_img=kwargs_img,
                          requires_reordering=requires_reordering,
                          increasing_dec_i=increasing_dec_i,
                          lens_light_R_sersic=sample['lens_light_R_sersic'],
                          )
        return formatted_lens_model

    def get_img_pos(self, ps_dict, kwargs_lens):
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
            ra_image, dec_image = ps_class.image_position(kwargs_ps_source, kwargs_lens)
            kwargs_image = [dict(ra_image=ra_image[0],
                                   dec_image=dec_image[0])]
            requires_reordering = True # Since the ra_image is coming out of lenstronomy, we need to reorder it to agree with TDLMC
        else:
            kwargs_image = [ps_dict]
            requires_reordering = False # If the user is providing `ra_image` inside `ps_dict`, the order is required to agree with `measured_time_delays`.
        return kwargs_image, requires_reordering

    def sample_H0(self, random_state):
        return self.H0_prior.rvs(random_state=random_state)

    def sample_kappa_ext(self, random_state):
        return self.kappa_ext_prior.rvs(random_state=random_state)

    def sample_aniso_param(self, random_state):
        return self.aniso_param_prior.rvs(random_state=random_state)

    def calculate_offset_from_true_image_positions(self, model_ra, model_dec, true_img_ra, true_img_dec,  increasing_dec_i, abcd_ordering_i):
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
        if self.requires_reordering:
            model_ra = h0_utils.reorder_to_tdlmc(model_ra, increasing_dec_i, abcd_ordering_i)
            model_dec = h0_utils.reorder_to_tdlmc(model_dec, increasing_dec_i, abcd_ordering_i)
        
        ra_offset = model_ra - true_img_ra
        dec_offset = model_dec - true_img_dec

        ra_offset = ra_offset.reshape(-1, 1)
        dec_offset = dec_offset.reshape(-1, 1)

        offset = np.concatenate([ra_offset, dec_offset], axis=1)
        return np.linalg.norm(offset, axis=1)

    def get_h0_sample(self, sampled_lens_model_raw, random_state):
        """Get MC samples from the H0Posterior

        Parameters
        ----------
        sampled_lens_model_raw : dict
            sampled lens model parameters, pre-formatting
        random_state : np.random.RandomState object

        Returns
        -------
        tuple of floats
            the candidate H0 and its weight

        """
        # Samples from the lens posterior are reinterpreted as samples from the lens model prior in the H0 inference stage
        lens_prior_sample = self.format_lens_model(sampled_lens_model_raw)
        kwargs_lens = lens_prior_sample['kwargs_lens']
        lens_light_R_sersic = lens_prior_sample['lens_light_R_sersic']
        increasing_dec_i = lens_prior_sample['increasing_dec_i']
        # Sample from respective predefined priors
        h0_candidate = self.sample_H0(random_state)
        k_ext = self.sample_kappa_ext(random_state)
        # Define cosmology
        cosmo = FlatLambdaCDM(H0=h0_candidate, Om0=self.Om0)
        # Tool for getting time delays and velocity dispersions
        td_cosmo = TDCosmography(self.z_lens, self.z_src, self.kwargs_model, cosmo_fiducial=cosmo)
        # Velocity dispersion
        # TODO: separate sampling function if vel_disp is excluded
        if self.exclude_vel_disp:
            ll_vd = 0.0
        else:
            aniso_param = self.sample_aniso_param(random_state)
            inferred_vd = self.get_velocity_dispersion(
                                                       td_cosmo, 
                                                       kwargs_lens, 
                                                       None, #FIXME: only analytic
                                                       {'aniso_param': aniso_param}, 
                                                       self.kinematics.kwargs_aperture, 
                                                       self.kinematics.kwargs_psf, 
                                                       self.kinematics.anisotropy_model, 
                                                       lens_light_R_sersic,
                                                       self.kinematics.kwargs_numerics,
                                                       k_ext
                                                       )
            ll_vd = gaussian_ll_pdf(inferred_vd, self.measured_vd, self.measured_vd_err)
        # Time delays
        inferred_td, x_image, y_image = td_cosmo.time_delays(kwargs_lens, lens_prior_sample['kwargs_img'], kappa_ext=k_ext)
        if lens_prior_sample['requires_reordering']:
            inferred_td = h0_utils.reorder_to_tdlmc(inferred_td, increasing_dec_i, self.abcd_ordering_i)
        else:
            inferred_td = np.array(inferred_td)
        inferred_td_wrt0 = inferred_td[1:] - inferred_td[0]
        #print(inferred_td, self.measured_td)
        ll_td = np.sum(gaussian_ll_pdf(inferred_td_wrt0, self.measured_td_wrt0, self.measured_td_err))
        log_w = ll_vd + ll_td
        weight = mp.exp(log_w)
        return h0_candidate, weight 