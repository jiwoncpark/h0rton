import numpy as np
import mpmath as mp
from astropy.cosmology import FlatLambdaCDM
import baobab.sim_utils.kinematics_utils as kinematics_utils
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.PointSource.point_source import PointSource
from lenstronomy.Analysis.td_cosmography import TDCosmography
from scipy.stats import norm
from h0rton.h0_inference import h0_utils

__all__ = ['H0Posterior']

class H0Posterior:
    """Represents the posterior over H0

    """
    required_params = ["lens_mass_center_x", "src_light_center_x","lens_mass_center_y", "src_light_center_y", "lens_mass_gamma", "lens_mass_theta_E", "lens_mass_e1", "lens_mass_e2", "external_shear_gamma1", "external_shear_gamma2", "src_light_R_sersic"]
    def __init__(self, H0_prior, kappa_ext_prior,  kwargs_model, baobab_time_delays, Om0, define_src_pos_wrt_lens, exclude_vel_disp=True, aniso_param_prior=None, kinematics=None, kappa_transformed=True, kwargs_lens_eqn_solver={}):
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
        self.kappa_transformed = kappa_transformed
        self.kappa_ext_prior = kappa_ext_prior
        if self.kappa_transformed:
            self.sample_kappa_ext = self.sample_kappa_ext_transformed
        else:
            self.sample_kappa_ext = self.sample_kappa_ext_original
        self.aniso_param_prior = aniso_param_prior
        self.exclude_vel_disp = exclude_vel_disp
        self.kwargs_model = kwargs_model
        self.baobab_time_delays = baobab_time_delays
        self.define_src_pos_wrt_lens = define_src_pos_wrt_lens
        self.kinematics = kinematics
        self.Om0 = Om0 # Omega matter
        self.kwargs_lens_eqn_solver = kwargs_lens_eqn_solver
        self.kwargs_model.update(dict(point_source_model_list=['SOURCE_POSITION']))
        if not self.exclude_vel_disp:
            if self.kinematics is None:
                raise ValueError("kinematics is required to calculate velocity dispersion.")
            if self.kinematics.anisotropy_model == 'analytic':
                self.get_velocity_dispersion = getattr(kinematics_utils, 'velocity_dispersion_analytic')
            else:
                # TODO: currently not available, as BNN does not predict lens light profile
                self.get_velocity_disperison = getattr(kinematics_utils, 'velocity_dispersion_numerical')

    @classmethod
    def from_dict(cls, lens_dict):
        """Initialize H0Posterior from a dictionary

        Parameters
        ----------
        lens_dict : dict
            contains properties required to initialize H0Posterior. See `__init__` method above for the required parameters and their formats.

        """
        return cls(lens_dict.items())

    def set_cosmology_observables(self, z_lens, z_src, measured_td_wrt0, measured_td_err, abcd_ordering_i, true_img_dec, true_img_ra, kappa_ext, measured_vd=None, measured_vd_err=None):
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
        self.measured_td_wrt0 = np.array(measured_td_wrt0)
        self.measured_td_err = np.array(measured_td_err)
        self.abcd_ordering_i = abcd_ordering_i
        self.true_img_dec = true_img_dec
        self.true_img_ra = true_img_ra
        self.kappa_ext = kappa_ext
        #self._reorder_measured_td_to_tdlmc()
        # Number of AGN images
        self.n_img = len(measured_td_wrt0) + 1

    def _reorder_measured_td_to_tdlmc(self):
        """Reorder the measured time delays (same for all lens model samples)
        
        Note
        ----
        Unused!

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
        #kwargs_img, requires_reordering = self.get_img_pos(kwargs_ps, kwargs_lens)
        # Pre-store for reordering image arrays
        #dec_image = kwargs_img[0]['dec_image']
        #increasing_dec_i = np.argsort(dec_image)

        formatted_lens_model = dict(
                          # TODO: key checking depending on kwargs_model
                          kwargs_lens=kwargs_lens,
                          kwargs_ps=[kwargs_ps],
                          #kwargs_img=kwargs_img,
                          requires_reordering=True, #FIXME: get edge cases
                          #increasing_dec_i=increasing_dec_i,
                          #lens_light_R_sersic=sample['lens_light_R_sersic'],
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

    def sample_kappa_ext_original(self, random_state):
        return self.kappa_ext_prior.rvs(random_state=random_state)

    def sample_kappa_ext_transformed(self, random_state):
        x = self.kappa_ext_prior.rvs(random_state=random_state)
        i = 0
        while ~np.isfinite(1.0 - 1.0/x):
            x = self.kappa_ext_prior.rvs(random_state=random_state + i)
            i += 1
        return 1.0 - 1.0/x

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
        #lens_light_R_sersic = lens_prior_sample['lens_light_R_sersic']
        #increasing_dec_i = lens_prior_sample['increasing_dec_i']
        # Sample from respective predefined priors
        h0_candidate = self.sample_H0(random_state)
        k_ext = self.sample_kappa_ext(random_state)
        # Define cosmology
        cosmo = FlatLambdaCDM(H0=h0_candidate, Om0=self.Om0)
        # Tool for getting time delays and velocity dispersions
        td_cosmo = TDCosmography(self.z_lens, self.z_src, self.kwargs_model, cosmo_fiducial=cosmo, kwargs_lens_eqn_solver=self.kwargs_lens_eqn_solver)
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
                                                       None,
                                                       self.kinematics.kwargs_numerics,
                                                       k_ext
                                                       )
            ll_vd = h0_utils.gaussian_ll_pdf(inferred_vd, self.measured_vd, self.measured_vd_err)
        # Time delays
        inferred_td, x_image, y_image = td_cosmo.time_delays(kwargs_lens, lens_prior_sample['kwargs_ps'], kappa_ext=k_ext)
        if len(inferred_td) > len(self.measured_td_wrt0) + 1:
            inferred_td, x_image, y_image = self.chuck_images(inferred_td, x_image, y_image)
        if lens_prior_sample['requires_reordering']:
            increasing_dec_i = np.argsort(y_image)
            inferred_td = h0_utils.reorder_to_tdlmc(inferred_td, increasing_dec_i, self.abcd_ordering_i)
        else:
            inferred_td = np.array(inferred_td)
        inferred_td_wrt0 = inferred_td[1:] - inferred_td[0]
        #print(inferred_td, self.measured_td)
        ll_td = np.sum(h0_utils.gaussian_ll_pdf(inferred_td_wrt0, self.measured_td_wrt0, self.measured_td_err))
        log_w = ll_vd + ll_td
        weight = mp.exp(log_w)
        return h0_candidate, weight

    def set_truth_lens_model(self, sampled_lens_model_raw):
        # Set once per lens
        # Samples from the lens posterior are reinterpreted as samples from the lens model prior in the H0 inference stage
        self.kwargs_model.update(dict(point_source_model_list=['SOURCE_POSITION']))
        self.lens_prior_sample = self.format_lens_model(sampled_lens_model_raw)
        cosmo = FlatLambdaCDM(H0=70.0, Om0=self.Om0) # fiducial cosmology, doesn't matter
        td_cosmo = TDCosmography(self.z_lens, self.z_src, self.kwargs_model, cosmo_fiducial=cosmo, kwargs_lens_eqn_solver=self.kwargs_lens_eqn_solver)
        _, x_image, y_image = td_cosmo.time_delays(self.lens_prior_sample['kwargs_lens'], self.lens_prior_sample['kwargs_ps'], kappa_ext=0.0)
        while len(y_image) not in [2, 4]:
            _, x_image, y_image = td_cosmo.time_delays(self.lens_prior_sample['kwargs_lens'], self.lens_prior_sample['kwargs_ps'], kappa_ext=0.0)
            #raise ValueError("Defective lens?")
        self.kwargs_model.update(dict(point_source_model_list=['LENSED_POSITION']))
        self.kwargs_image = [dict(ra_image=x_image, dec_image=y_image)]

    def get_h0_sample_truth(self, random_state):
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
        #increasing_dec_i = lens_prior_sample['increasing_dec_i']
        # Sample from respective predefined priors
        h0_candidate = self.sample_H0(random_state)
        k_ext = self.sample_kappa_ext_transformed(random_state)#self.kappa_ext #self.sample_kappa_ext(random_state) #
        # Define cosmology
        cosmo = FlatLambdaCDM(H0=h0_candidate, Om0=self.Om0)
        # Tool for getting time delays and velocity dispersions
        td_cosmo = TDCosmography(self.z_lens, self.z_src, self.kwargs_model, cosmo_fiducial=cosmo, kwargs_lens_eqn_solver=self.kwargs_lens_eqn_solver)
        # Velocity dispersion
        # TODO: separate sampling function if vel_disp is excluded
        # Time delays
        inferred_td, x_image, y_image = td_cosmo.time_delays(self.lens_prior_sample['kwargs_lens'], self.kwargs_image, kappa_ext=k_ext)
        #print(inferred_td, y_image)
        if len(inferred_td) > len(self.measured_td_wrt0) + 1:
            inferred_td, x_image, y_image = self.chuck_images(inferred_td, x_image, y_image)
            #print("after correct: ", inferred_td, y_image)
        if self.lens_prior_sample['requires_reordering']:
            increasing_dec_i = np.argsort(y_image)
            inferred_td = h0_utils.reorder_to_tdlmc(inferred_td, increasing_dec_i, self.abcd_ordering_i)
        else:
            inferred_td = np.array(inferred_td)
        inferred_td_wrt0 = inferred_td[1:] - inferred_td[0]
        #print(inferred_td_wrt0, self.measured_td_wrt0)
        ll_td = np.sum(h0_utils.gaussian_ll_pdf(inferred_td_wrt0, self.measured_td_wrt0, self.measured_td_err))
        log_w = ll_td
        weight = mp.exp(log_w)
        return h0_candidate, weight

    def chuck_images(self, inferred_td, x_image, y_image):
        """If the number of predicted images are greater than the measured, choose the images that best correspond to the measured.

        """
        # Find index or indices that must be removed.
        # Candidates 4 choose 2 (=6) or 3 choose 2 (=3)
        keep_idx = np.zeros(len(self.true_img_dec))
        for i, actual_y_i in enumerate(self.true_img_dec): # FIXME: use measured_img_dec
            keep_idx[i] = np.argmin((y_image - actual_y_i)**2.0)
        keep_idx = np.sort(keep_idx).astype(int)
        inferred_td = inferred_td[keep_idx]
        x_image = x_image[keep_idx]
        y_image = y_image[keep_idx]
        return inferred_td, x_image, y_image