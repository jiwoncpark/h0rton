"""Script to run MCMC cosmological sampling for individual lenses, using the BNN posterior

It borrows heavily from the `catalogue modelling.ipynb` notebook in Lenstronomy Extensions, which you can find `here <https://github.com/sibirrer/lenstronomy_extensions/blob/master/lenstronomy_extensions/Notebooks/catalogue%20modelling.ipynb>`_.

Example
-------
To run this script, pass in the path to the user-defined inference config file as the argument::
    
    $ python h0rton/infer_h0_mcmc_default.py mcmc_default.json

"""

import numpy as np
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LensModel.Solver.lens_equation_solver import LensEquationSolver
from lenstronomy.Workflow.fitting_sequence import FittingSequence
import lenstronomy.Util.util as util
from baobab.sim_utils.psf_utils import get_PSF_model
from baobab.data_augmentation import noise_lenstronomy
__all__ = ["ForwardModelingPosterior"]


class ForwardModelingPosterior():
    """Joint posterior over the model parameters of lens mass, lens light, source light, and AGN point source
    as well as D_dt (H0)

    """
    def __init__(self, kwargs_lens_eqn_solver=None, astrometric_sigma=0.005, supersampling_factor=1):
        """Configure reasonable MCMC parameter kwargs and numerics common to all lenses

        """
        # Define assumed model profiles, with MCMC kwargs
        self.kwargs_model = dict(lens_model_list=['PEMD', 'SHEAR'],
                                point_source_model_list=['LENSED_POSITION'],
                                lens_light_model_list=['SERSIC_ELLIPSE'],
                                source_light_model_list=['SERSIC_ELLIPSE'])
        lens_params = self.get_lens_mass_mcmc_kwargs()
        lens_light_params = self.get_lens_light_mcmc_kwargs()
        source_params = self.get_source_mcmc_kwargs()
        ps_params = self.get_ps_mcmc_kwargs(lens_params[0])
        cosmo_params = self.get_cosmo_mcmc_kwargs()
        self.kwargs_params = {'lens_model': lens_params,
                            'source_model': source_params,
                            'lens_light_model': lens_light_params,
                            'point_source_model': ps_params,
                            'special': cosmo_params}
        # Define MCMC numerics
        self.kwargs_numerics = {'supersampling_factor': supersampling_factor,
                                'supersampling_convolution': False}
        self.num_source_model = len(self.kwargs_model['source_light_model_list'])
        self.kwargs_constraints = dict(
                                       joint_source_with_point_source=[[0, 0]], # AGN and host share centers
                                       joint_lens_with_light=[[0, 0, ['center_x', 'center_y']]], # Lens mass and light share centers (some nonlinear solvers override this assumption)
                                       #joint_lens_with_lens=[[0, 1, ['center_x', 'ra_0'], [0, 1, ['center_y', 'dec_0']]]],
                                       Ddt_sampling=True,
                                       )
        if kwargs_lens_eqn_solver is None:
            kwargs_lens_eqn_solver = {'min_distance': 0.05, 'search_window': 5.12, 'num_iter_max': 200}
        n_img = len(ps_params[0][0]['ra_image'])
        self.kwargs_likelihood = {'check_bounds': True,
                                  'sort_images_by_dec': True,
                                  'prior_lens': [],
                                  'prior_special': [],
                                 'force_no_add_image': False,
                                 'custom_logL_addition': self.couple_lens_mass_light_centroids if n_img == 4 else None, # necessary for quads for which the PROFILE_SHEAR solver ignores the joint_lens_with_light constraint
                                 'source_marg': False,
                                 'image_position_uncertainty': astrometric_sigma,
                                 'check_matched_source_position': True,
                                 'source_position_tolerance': 0.001,
                                 'time_delay_likelihood': True,
                                 'kwargs_lens_eqn_solver': kwargs_lens_eqn_solver,
                                         }

    def couple_lens_mass_light_centroids(self, kwargs_lens, kwargs_source, kwargs_lens_light=None, kwargs_ps=None, kwargs_special=None, kwargs_extinction=None):
        """Force the lens mass and lens light to share centroids by appending to the MCMC objective

        """
        return -(kwargs_lens[0]['center_x'] - kwargs_lens_light[0]['center_x'])**2.0 - (kwargs_lens[0]['center_y'] - kwargs_lens_light[0]['center_y'])**2.0

    def set_kwargs_data_joint(self, image, measured_td, measured_td_sigma, survey_object_dict, eff_exposure_time=5400.0, inverse=False):
        """Feed in time delay and imaging data, different across lenses

        Parameters
        ----------
        image : np.array, of shape [n_pix, n_pix]
        measured_td : np.array, of shape [n_img - 1,]
        measured_td_sigma : float or np.array of shape [n_img - 1,]

        """
        num_pix = image.shape[0]
        for i, (bp, survey_object) in enumerate(survey_object_dict.items()): # FIXME: only single band
            noise_kwargs = survey_object.kwargs_single_band()
            noise_kwargs['exposure_time'] = eff_exposure_time
            psf_kernel_size = survey_object.psf_kernel_size
            which_psf_maps = survey_object.which_psf_maps
        _, _, ra_0, dec_0, _, _, Mpix2coord, _ = util.make_grid_with_coordtransform(numPix=num_pix,
                                                                                    deltapix=noise_kwargs['pixel_scale'],
                                                                                    center_ra=0,
                                                                                    center_dec=0,
                                                                                    subgrid_res=1,
                                                                                    inverse=inverse)
        noise_dict = noise_lenstronomy.get_noise_sigma2_lenstronomy(image, **noise_kwargs)
        pixel_rms = (noise_dict['sky'] + noise_dict['readout'])**0.5
        total_exptime = noise_kwargs['exposure_time']*noise_kwargs['num_exposures']
        kwargs_data = dict(
                           background_rms=pixel_rms,
                           exposure_time=total_exptime,
                           ra_at_xy_0=ra_0,
                           dec_at_xy_0=dec_0,
                           transform_pix2angle=Mpix2coord,
                           image_data=image,
                           )
        psf_model = get_PSF_model(psf_type=noise_kwargs['psf_type'], pixel_scale=noise_kwargs['pixel_scale'], seeing=noise_kwargs['seeing'], kernel_size=psf_kernel_size, which_psf_maps=which_psf_maps)
        kwargs_psf = dict(
                          psf_type=psf_model.psf_type,
                          pixel_size=psf_model._pixel_size,
                          kernel_point_source=psf_model._kernel_point_source,
                          )
        image_band = [kwargs_data, kwargs_psf, self.kwargs_numerics]
        self.multi_band_list = [image_band] # single band
        self.kwargs_data_joint = dict(
                                      multi_band_list=self.multi_band_list,
                                      multi_band_type='multi-linear',
                                      time_delays_measured=measured_td,
                                      time_delays_uncertainties=measured_td_sigma
                                      )

    def get_lens_mass_mcmc_kwargs(self):
        """Compile reasonable MCMC kwargs for the lens mass components, PEMD and SHEAR

        """
        fixed_lens = []
        kwargs_lens_init = []
        kwargs_lens_sigma = []
        kwargs_lower_lens = []
        kwargs_upper_lens = []
        # Populate PEMD kwargs
        fixed_lens.append({}) 
        kwargs_lens_init.append({'theta_E': 1.0, 'e1': 0.0, 'e2': 0.0, 'gamma': 2.0, 'center_x': 0.0, 'center_y': 0.0})
        kwargs_lens_sigma.append({'theta_E': 0.2, 'e1': 0.1, 'e2': 0.1, 'gamma': 0.1, 'center_x': 0.01, 'center_y': 0.01})
        kwargs_lower_lens.append({'theta_E': 0.01, 'e1': -0.5, 'e2': -0.5, 'gamma': 1.5, 'center_x': -10, 'center_y': -10})
        kwargs_upper_lens.append({'theta_E': 10, 'e1': 0.5, 'e2': 0.5, 'gamma': 2.5, 'center_x': 10, 'center_y': 10})
        # Populate SHEAR kwargs
        fixed_lens.append({'ra_0': 0, 'dec_0': 0}) #fixed_lens.append({}) 
        kwargs_lens_init.append({'gamma1': 0.0, 'gamma2': 0.0}) #'ra_0': 0.0, 'dec_0': 0.0})
        kwargs_lens_sigma.append({'gamma1': 0.1, 'gamma2': 0.1}) #'ra_0': 0.05, 'dec_0': 0.05})
        kwargs_lower_lens.append({'gamma1': -0.5, 'gamma2': -0.5}) #'ra_0': -10.0, 'dec_0': -10.0})
        kwargs_upper_lens.append({'gamma1': 0.5, 'gamma2': 0.5}) #'ra_0': 10.0, 'dec_0': 10.0})
        # Collect into list
        lens_params = [kwargs_lens_init, kwargs_lens_sigma, fixed_lens, kwargs_lower_lens, kwargs_upper_lens]
        return lens_params

    def get_lens_light_mcmc_kwargs(self):
        """Compile reasonable MCMC kwargs for the SERSIC_ELLIPSE lens light

        """
        fixed_lens_light = []
        kwargs_lens_light_init = []
        kwargs_lens_light_sigma = []
        kwargs_lower_lens_light = []
        kwargs_upper_lens_light = []
        # Populate SERSIC_ELLIPSE kwargs
        fixed_lens_light.append({})
        kwargs_lens_light_init.append({'R_sersic': 0.5, 'n_sersic': 3.0, 'e1': 0.0, 'e2': 0.0, 'center_x': 0.0, 'center_y': 0.0})
        kwargs_lens_light_sigma.append({'R_sersic': 0.1, 'n_sersic': 0.5, 'e1': 0.1, 'e2': 0.1, 'center_x': 0.1, 'center_y': 0.1})
        kwargs_lower_lens_light.append({'e1': -0.5, 'e2': -0.5, 'R_sersic': 0.01, 'n_sersic': 0.5, 'center_x': -10, 'center_y': -10})
        kwargs_upper_lens_light.append({'e1': 0.5, 'e2': 0.5, 'R_sersic': 10, 'n_sersic': 8, 'center_x': 10, 'center_y': 10})
        # Collect into list
        lens_light_params = [kwargs_lens_light_init, kwargs_lens_light_sigma, fixed_lens_light, kwargs_lower_lens_light, kwargs_upper_lens_light]
        return lens_light_params

    def get_source_mcmc_kwargs(self):
        """Compile reasonable MCMC kwargs for the SERSIC_ELLIPSE source light

        """
        fixed_source = []
        kwargs_source_init = []
        kwargs_source_sigma = []
        kwargs_lower_source = []
        kwargs_upper_source = []
        # Populate SERSIC_ELLIPSE kwargs
        fixed_source.append({})
        kwargs_source_init.append({'R_sersic': 0.3, 'n_sersic': 2.0, 'e1': 0.0, 'e2': 0.0, 'center_x': 0.0, 'center_y': 0.0})
        kwargs_source_sigma.append({'R_sersic': 0.1, 'n_sersic': 0.5, 'e1': 0.1, 'e2': 0.1, 'center_x': 0.1, 'center_y': 0.1})
        kwargs_lower_source.append({'e1': -0.5, 'e2': -0.5, 'R_sersic': 0.001, 'n_sersic': .1, 'center_x': -10, 'center_y': -10})
        kwargs_upper_source.append({'e1': 0.5, 'e2': 0.5, 'R_sersic': 1.0, 'n_sersic': 5., 'center_x': 10, 'center_y': 10})
        # Collect into list
        source_params = [kwargs_source_init, kwargs_source_sigma, fixed_source, kwargs_lower_source, kwargs_upper_source]
        return source_params

    def get_ps_mcmc_kwargs(self, lens_init_kwargs):
        """Get the init kwargs for point source LENSED_POSITION based on the init kwargs for the lens model,
        a centered point source, and reasonable redshifts 

        Returns
        -------
        dict
            kwargs for LENSED_POSITION that can be used to initialize the MCMC
        
        """
        lens_model_class = LensModel(lens_model_list=['PEMD', 'SHEAR'], z_lens=0.5, z_source=1.5)
        kwargs_lens = lens_init_kwargs
        lensEquationSolver = LensEquationSolver(lens_model_class)
        x_image, y_image = lensEquationSolver.findBrightImage(0.0, 0.0, # center the point source
                                                              kwargs_lens, numImages=4,
                                                              min_distance=0.05, search_window=5.12)
        magnification = lens_model_class.magnification(x_image, y_image, kwargs=kwargs_lens)
        fixed_ps = [{}]
        kwargs_ps_init = [{'ra_image': x_image, 'dec_image': y_image, 'point_amp': np.abs(magnification)*1000}]  
        kwargs_ps_sigma = [{'ra_image': 0.01 * np.ones(len(x_image)), 'dec_image': 0.01 * np.ones(len(x_image))}]
        kwargs_lower_ps = [{'ra_image': -10 * np.ones(len(x_image)), 'dec_image': -10 * np.ones(len(y_image))}]
        kwargs_upper_ps = [{'ra_image': 10* np.ones(len(x_image)), 'dec_image': 10 * np.ones(len(y_image))}]
        ps_params = [kwargs_ps_init, kwargs_ps_sigma, fixed_ps, kwargs_lower_ps, kwargs_upper_ps]
        return ps_params

    def get_cosmo_mcmc_kwargs(self):
        """Compile reasonable MCMC kwargs for D_dt

        """
        fixed_cosmo = {}
        kwargs_cosmo_init = {'D_dt': 5000}
        kwargs_cosmo_sigma = {'D_dt': 3000}
        kwargs_lower_cosmo = {'D_dt': 0}
        kwargs_upper_cosmo = {'D_dt': 15000}
        cosmo_params = [kwargs_cosmo_init, kwargs_cosmo_sigma, fixed_cosmo, kwargs_lower_cosmo, kwargs_upper_cosmo]
        return cosmo_params

    def run_mcmc(self, mcmc_numerics):
        """Sample from the joint likelihood

        """
        fitting_seq = FittingSequence(self.kwargs_data_joint, 
                                      self.kwargs_model, 
                                      self.kwargs_constraints, 
                                      self.kwargs_likelihood, 
                                      self.kwargs_params)
        fitting_kwargs_list = [['MCMC', mcmc_numerics]]
        #with script_utils.HiddenPrints():
        chain_list_mcmc = fitting_seq.fit_sequence(fitting_kwargs_list)
        kwargs_result_mcmc = fitting_seq.best_fit()
        return chain_list_mcmc, kwargs_result_mcmc

    @property
    def kwargs_lens_init(self):
        return self.kwargs_params['lens_model'][0][0]

    @kwargs_lens_init.setter
    def kwargs_lens_init(self, new_kwargs_lens_init):
        self.kwargs_params['lens_model'][0][0] = new_kwargs_lens_init[0]
        self.kwargs_params['lens_model'][0][1] = new_kwargs_lens_init[1]

    @property
    def kwargs_lens_light_init(self):
        return self.kwargs_params['lens_light_model'][0][0]

    @kwargs_lens_light_init.setter
    def kwargs_lens_light_init(self, new_kwargs_lens_light_init):
        self.kwargs_params['lens_light_model'][0][0] = new_kwargs_lens_light_init

    @property
    def kwargs_source_init(self):
        return self.kwargs_params['source_model'][0][0]

    @kwargs_source_init.setter
    def kwargs_source_init(self, new_kwargs_source_init):
        self.kwargs_params['source_model'][0][0] = new_kwargs_source_init

    @property
    def kwargs_ps_init(self):
        return self.kwargs_params['point_source_model'][0][0]

    @kwargs_ps_init.setter
    def kwargs_ps_init(self, new_kwargs_ps_init):
        self.kwargs_params['point_source_model'][0][0] = new_kwargs_ps_init

    @property
    def kwargs_special_init(self):
        return self.kwargs_params['special'][0][0]

    @kwargs_special_init.setter
    def kwargs_special_init(self, new_kwargs_special_init):
        self.kwargs_params['special'][0][0] = new_kwargs_special_init