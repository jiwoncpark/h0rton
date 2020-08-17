import numpy as np
import unittest
from h0rton.h0_inference import H0Posterior, h0_utils
import scipy.stats
from astropy.cosmology import FlatLambdaCDM
from lenstronomy.Analysis.td_cosmography import TDCosmography

class TestH0Posterior(unittest.TestCase):
    """A suite of tests verifying that the simple MC sampling of the H0 Posterior is correct
    
    """
    @classmethod
    def setUpClass(cls):
        cls.kappa_loc = 0.0
        cls.transformed_kappa_loc = 1.0/(1.0 - cls.kappa_loc)
        cls.kappa_std = 0.025
        cls.H0_prior = getattr(scipy.stats, 'uniform')(loc=50.0, scale=40.0)
        cls.kappa_ext_prior_original = getattr(scipy.stats, 'norm')(loc=cls.kappa_loc, scale=cls.kappa_std)
        cls.kappa_ext_prior_transformed = getattr(scipy.stats, 'norm')(loc=cls.transformed_kappa_loc, scale=cls.kappa_std)

        cls.true_kappa_ext = -0.002
        cls.kappa_ext_prior_true = h0_utils.DeltaFunction(cls.true_kappa_ext)
        # Generate lens with some true kappa, H0 
        cls.true_H0 = 70.0
        cls.true_Om0 = 0.3
        cls.z_lens = 0.5
        cls.z_src = 1.5
        cls.lens_model = dict(
                            external_shear_gamma1=0.01,
                            external_shear_gamma2=-0.005,
                            lens_light_R_sersic=1.0,
                            lens_light_center_x=0.01,
                            lens_light_center_y=0.01,
                            lens_light_e1=-0.01,
                            lens_light_e2=-0.2,
                            lens_mass_center_x=0.02,
                            lens_mass_center_y=0.01,
                            lens_mass_e1=-0.1,
                            lens_mass_e2=-0.1,
                            lens_mass_gamma=2.0,
                            lens_mass_theta_E=1.2,
                            src_light_center_x=-0.04,
                            src_light_center_y=0.08,
                            )
        cls.kwargs_model = dict(
                                lens_model_list=['PEMD', 'SHEAR'],
                                lens_light_model_list=['SERSIC_ELLIPSE'],
                                source_light_model_list=['SERSIC_ELLIPSE'],
                                point_source_model_list=['SOURCE_POSITION'],
                                )       
        cls.true_cosmo = FlatLambdaCDM(H0=cls.true_H0, Om0=cls.true_Om0)
        cls.kwargs_lens_eqn_solver = {'min_distance': 0.05, 'search_window': 5, 'num_iter_max': 100}
        cls.td_cosmo = TDCosmography(cls.z_lens, cls.z_src, cls.kwargs_model, cosmo_fiducial=cls.true_cosmo, kwargs_lens_eqn_solver=cls.kwargs_lens_eqn_solver)
        cls.baobab_time_delays = True
        cls.define_src_pos_wrt_lens = True
        cls.aniso_param_prior = None
        cls.kinematics = None
        

    def setUp(self):
        np.random.seed(1113)

    def test_H0Posterior_transformed_kappa_sampling(self):
        """Test sampling of transformed kappa

        """
        h0_post = H0Posterior(self.H0_prior, self.kappa_ext_prior_transformed, self.kwargs_model, self.baobab_time_delays, self.true_Om0, self.define_src_pos_wrt_lens, exclude_vel_disp=True, aniso_param_prior=None, kinematics=None, kappa_transformed=True, kwargs_lens_eqn_solver=self.kwargs_lens_eqn_solver)
        kappa_samples = np.empty(10000)
        for i in range(10000):
            kappa_samples[i] = h0_post.sample_kappa_ext(i)
        kappa_transformed_samples = 1.0/(1.0 - kappa_samples)
        np.testing.assert_almost_equal(np.mean(kappa_transformed_samples), self.transformed_kappa_loc, decimal=2, err_msg="transformed kappa mean")
        np.testing.assert_almost_equal(np.std(kappa_transformed_samples, ddof=1), self.kappa_std, decimal=2, err_msg="transformed kappa std")

    def test_H0Posterior_original_kappa_sampling(self):
        """Test sampling of original kappa

        """
        h0_post = H0Posterior(self.H0_prior, self.kappa_ext_prior_original, self.kwargs_model, self.baobab_time_delays, self.true_Om0, self.define_src_pos_wrt_lens, exclude_vel_disp=True, aniso_param_prior=None, kinematics=None, kappa_transformed=False, kwargs_lens_eqn_solver=self.kwargs_lens_eqn_solver)
        kappa_samples = np.empty(10000)
        for i in range(10000):
            kappa_samples[i] = h0_post.sample_kappa_ext(i)
        np.testing.assert_almost_equal(np.mean(kappa_samples), self.kappa_loc, decimal=2, err_msg="original kappa")
        np.testing.assert_almost_equal(np.std(kappa_samples, ddof=1), self.kappa_std, decimal=2, err_msg="original kappa std")

    def test_H0Posterior_H0_sampling(self):
        """Test sampling of H0

        """
        h0_post = H0Posterior(self.H0_prior, self.kappa_ext_prior_original, self.kwargs_model, self.baobab_time_delays, self.true_Om0, self.define_src_pos_wrt_lens, exclude_vel_disp=True, aniso_param_prior=None, kinematics=None, kappa_transformed=False, kwargs_lens_eqn_solver=self.kwargs_lens_eqn_solver)
        h0_samples = np.empty(10000)
        for i in range(10000):
            h0_samples[i] = h0_post.sample_H0(i)
        np.testing.assert_almost_equal(np.mean(h0_samples), 70.0, decimal=1, err_msg="H0 sampling")

    def test_H0Posterior_format_lens_model(self):
        """Test if the formatted lens model (nesting of the lens model dictionary) can be used as input to TDCosmography

        """
        # Instantiate H0Posterior to test its method, format_lens_model
        h0_post = H0Posterior(self.H0_prior, self.kappa_ext_prior_transformed, self.kwargs_model, self.baobab_time_delays, self.true_Om0, self.define_src_pos_wrt_lens, exclude_vel_disp=True, aniso_param_prior=None, kinematics=None, kappa_transformed=True, kwargs_lens_eqn_solver=self.kwargs_lens_eqn_solver)
        formatted_lens_model = h0_post.format_lens_model(self.lens_model)
        # Instantiate TDCosmography to test if its method, time_delays, can be run when the output of format_lens_model is passed in
        td_cosmo = TDCosmography(self.z_lens, self.z_src, self.kwargs_model, cosmo_fiducial=self.true_cosmo, kwargs_lens_eqn_solver=self.kwargs_lens_eqn_solver)
        _ = td_cosmo.time_delays(formatted_lens_model['kwargs_lens'], formatted_lens_model['kwargs_ps'], kappa_ext=self.true_kappa_ext)

    def test_H0Posterior_H0_recovery_true_kappa_time_delays_lens_model(self):
        """Test if true H0 is recovered if true lens model, kappa, and time delays are input

        """
        # Instantiate H0Posterior with the delta-function prior on kappa_ext at the truth
        h0_post = H0Posterior(self.H0_prior, self.kappa_ext_prior_true, self.kwargs_model, self.baobab_time_delays, self.true_Om0, self.define_src_pos_wrt_lens, exclude_vel_disp=True, aniso_param_prior=None, kinematics=None, kappa_transformed=False, kwargs_lens_eqn_solver=self.kwargs_lens_eqn_solver)
        formatted_lens_model = h0_post.format_lens_model(self.lens_model) # tested separately in test_H0Posterior_format_lens_model, used as a utility function here
        # Generate the true time delays and image positions
        true_td, true_x_image, true_y_image = self.td_cosmo.time_delays(formatted_lens_model['kwargs_lens'], formatted_lens_model['kwargs_ps'], kappa_ext=self.true_kappa_ext)
        # Compute the true time delay offset
        increasing_dec_i = np.argsort(true_y_image)
        measured_td = true_td[increasing_dec_i]
        measured_td_wrt0 = measured_td[1:] - measured_td[0]
        
        h0_post.set_cosmology_observables(self.z_lens, self.z_src, measured_td_wrt0, 0.25, abcd_ordering_i=range(len(true_y_image)), true_img_dec=true_y_image, true_img_ra=true_x_image, kappa_ext=self.true_kappa_ext)
        # "Infer" the H0 given the true time delays as measurement data, true kappa, and true lens model
        h0_samples = np.empty(5000)
        h0_weights = np.empty(5000)
        for i in range(5000):
            h0_samples[i], h0_weights[i] = h0_post.get_h0_sample(self.lens_model, i)
        normal_stats = h0_utils.get_normal_stats_naive(h0_samples, h0_weights)

        # Compare the inferred central H0 with truth
        np.testing.assert_almost_equal(normal_stats['mean'], self.true_H0, decimal=1, err_msg="H0 sampling")

    def test_chuck_images(self):
        """Test if the correct images are removed in the case of extra image detections

        """
        # Instantiate a dummy H0Posterior to test its method, chuck_images
        h0_post = H0Posterior(self.H0_prior, self.kappa_ext_prior_true, self.kwargs_model, self.baobab_time_delays, self.true_Om0, self.define_src_pos_wrt_lens, exclude_vel_disp=True, aniso_param_prior=None, kinematics=None, kappa_transformed=False, kwargs_lens_eqn_solver=self.kwargs_lens_eqn_solver)
        formatted_lens_model = h0_post.format_lens_model(self.lens_model) # tested separately in test_H0Posterior_format_lens_model, used as a utility function here
        # Generate the true time delays and image positions
        true_td, true_x_image, true_y_image = self.td_cosmo.time_delays(formatted_lens_model['kwargs_lens'], formatted_lens_model['kwargs_ps'], kappa_ext=self.true_kappa_ext)
        # Compute the true time delay offset
        increasing_dec_i = np.argsort(true_y_image)
        measured_td = true_td[increasing_dec_i]
        measured_td_wrt0 = measured_td[1:] - measured_td[0]
        h0_post.set_cosmology_observables(self.z_lens, self.z_src, measured_td_wrt0, 0.25, abcd_ordering_i=range(len(true_y_image)), true_img_dec=true_y_image, true_img_ra=true_x_image, kappa_ext=self.true_kappa_ext)

        true_n_img = len(true_x_image)
        # Generate some fake measured td (10% error) and images (1% error) with 2 extra images
        extra_images_x = np.random.randn(true_n_img + 2)
        extra_images_y = np.random.randn(true_n_img + 2)
        extra_td = np.random.randn(true_n_img + 2)
        measured_images_x = true_x_image*(1.0 + np.random.randn(true_n_img)*0.01)
        measured_images_y = true_y_image*(1.0 + np.random.randn(true_n_img)*0.01)
        measured_td = true_td*(1.0 + np.random.randn(true_n_img)*0.1)
        extra_images_x[[0, 2, 4, 5]] = measured_images_x # randomly assign location of extra images but note that the measured y image must be ordered increasing dec at this point (so the random assignment is monotonic increasing)
        extra_images_y[[0, 2, 4, 5]] = measured_images_y
        extra_td[[0, 2, 4, 5]] = measured_td
        inferred_td, x_image, y_image = h0_post.chuck_images(extra_td, extra_images_x, extra_images_y)

        np.testing.assert_array_almost_equal(x_image, measured_images_x, err_msg="x image")
        np.testing.assert_array_almost_equal(y_image, measured_images_y, err_msg="y image")
        np.testing.assert_array_almost_equal(inferred_td, measured_td, err_msg="td")

if __name__ == '__main__':
    unittest.main()