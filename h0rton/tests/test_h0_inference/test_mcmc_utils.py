import numpy as np
import h0rton.h0_inference.mcmc_utils as mcmc_utils
import unittest

class TestMCMCUtils(unittest.TestCase):
    """A suite of tests for the h0rton.h0_inference.mcmc_utils package
    
    """

    @classmethod
    def setUpClass(cls):
        cls.init_dict = dict(
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

    def test_get_lens_kwargs(self):
        """Test if the default lens kwargs dictionary config is plausible

        """
        not_null_spread = mcmc_utils.get_lens_kwargs(self.init_dict, null_spread=False)
        lens_param_names = not_null_spread[-1][0].keys()
        ext_shear_names = not_null_spread[-1][-1].keys()
        # Check that lower is less than upper
        for p in lens_param_names:
            assert not_null_spread[-2][0][p] < not_null_spread[-1][0][p]
        for p in ext_shear_names:
            assert not_null_spread[-2][-1][p] < not_null_spread[-1][-1][p]
        null_spread = mcmc_utils.get_lens_kwargs(self.init_dict, null_spread=True)
        # Fixed, external_shear, ra_0 should equal init, lens_mass, center_x
        assert null_spread[2][-1]['ra_0'] == null_spread[0][0]['center_x']
        assert null_spread[2][-1]['dec_0'] == null_spread[0][0]['center_y']
        # TODO: check that std is small < 1.e-5
        for param_name, param_sigma in null_spread[1][0].items():
            assert param_sigma < 1.e-5
        for param_name, param_sigma in null_spread[1][-1].items():
            assert param_sigma < 1.e-5

    def test_get_ps_kwargs(self):
        """Test if the default ps kwargs dictionary config defined on the image plane is plausible

        """
        ps_kwargs = mcmc_utils.get_ps_kwargs(measured_img_ra=np.random.randn(4), measured_img_dec=np.random.randn(4), astrometry_sigma=0.005)
        ps_param_names = ps_kwargs[-1][0].keys()
        # Check that lower is less than upper
        for p in ps_param_names:
            assert np.all(ps_kwargs[-2][0][p] < ps_kwargs[-1][0][p])

    def test_get_ps_kwargs_src_plane(self):
        """Test if the default ps kwargs dictionary config defined on the source plane is plausible

        """
        ps_kwargs = mcmc_utils.get_ps_kwargs_src_plane(self.init_dict, astrometry_sigma=0.005)
        ps_param_names = ps_kwargs[-1][0].keys()
        # Check that lower is less than upper
        for p in ps_param_names:
            assert ps_kwargs[-2][0][p] < ps_kwargs[-1][0][p]

    def test_get_light_kwargs(self):
        """Test if the default light kwargs dictionary config is plausible

        """
        init_R = 0.5
        not_null_spread = mcmc_utils.get_light_kwargs(init_R, null_spread=False)
        lens_light_param_names = not_null_spread[-1][0].keys()
        # Check that lower is less than upper
        for p in lens_light_param_names:
            assert not_null_spread[-2][0][p] < not_null_spread[-1][0][p]
        null_spread = mcmc_utils.get_light_kwargs(init_R, null_spread=True)
        # TODO: check that std is small < 1.e-5
        for param_name, param_sigma in null_spread[1][0].items():
            assert param_sigma < 1.e-5

    def test_get_special_kwargs(self):
        """Test if the default special kwargs dictionary config is plausible

        """
        special_kwargs = mcmc_utils.get_special_kwargs(n_img=4, astrometry_sigma=5.e-3)
        special_param_names = special_kwargs[-1].keys()
        # Check that lower is less than upper
        for p in special_param_names:
            assert np.all(special_kwargs[-2][p] < special_kwargs[-1][p])

    def test_postprocess_mcmc_chain(self):
        # TODO
        pass

    def test_HybridBNNPenalty(self):
        # TODO
        pass

    def test_get_idx_for_params(self):
        """Test if `get_idx_for_params` returns the right indices

        """
        Y_dim = 4
        out_dim = Y_dim**2 + 3*Y_dim + 1
        orig_Y_cols = ['a', 'b', 'c', 'd']
        to_test = mcmc_utils.get_idx_for_params(out_dim, orig_Y_cols, ['a', 'c'], 'DoubleGaussianNLL', debug=True)
        tril_mask = np.array([0, 1, 3, 4, 5, 6, 8])
        idx_within_tril1 = Y_dim + tril_mask
        param_idx = [0, 2]
        np.testing.assert_array_equal(to_test['param_idx'], param_idx)
        np.testing.assert_array_equal(np.sort(to_test['tril_mask']), np.sort(tril_mask))
        np.testing.assert_array_equal(np.sort(to_test['idx_within_tril1']), np.sort(idx_within_tril1))

    def test_remove_parameters_from_pred(self):
        """Test if correct parameters are removed from the NN output

        """
        orig_pred = np.arange(20).reshape([4, 5])
        #array([[ 0,  1,  2,  3,  4],
        # [ 5,  6,  7,  8,  9],
        # [10, 11, 12, 13, 14],
        # [15, 16, 17, 18, 19]])
        remove_idx = [1, 3]
        new_pred = mcmc_utils.remove_parameters_from_pred(orig_pred, remove_idx, return_as_tensor=False, device='cpu')
        expected_new_pred = np.array([[ 0,   2,   4],
                                   [ 5,    7,   9],
                                   [10,  12,  14],
                                   [15,  17,  19]])
        np.testing.assert_array_equal(new_pred, expected_new_pred, err_msg="test_remove_parameters_from_pred")

    def test_split_component_param(self):
        """Test if string split of Baobab column names is done correctly

        """
        actual = mcmc_utils.split_component_param('lens_mass_theta_E', sep='_', pos=2)
        expected = ('lens_mass', 'theta_E')
        assert actual == expected

    def test_dict_to_array(self):
        """Test if dict from MCMC iteration is converted into array form correctly

        """
        Y_cols = ['lens_mass_gamma', 'lens_mass_theta_E', 'lens_mass_center_x', 'lens_mass_center_y', 'external_shear_gamma1', 'external_shear_gamma2', 'src_light_R_sersic', 'src_light_center_x', 'src_light_center_y']
        kwargs_ps = [{'ra_source': 0.1, 'dec_source': -0.2}]
        kwargs_source = [{'R_sersic': 0.3}]
        kwargs_lens = [{'gamma': 2.0, 'theta_E': 1.5, 'center_x': -0.05, 'center_y': 0.01}, {'gamma1': -0.01, 'gamma2': 0.005}]
        actual = mcmc_utils.dict_to_array(Y_cols, kwargs_lens, kwargs_source, kwargs_lens_light=None, kwargs_ps=kwargs_ps,)
        expected = np.array([[2.0, 1.5, -0.05, 0.01, -0.01, 0.005, 0.3, 0.1 - (-0.05), -0.2 - 0.01]])
        np.testing.assert_array_equal(actual, expected, err_msg="test_dict_to_array")

    def test_reorder_to_param_class(self):
        """Test if dict from MCMC iteration is converted into array form correctly

        """
        bnn_array = np.arange(20).reshape([1, 4, 5])
        bnn_Y_cols = ['lens_mass_center_x', 'lens_mass_center_y', 'src_light_center_x', 'src_light_center_y', 'lens_mass_theta_E']
        D_dt_array = -np.arange(4).reshape([1, 4, 1])
        param_class_Y_cols = ['ra_source', 'dec_source', 'theta_E_lens0', 'center_x_lens0', 'center_y_lens0', 'D_dt']
        actual = mcmc_utils.reorder_to_param_class(bnn_Y_cols, param_class_Y_cols, bnn_array, D_dt_array)
        expected = np.array([[[   2+0,  3+1,  4, 0,  1, -0],
                             [   7+5,  8+6,  9, 5,  6, -1],
                             [ 12+10, 13+11, 14, 10, 11, -2],
                             [ 17+15, 18+16, 19, 15, 16, -3]]])
        np.testing.assert_array_equal(actual, expected, err_msg="test_reorder_to_param_class")

if __name__ == '__main__':
    unittest.main()