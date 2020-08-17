import os
import shutil
import unittest
import numpy as np
import pandas as pd
from addict import Dict
from torch.utils.data import DataLoader
from h0rton.trainval_data import XYData
from baobab.configs import BaobabConfig

class TestXYData(unittest.TestCase):
    """A suite of tests on data preprocessing
    
    """
    @classmethod
    def setUpClass(cls):
        cls.Y_cols = ["lens_mass_center_x", "src_light_center_x","lens_mass_center_y", "src_light_center_y", "external_shear_gamma_ext", "external_shear_psi_ext"]
        cls.train_Y_mean = np.random.randn(len(cls.Y_cols))
        cls.train_Y_std = np.abs(np.random.randn(len(cls.Y_cols))) + 1.0
        cls.train_baobab_cfg_path = 'h0rton/tests/test_trainval_data/baobab_train.json'
        cls.val_baobab_cfg_path = 'h0rton/tests/test_trainval_data/baobab_val.json'
        cls.train_baobab_cfg = BaobabConfig.from_file(cls.train_baobab_cfg_path)
        cls.val_baobab_cfg = BaobabConfig.from_file(cls.val_baobab_cfg_path)
        cls.original_exptime = 5400.0 # value in baobab_[train/val].json

        #####################
        # Generate toy data #
        #####################
        # Training (n_data = 2)
        os.makedirs(cls.train_baobab_cfg.out_dir, exist_ok=True)
        cls.train_metadata = pd.DataFrame.from_dict({
                                                    "lens_mass_center_x": [1.5, 2.0], 
                                                    "lens_mass_center_y": [1.8, 9.0], 
                                                    "src_light_center_x": [10.1, 12.5], 
                                                    "src_light_center_y": [29.2, 18.0],
                                                    "external_shear_gamma_ext": [-0.02, 0.02],
                                                    "external_shear_psi_ext": [-0.5, 0.5], 
                                                    "img_filename": ['X_{0:07d}.npy'.format(i) for i in range(2)],
                                                    })
        cls.train_metadata.to_csv(os.path.join(cls.train_baobab_cfg.out_dir, 'metadata.csv'), index=False)
        cls.img_0 = np.abs(np.random.randn(9)*2.0).reshape([1, 3, 3])
        cls.img_1 = np.abs(np.random.randn(9)*2.0).reshape([1, 3, 3])
        np.save(os.path.join(cls.train_baobab_cfg.out_dir, 'X_{0:07d}.npy'.format(0)), cls.img_0)
        np.save(os.path.join(cls.train_baobab_cfg.out_dir, 'X_{0:07d}.npy'.format(1)), cls.img_1)
        # Validation (n_data = 3)
        os.makedirs(cls.val_baobab_cfg.out_dir, exist_ok=True)
        cls.val_metadata = pd.DataFrame.from_dict({
                                                  "lens_mass_center_x": np.random.randn(3), 
                                                  "lens_mass_center_y": np.random.randn(3), 
                                                  "src_light_center_x": np.random.randn(3), 
                                                  "src_light_center_y": np.random.randn(3), 
                                                  "external_shear_gamma_ext": np.random.randn(3),
                                                    "external_shear_psi_ext": np.random.randn(3), 
                                                  "img_filename": ['X_{0:07d}.npy'.format(i) for i in range(3)],
                                                  })
        cls.img_0_val = np.abs(np.random.randn(9)*2.0).reshape([1, 3, 3])
        cls.img_1_val = np.abs(np.random.randn(9)*2.0).reshape([1, 3, 3])
        cls.img_2_val = np.abs(np.random.randn(9)*2.0).reshape([1, 3, 3])
        np.save(os.path.join(cls.val_baobab_cfg.out_dir, 'X_{0:07d}.npy'.format(0)), cls.img_0_val)
        np.save(os.path.join(cls.val_baobab_cfg.out_dir, 'X_{0:07d}.npy'.format(1)), cls.img_1_val)
        np.save(os.path.join(cls.val_baobab_cfg.out_dir, 'X_{0:07d}.npy'.format(2)), cls.img_2_val)
        cls.val_metadata.to_csv(os.path.join(cls.val_baobab_cfg.out_dir, 'metadata.csv'), index=False)

    @classmethod
    def tearDownClass(cls):
        """Remove the toy data

        """
        shutil.rmtree(cls.train_baobab_cfg.out_dir)
        shutil.rmtree(cls.val_baobab_cfg.out_dir)

    def test_X_identity(self):
        """Test if the input iamge equals the dataset image, when nothing is done to the image at all

        """
        train_data = XYData(True, self.Y_cols, 'FloatTensor', define_src_pos_wrt_lens=True, rescale_pixels=False, log_pixels=False, add_pixel_noise=False, eff_exposure_time={'TDLMC_F160W': self.original_exptime}, train_Y_mean=self.train_Y_mean, train_Y_std=self.train_Y_std, train_baobab_cfg_path=self.train_baobab_cfg_path, val_baobab_cfg_path=self.val_baobab_cfg_path, for_cosmology=False)
        actual_img, _ = train_data[0]
        expected_img = self.img_0
        np.testing.assert_array_almost_equal(actual_img, expected_img, err_msg='test_X_identity')

    def test_X_transformation_log(self):
        """Test if the images transform as expected, with log(1+X)

        """
        train_data = XYData(True, self.Y_cols, 'FloatTensor', define_src_pos_wrt_lens=True, rescale_pixels=False, log_pixels=True, add_pixel_noise=False, eff_exposure_time={'TDLMC_F160W': self.original_exptime}, train_Y_mean=self.train_Y_mean, train_Y_std=self.train_Y_std, train_baobab_cfg_path=self.train_baobab_cfg_path, val_baobab_cfg_path=self.val_baobab_cfg_path, for_cosmology=False)
        actual_img, _ = train_data[0]
        expected_img = self.img_0
        expected_img = np.log1p(expected_img)
        np.testing.assert_array_almost_equal(actual_img, expected_img, err_msg='test_X_transformation_log')

    def test_X_transformation_rescale(self):
        """Test if the images transform as expected, with whitening

        """
        train_data = XYData(True, self.Y_cols, 'FloatTensor', define_src_pos_wrt_lens=True, rescale_pixels=True, log_pixels=False, add_pixel_noise=False, eff_exposure_time={'TDLMC_F160W': self.original_exptime}, train_Y_mean=self.train_Y_mean, train_Y_std=self.train_Y_std, train_baobab_cfg_path=self.train_baobab_cfg_path, val_baobab_cfg_path=self.val_baobab_cfg_path, for_cosmology=False)
        actual_img, _ = train_data[0]
        expected_img = self.img_0
        expected_img = (expected_img - np.mean(expected_img))/np.std(expected_img, ddof=1)
        np.testing.assert_array_almost_equal(actual_img, expected_img, err_msg='test_X_transformation_rescale')

    def test_X_transformation_log_rescale(self):
        """Test if the images transform as expected, with log(1+X) and whitening

        """
        # Without exposure time factor
        train_data = XYData(True, self.Y_cols, 'FloatTensor', define_src_pos_wrt_lens=True, rescale_pixels=True, log_pixels=True, add_pixel_noise=False, eff_exposure_time={'TDLMC_F160W': self.original_exptime}, train_Y_mean=self.train_Y_mean, train_Y_std=self.train_Y_std, train_baobab_cfg_path=self.train_baobab_cfg_path, val_baobab_cfg_path=self.val_baobab_cfg_path, for_cosmology=False)
        actual_img, _ = train_data[0]
        expected_img = self.img_0
        expected_img = np.log1p(expected_img)
        # Note torch std takes into account Bessel correction
        expected_img = (expected_img - np.mean(expected_img))/np.std(expected_img, ddof=1)
        np.testing.assert_array_almost_equal(actual_img, expected_img, err_msg='test_X_transformation_log_rescale, without exposure time factor')
        # With exposure time factor
        train_data = XYData(True, self.Y_cols, 'FloatTensor', define_src_pos_wrt_lens=True, rescale_pixels=True, log_pixels=True, add_pixel_noise=False, eff_exposure_time={'TDLMC_F160W': self.original_exptime*2.0}, train_Y_mean=self.train_Y_mean, train_Y_std=self.train_Y_std, train_baobab_cfg_path=self.train_baobab_cfg_path, val_baobab_cfg_path=self.val_baobab_cfg_path, for_cosmology=False)
        actual_img, _ = train_data[0]
        expected_img = self.img_0*2.0
        expected_img = np.log1p(expected_img)
        # Note torch std takes into account Bessel correction
        expected_img = (expected_img - np.mean(expected_img))/np.std(expected_img, ddof=1)
        np.testing.assert_array_almost_equal(actual_img, expected_img, err_msg='test_X_transformation_log_rescale, with exposure time factor')

    def test_X_exposure_time_factor(self):
        """Test if the images scale by the new effective exposure time correctly

        """
        train_data = XYData(True, self.Y_cols, 'FloatTensor', define_src_pos_wrt_lens=True, rescale_pixels=False, log_pixels=False, add_pixel_noise=False, eff_exposure_time={'TDLMC_F160W': self.original_exptime*2.0}, train_Y_mean=self.train_Y_mean, train_Y_std=self.train_Y_std, train_baobab_cfg_path=self.train_baobab_cfg_path, val_baobab_cfg_path=self.val_baobab_cfg_path, for_cosmology=False)
        actual_img, _ = train_data[0]
        expected_img = self.img_0*2.0
        np.testing.assert_array_almost_equal(actual_img, expected_img, err_msg='test_X_exposure_time_factor')

    def test_Y_transformation_(self):
        """Test if the target Y whitens correctly

        """
        # Training
        train_data = XYData(True, self.Y_cols, 'FloatTensor', define_src_pos_wrt_lens=True, rescale_pixels=False, log_pixels=False, add_pixel_noise=False, eff_exposure_time={'TDLMC_F160W': self.original_exptime*2.0}, train_Y_mean=None, train_Y_std=None, train_baobab_cfg_path=self.train_baobab_cfg_path, val_baobab_cfg_path=self.val_baobab_cfg_path, for_cosmology=False)
        _, actual_Y_0 = train_data[0]
        _, actual_Y_1 = train_data[1]
        actual_Y = np.stack([actual_Y_0, actual_Y_1], axis=0)
        Y_df = self.train_metadata[self.Y_cols].copy()
        Y_df['src_light_center_x'] -= Y_df['lens_mass_center_x']
        Y_df['src_light_center_y'] -= Y_df['lens_mass_center_y']
        expected_Y = Y_df.values
        before_whitening_Y = Y_df.values
        #expected_Y = (expected_Y - self.train_Y_mean.reshape([1, -1]))/self.train_Y_std.reshape([1, -1])
        expected_Y[np.argmin(before_whitening_Y, axis=0), np.arange(len(self.Y_cols))] = -1
        expected_Y[np.argmax(before_whitening_Y, axis=0), np.arange(len(self.Y_cols))] = 1
        np.testing.assert_array_equal(actual_Y_0.shape, [len(self.Y_cols),], err_msg='shape of single example Y for training')
        np.testing.assert_array_almost_equal(actual_Y, expected_Y, err_msg='transformed Y for training')
        # Validation
        val_data = XYData(False, self.Y_cols, 'FloatTensor', define_src_pos_wrt_lens=True, rescale_pixels=False, log_pixels=False, add_pixel_noise=False, eff_exposure_time={'TDLMC_F160W': self.original_exptime*2.0}, train_Y_mean=self.train_Y_mean, train_Y_std=self.train_Y_std, train_baobab_cfg_path=self.train_baobab_cfg_path, val_baobab_cfg_path=self.val_baobab_cfg_path, for_cosmology=False)
        _, actual_Y_0 = val_data[0]
        _, actual_Y_1 = val_data[1]
        _, actual_Y_2 = val_data[2]
        actual_Y = np.stack([actual_Y_0, actual_Y_1, actual_Y_2], axis=0)
        expected_Y = self.val_metadata[self.Y_cols].copy()
        expected_Y['src_light_center_x'] -= expected_Y['lens_mass_center_x']
        expected_Y['src_light_center_y'] -= expected_Y['lens_mass_center_y']
        expected_Y = expected_Y.values
        expected_Y = (expected_Y - self.train_Y_mean.reshape([1, -1]))/self.train_Y_std.reshape([1, -1])
        np.testing.assert_array_equal(actual_Y_0.shape, [len(self.Y_cols),], err_msg='shape of single example Y for validation for arbitrary train mean and std')
        np.testing.assert_array_almost_equal(actual_Y, expected_Y, err_msg='transformed Y for validation for arbitrary train mean and std')

    def test_train_vs_val(self):
        """Test if the images and metadata are loaded from the correct folder (train/val)

        """
        train_data = XYData(True, self.Y_cols, 'FloatTensor', define_src_pos_wrt_lens=True, rescale_pixels=False, log_pixels=False, add_pixel_noise=False, eff_exposure_time={'TDLMC_F160W': self.original_exptime*2.0}, train_Y_mean=self.train_Y_mean, train_Y_std=self.train_Y_std, train_baobab_cfg_path=self.train_baobab_cfg_path, val_baobab_cfg_path=self.val_baobab_cfg_path, for_cosmology=False)
        val_data = XYData(False, self.Y_cols, 'FloatTensor', define_src_pos_wrt_lens=True, rescale_pixels=False, log_pixels=False, add_pixel_noise=False, eff_exposure_time={'TDLMC_F160W': self.original_exptime*2.0}, train_Y_mean=self.train_Y_mean, train_Y_std=self.train_Y_std, train_baobab_cfg_path=self.train_baobab_cfg_path, val_baobab_cfg_path=self.val_baobab_cfg_path, for_cosmology=False)
        np.testing.assert_equal(len(train_data), 2, err_msg='reading from correct folder (train/val)')
        np.testing.assert_equal(len(val_data), 3, err_msg='reading from correct folder (train/val)')

    def test_tensor_type(self):
        """Test if X, Y are of the configured data type

        """
        # DoubleTensor
        train_data = XYData(True, self.Y_cols, 'DoubleTensor', define_src_pos_wrt_lens=True, rescale_pixels=False, log_pixels=False, add_pixel_noise=False, eff_exposure_time={'TDLMC_F160W': self.original_exptime*2.0}, train_Y_mean=self.train_Y_mean, train_Y_std=self.train_Y_std, train_baobab_cfg_path=self.train_baobab_cfg_path, val_baobab_cfg_path=self.val_baobab_cfg_path, for_cosmology=False)
        actual_X_0, actual_Y_0 = train_data[0]
        assert actual_X_0.type() == 'torch.DoubleTensor'
        assert actual_Y_0.type() == 'torch.DoubleTensor'

        # FloatTensor
        train_data = XYData(True, self.Y_cols, 'FloatTensor', define_src_pos_wrt_lens=True, rescale_pixels=False, log_pixels=False, add_pixel_noise=False, eff_exposure_time={'TDLMC_F160W': self.original_exptime*2.0}, train_Y_mean=self.train_Y_mean, train_Y_std=self.train_Y_std, train_baobab_cfg_path=self.train_baobab_cfg_path, val_baobab_cfg_path=self.val_baobab_cfg_path, for_cosmology=False)
        actual_X_0, actual_Y_0 = train_data[0]
        assert actual_X_0.type() == 'torch.FloatTensor'
        assert actual_Y_0.type() == 'torch.FloatTensor'

if __name__ == '__main__':
    unittest.main()