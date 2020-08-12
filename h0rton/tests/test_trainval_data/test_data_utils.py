import os
import shutil
import unittest
import numpy as np
import torch
import pandas as pd
from addict import Dict
from torch.utils.data import DataLoader
from h0rton.trainval_data import XYData
import h0rton.trainval_data.data_utils as data_utils

class TestDataUtils(unittest.TestCase):
    """"A suite of tests for data utility functions
    
    """
    @classmethod
    def setUpClass(cls):
        cls.Y_cols = ["lens_mass_center_x", "src_light_center_x","lens_mass_center_y", "src_light_center_y", "external_shear_gamma_ext", "external_shear_psi_ext"]
        cls.train_Y_mean = np.random.randn(len(cls.Y_cols))
        cls.train_Y_std = np.abs(np.random.randn(len(cls.Y_cols))) + 1.0
        cls.img_numpy = np.abs(np.random.randn(50*50)*2.0).reshape([1, 50, 50])
        cls.img_torch = torch.from_numpy(cls.img_numpy)
        n_data = 2
        cls.metadata = pd.DataFrame.from_dict({"lens_mass_center_x": np.random.randn(n_data), 
                                              "lens_mass_center_y": np.random.randn(n_data), 
                                              "src_light_center_x": np.random.randn(n_data), 
                                              "src_light_center_y": np.random.randn(n_data), 
                                              "external_shear_gamma_ext": np.random.randn(n_data),
                                                "external_shear_psi_ext": np.random.randn(n_data)
                                                })

    def test_whiten_pixels(self):
        """Test the torch pixel whitening vs. numpy

        """
        actual = data_utils.whiten_pixels(self.img_torch)
        expected = (self.img_numpy - np.mean(self.img_numpy))/np.std(self.img_numpy, ddof=1)
        np.testing.assert_array_almost_equal(actual, expected, err_msg='test_whiten_pixels')

    def test_asinh(self):
        """Test the torch asinh approximation vs. numpy

        """
        actual = data_utils.asinh(self.img_torch)
        expected = np.arcsinh(self.img_numpy)
        np.testing.assert_array_almost_equal(actual, expected, err_msg='test_asinh')

    def test_plus_1_log(self):
        """Test the torch log(1+X) vs. numpy

        """
        actual = data_utils.plus_1_log(self.img_torch)
        expected = np.log1p(self.img_numpy)
        np.testing.assert_array_almost_equal(actual, expected, err_msg='test_plus_1_log')

    def test_rescale_01(self):
        """Test the torch minmax stretching vs. numpy

        """
        actual = data_utils.rescale_01(self.img_torch)
        expected = (self.img_numpy - self.img_numpy.min())/(self.img_numpy.max() - self.img_numpy.min())
        np.testing.assert_array_almost_equal(actual, expected, err_msg='test_rescale_01')

    def test_whiten_Y_cols(self):
        """Test the Y whitening in pandas vs. numpy

        """
        # All columns
        actual = self.metadata.copy()
        data_utils.whiten_Y_cols(actual, self.train_Y_mean, self.train_Y_std, self.Y_cols)
        expected = (self.metadata[self.Y_cols].values - self.train_Y_mean.reshape([1, -1]))/self.train_Y_std.reshape([1, -1])
        np.testing.assert_array_almost_equal(actual[self.Y_cols].values, expected, err_msg='test_whiten_Y_cols')
        # Subset of columns
        actual = self.metadata.copy()
        subset_train_Y_mean = self.train_Y_mean[:3]
        subset_train_Y_std = self.train_Y_std[:3]
        subset_Y_cols = self.Y_cols[:3]
        data_utils.whiten_Y_cols(actual, subset_train_Y_mean, subset_train_Y_std, subset_Y_cols)
        expected = (self.metadata[subset_Y_cols].values - subset_train_Y_mean.reshape([1, -1]))/subset_train_Y_std.reshape([1, -1])
        np.testing.assert_array_almost_equal(actual[subset_Y_cols].values, expected, err_msg='test_whiten_Y_cols with a subset of the columns')

if __name__ == '__main__':
    unittest.main()