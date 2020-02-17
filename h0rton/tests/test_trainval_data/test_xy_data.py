import os
import shutil
import unittest
import numpy as np
import pandas as pd
from addict import Dict
from torch.utils.data import DataLoader
from h0rton.trainval_data import XYData

class TestXYData(unittest.TestCase):
    """A suite of tests verifying that the input PDFs and the sample distributions
    match.
    
    """

    def test_xy_data(self):
        """Test the XYData Dataset and DataLoader

        """
        test_data_dir = os.path.abspath('test_data_dir')
        os.makedirs(test_data_dir) 
        # Generate fake data
        data_cfg = Dict(
                        data_dir=test_data_dir,
                        Y_cols=['external_shear_gamma_ext', 'external_shear_psi_ext', 'c'],
                        add_noise=False,
                        )
        n_data = 9
        batch_size = 7
        X_dim = 8
        metadata = pd.DataFrame({'external_shear_gamma_ext': np.arange(n_data), 'external_shear_psi_ext': np.arange(n_data) + 1, 'c': np.arange(n_data) + 1})
        expected = log_parameterize_Y_cols(metadata.copy(), data_cfg.Y_cols_to_log_parameterize)[data_cfg.Y_cols_to_whiten].values
        data_cfg.train_Y_mean = np.mean(expected, axis=0, keepdims=True)
        data_cfg.train_Y_std = np.std(expected, axis=0, keepdims=True)
        for i in range(n_data):
            img_filename = 'X_{0:07d}.npy'.format(i)
            metadata.loc[i, 'img_filename'] = img_filename
            img = np.arange(X_dim**2).reshape(X_dim, X_dim)
            np.save(os.path.join(test_data_dir, img_filename), img)
        metadata.to_csv(os.path.join(test_data_dir, 'metadata.csv'), index=None)

        # Instantiate Dataset, DataLoader
        data = XYData(data_cfg.data_dir, data_cfg=data_cfg)
        loader = DataLoader(data, batch_size=batch_size, shuffle=False, drop_last=True)

        # Check Y_df shape
        np.testing.assert_array_equal(data.Y_df.values.shape, [n_data, len(data_cfg.Y_cols) + 1])
        # Test DataLoader iterator
        for batch_idx, (X_, Y_) in enumerate(loader):
            X = X_.numpy()
            Y = Y_.numpy()
            # Check shapes for X, Y
            np.testing.assert_array_equal(X.shape, [batch_size, 3, X_dim, X_dim], "X shape")
            np.testing.assert_array_equal(Y.shape, [batch_size, len(data_cfg.Y_cols)], "Y shape")
            # Check values for X
            single_img = np.arange(X_dim**2).reshape(X_dim, X_dim)
            X_numpy = (single_img - single_img.min())/(single_img.max() - single_img.min())
            X_numpy = np.stack([X_numpy]*3, axis=0)
            X_numpy = (X_numpy - np.array([0.485, 0.456, 0.406]).reshape(-1, 1, 1))
            X_numpy /= np.array([0.229, 0.224, 0.225]).reshape(-1, 1, 1)
            X_numpy = np.stack([X_numpy]*batch_size, axis=0)
            np.testing.assert_array_almost_equal(X, X_numpy, err_msg="X values")
            # Check values for Y
            Y_numpy = np.empty((batch_size, len(data_cfg.Y_cols)))
            Y_numpy[:, 0] = (np.arange(batch_size) - np.mean(np.arange(n_data)))/np.std(np.arange(n_data)) # only whiten
            Y_numpy[:, 1] = (np.log(np.arange(batch_size) + 1) - np.mean(np.log(np.arange(n_data) + 1)))/np.std(np.log(np.arange(n_data) + 1)) # log and whiten
            Y_numpy[:, 2] = np.log(np.arange(batch_size) + 1) # only log
            np.testing.assert_array_almost_equal(Y, Y_numpy, err_msg="Y_values")
            break
        shutil.rmtree(test_data_dir)

if __name__ == '__main__':
    unittest.main()