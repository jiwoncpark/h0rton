import numpy as np
import h0rton.h0_inference.mcmc_utils as mcmc_utils
import unittest

class TestMCMCUtils(unittest.TestCase):
    """A suite of tests verifying that the input PDFs and the sample distributions
    match.
    
    """

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

if __name__ == '__main__':
    unittest.main()