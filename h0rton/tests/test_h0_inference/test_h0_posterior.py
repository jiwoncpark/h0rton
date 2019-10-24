import os, sys
import numpy as np
import unittest

class TestH0Posterior(unittest.TestCase):
    """A suite of tests verifying that the input PDFs and the sample distributions
    match.
    
    """

    def test_gaussian_ll_pdf(self):
        """Test the log normal PDF

        """
        from h0rton.h0_inference import gaussian_ll_pdf
        from scipy.stats import norm
        m = 0.5
        s = 1
        eval_x = np.linspace(-3, 3, 100)
        truth = np.log(norm.pdf(eval_x, m, s))
        pred = gaussian_ll_pdf(eval_x, m, s)
        np.testing.assert_array_almost_equal(pred, truth)

if __name__ == '__main__':
    unittest.main()