import os, sys
import numpy as np
import unittest

class TestBNNConfig(unittest.TestCase):
    """A suite of tests verifying that the input PDFs and the sample distributions
    match.
    
    """

    def test_bnn_config(self):
        """Test the instantiation of BNNConfig

        """
        from h0rton.configs import BNNConfig
        from h0rton.example_user_config import cfg
        bnn_cfg = BNNConfig(cfg)

if __name__ == '__main__':
    unittest.main()