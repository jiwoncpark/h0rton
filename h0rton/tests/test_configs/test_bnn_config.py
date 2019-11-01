import os, sys
import numpy as np
import unittest

class TestBNNConfig(unittest.TestCase):
    """A suite of tests verifying that the input PDFs and the sample distributions
    match.
    
    """

    def test_bnn_config_constructor(self):
        """Test the instantiation of BNNConfig

        """
        from h0rton.configs import BNNConfig
        from h0rton.example_user_config import cfg
        bnn_cfg = BNNConfig(cfg)

    def test_bnn_config_from_file_constructor(self):
        """Test the instantiation of BNNConfig

        """
        from h0rton.configs import BNNConfig
        import h0rton.example_user_config
        bnn_cfg = BNNConfig.from_file(h0rton.example_user_config.__file__)

if __name__ == '__main__':
    unittest.main()