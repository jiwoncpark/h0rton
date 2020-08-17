import numpy as np
import unittest
from h0rton.configs import TestConfig

class TestTestConfig(unittest.TestCase):
    """A suite of tests for TrainValConfig
    
    """

    @classmethod
    def setUpClass(cls):
        cls.test_dict = dict(
                             )

    def test_test_config_constructor(self):
        test_cfg = TestConfig(self.test_dict)

if __name__ == '__main__':
    unittest.main()