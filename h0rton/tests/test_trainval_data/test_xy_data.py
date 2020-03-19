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
        pass

if __name__ == '__main__':
    unittest.main()