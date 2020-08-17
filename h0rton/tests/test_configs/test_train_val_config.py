import numpy as np
import unittest
import copy
from h0rton.configs import TrainValConfig

class TestTrainValConfig(unittest.TestCase):
    """A suite of tests for TrainValConfig
    
    """

    @classmethod
    def setUpClass(cls):
        cls.train_val_dict = dict(
                                  data=dict(
                                            ),
                                  monitoring=dict(
                                                  n_plotting=20
                                                  ),
                                  model=dict(
                                             likelihood_class='DoubleGaussianNLL'
                                             ),
                                  optim=dict(
                                             batch_size=100
                                             )
                                  )

    def test_train_val_config_constructor(self):
        """Test the instantiation of TrainValConfig from a dictionary with minimum required keys

        """
        train_val_dict = copy.deepcopy(self.train_val_dict)
        train_val_dict['data']['train_baobab_cfg_path'] = 'some_path'
        train_val_dict['data']['val_baobab_cfg_path'] = 'some_other_path'
        train_val_cfg = TrainValConfig(train_val_dict)

    def test_train_val_absent(self):
        """Test if an error is raised when the either the train or val baobab config is not passed in

        """
        train_val_dict = copy.deepcopy(self.train_val_dict)
        train_val_dict['data']['val_baobab_cfg_path'] = 'some_path'
        with np.testing.assert_raises(ValueError):
            train_val_cfg = TrainValConfig(train_val_dict)
        train_val_dict = copy.deepcopy(self.train_val_dict)
        train_val_dict['data']['train_baobab_cfg_path'] = 'some_path'
        with np.testing.assert_raises(ValueError):
            train_val_cfg = TrainValConfig(train_val_dict)

if __name__ == '__main__':
    unittest.main()