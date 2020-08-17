import os, sys
import importlib
import warnings
import json
import glob
import numpy as np
import pandas as pd
from addict import Dict
from baobab.sim_utils import add_g1g2_columns

class TrainValConfig:
    """Nested dictionary representing the configuration for H0rton training, h0_inference, visualization, and analysis

    """
    def __init__(self, user_cfg):
        """
        Parameters
        ----------
        user_cfg : dict or Dict
            user-defined configuration
        
        """
        self.__dict__ = Dict(user_cfg)
        self.validate_user_definition()
        self.preset_default()
        self.set_monitoring_cfg()        

    @classmethod
    def from_file(cls, user_cfg_path):
        """Alternative constructor that accepts the path to the user-defined configuration python file

        Parameters
        ----------
        user_cfg_path : str or os.path object
            path to the user-defined configuration python file

        """
        dirname, filename = os.path.split(os.path.abspath(user_cfg_path))
        module_name, ext = os.path.splitext(filename)
        sys.path.append(dirname)
        if ext == '.py':
            #user_cfg_file = map(__import__, module_name)
            #user_cfg = getattr(user_cfg_file, 'cfg')
            user_cfg_script = importlib.import_module(module_name)
            user_cfg = getattr(user_cfg_script, 'cfg')
            return cls(user_cfg)
        elif ext == '.json':
            with open(user_cfg_path, 'r') as f:
                user_cfg_str = f.read()
            user_cfg = Dict(json.loads(user_cfg_str))
            return cls(user_cfg)
        else:
            raise NotImplementedError("This extension is not supported.")

    def validate_user_definition(self):
        """Check to see if the user-defined config is valid

        """
        import h0rton.losses
        if not hasattr(h0rton.losses, self.model.likelihood_class):
            raise TypeError("Likelihood class supplied in cfg doesn't exist.")

    def preset_default(self):
        """Preset default config values

        """
        if 'train_baobab_cfg_path' not in self.data:
            raise ValueError("Must provide training data directory.")
        if 'val_baobab_cfg_path' not in self.data:
            raise ValueError("Must provide validation data directory.")
        # FIXME: doesn't check for contents of baobab config file, just the file names
        if self.data.train_baobab_cfg_path == self.data.val_baobab_cfg_path:
            warnings.warn("You're training and validating on the same dataset.", UserWarning, stacklevel=2)
        if 'float_type' not in self.data:
            self.data.float_type = 'FloatTensor'
            warnings.warn("Float type not provided. Defaulting to float32...")

    def set_monitoring_cfg(self):
        """Set general metadata relevant to network architecture and optimization

        """
        # Data to plot during monitoring
        if self.monitoring.n_plotting > 100:
            warnings.warn("Only plotting allowed max of 100 datapoints during training")
            self.monitoring.n_plotting = 100
        if self.monitoring.n_plotting > self.optim.batch_size:
            raise ValueError("monitoring.n_plotting must be smaller than optim.batch_size")