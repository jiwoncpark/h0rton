import os, sys
import importlib
import warnings
import json
import glob
import numpy as np
import pandas as pd
from baobab import BaobabConfig
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
        # Data
        self.set_baobab_metadata()
        self.set_XY_metadata()        
        self.set_model_metadata()

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

    def load_baobab_log(self, baobab_out_dir):
        """Load the baobab log

        Parameters
        ----------
        baobab_out_dir : str or os.path object
            path to the baobab output directory

        Returns
        -------
        baobab.BaobabConfig object
            log of the baobab-generated dataset, including the input config

        """
        baobab_log_path = glob.glob(os.path.join(baobab_out_dir, 'log_*_baobab.json'))[0]
        with open(baobab_log_path, 'r') as f:
            log_str = f.read()
        baobab_cfg = BaobabConfig(Dict(json.loads(log_str)))
        return baobab_cfg

    def validate_user_definition(self):
        """Check to see if the user-defined config is valid

        """
        import h0rton.losses
        if not hasattr(h0rton.losses, self.model.likelihood_class):
            raise TypeError("Likelihood class supplied in cfg doesn't exist.")

    def preset_default(self):
        """Preset default config values

        """
        if 'train_dir' not in self.data:
            raise ValueError("Must provide training data directory.")
        if 'val_dir' not in self.data:
            raise ValueError("Must provide validation data directory.")

    def set_baobab_metadata(self):
        """Migrate some of the metadata in the Baobab configs and check that they are reasonable

        """
        self.data.train_baobab_cfg = self.load_baobab_log(self.data.train_dir)
        self.data.val_baobab_cfg = self.load_baobab_log(self.data.val_dir)
        img_path = glob.glob(os.path.join(self.data.val_dir, '*.npy'))[0]
        img = np.load(img_path)
        self.data.raw_X_dim = img.shape[0]
        # TODO: update pixel_scale, exposure_time, num_exposures, mag zero point from baobab cfg
        self.check_train_val_diff()

    def set_XY_metadata(self):
        """Set general metadata relevant to network architecture and optimization

        """
        # Y metadata
        self.data.Y_dim = len(self.data.Y_cols)
        # Get training-set mean and std for whitening
        train_metadata_path = os.path.join(self.data.train_dir, 'metadata.csv')
        train_Y_to_whiten = pd.read_csv(train_metadata_path, index_col=None)
        train_Y_to_whiten = add_g1g2_columns(train_Y_to_whiten)[self.data.Y_cols].values
        self.data.train_Y_mean = np.mean(train_Y_to_whiten, axis=0, keepdims=True)
        self.data.train_Y_std = np.std(train_Y_to_whiten, axis=0, keepdims=True)
        del train_Y_to_whiten # not sure if necessary
        # Data to plot during monitoring
        if self.monitoring.n_plotting > 100:
            warnings.warn("Only plotting allowed max of 100 datapoints during training")
            self.monitoring.n_plotting = 100
        if self.monitoring.n_plotting > self.optim.batch_size:
            raise ValueError("monitoring.n_plotting must be smaller than optim.batch_size")
        # Import relevant noise-related detector and observation conditions from baobab
        if self.data.add_noise:
            self.data.noise_kwargs.update(self.data.train_baobab_cfg.instrument)
            self.data.noise_kwargs.update(self.data.train_baobab_cfg.bandpass)
            self.data.noise_kwargs.update(self.data.train_baobab_cfg.observation)
            self.data.noise_kwargs.update(psf_type='GAUSSIAN', # noise module doesn't actually use the PSF. "PIXEL", if used to generate the training set, is not an option.
                                          kernel_point_source=None,
                                          data_count_unit='e-',
                                          )
            print(self.data.noise_kwargs)

    def set_model_metadata(self):
        """Set metadata about the network architecture and the loss function (posterior type)

        """
        pass

    def check_train_val_diff(self):
        """Check that the training and validation datasets are different

        """
        if self.data.train_dir == self.data.val_dir:
            warnings.warn("You're training and validating on the same dataset.", UserWarning, stacklevel=2)
        if self.data.train_baobab_cfg.seed == self.data.val_baobab_cfg.seed:
            warnings.warn("The training and validation datasets were generated using the same seed.", UserWarning, stacklevel=2)