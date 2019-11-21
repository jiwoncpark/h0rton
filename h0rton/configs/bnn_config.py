import os, sys
import importlib
import warnings
import numpy as np
import pandas as pd
import torch
from baobab import BaobabConfig
from addict import Dict

class BNNConfig:
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
        self.set_device()
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
        #user_cfg_file = map(__import__, module_name)
        #user_cfg = getattr(user_cfg_file, 'cfg')
        user_cfg_script = importlib.import_module(module_name)
        user_cfg = getattr(user_cfg_script, 'cfg')
        return cls(user_cfg)

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
            self.data.train_dir = None
        if 'val_dir' not in self.data:
            self.data.val_dir = None
        if self.data.normalize_pixels and not ('mean_pixels' in self.data and 'std_pixels' in self.data):
            raise ValueError("Since data.normalize_pixels is True, please supply data.mean_pixels and data.std_pixels.")
        if len(self.data.mean_pixels) != len(self.data.std_pixels):
            raise ValueError("Lengths of data.mean_pixels and data.std_pixels should be equal.")

    def set_device(self):
        """Configure the device to use for training

        Note
        ----
        Disabling fallback to cpu when cuda is unavailable, for full reproducibility.

        """
        # Disable this check for reproducibility
        #self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = torch.device(self.device_type)
        if self.device.type == 'cuda':
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            torch.set_default_tensor_type('torch.FloatTensor')

    def set_baobab_metadata(self):
        """Migrate some of the metadata in the Baobab configs and check that they are reasonable

        """
        self.data.train_baobab_cfg = BaobabConfig.from_file(self.data.train_baobab_cfg_path)
        self.data.val_baobab_cfg = BaobabConfig.from_file(self.data.val_baobab_cfg_path)
        if self.data.train_dir is None:
            self.data.train_dir = self.data.train_baobab_cfg.out_dir
        if self.data.val_dir is None:
            self.data.val_dir = self.data.val_baobab_cfg.out_dir
        metadata_path = os.path.join(self.data.val_dir, 'metadata.csv')
        Y_df = pd.read_csv(metadata_path, index_col=False)
        img_path = os.path.join(self.data.val_dir, Y_df.iloc[0]['img_filename'])
        img = np.load(img_path)
        self.data.n_data = Y_df.shape[0]
        self.data.raw_X_dim = img.shape[0]
        original = Y_df.loc[:, self.data.Y_cols_to_whiten].values
        self.data.Y_mean = np.mean(original, axis=0, keepdims=True)
        self.data.Y_std = np.std(original, axis=0, keepdims=True)
        del Y_df, original # FIXME: bad, not sure if necessary
        self.check_train_val_diff()

    def set_XY_metadata(self):
        """Set general metadata relevant to network architecture and optimization

        """
        self.data.Y_dim = len(self.data.Y_cols)
        self.data.n_filters = len(self.data.mean_pixels)
        Y_col_idx_mapping = dict(zip(self.data.Y_cols, range(self.data.Y_dim)))
        self.data.Y_cols_to_log_parameterize_idx = list(map(Y_col_idx_mapping.get, self.data.Y_cols_to_log_parameterize))
        self.data.Y_cols_to_whiten_idx = list(map(Y_col_idx_mapping.get, self.data.Y_cols_to_whiten))

    def set_model_metadata(self):
        """Set metadata about the network architecture and the loss function (posterior type)

        """
        if self.model.load_pretrained:
            # Pretrained model expects exactly this normalization
            self.data.mean_pixels = [0.485, 0.456, 0.406]
            self.data.std_pixels = [0.229, 0.224, 0.225]

    def check_train_val_diff(self):
        """Check that the training and validation datasets are different

        """
        if self.data.train_dir == self.data.val_dir:
            warnings.warn("You're training and validating on the same dataset.", UserWarning, stacklevel=2)
        if self.data.train_baobab_cfg.seed == self.data.val_baobab_cfg.seed:
            warnings.warn("The training and validation datasets were generated using the same seed.", UserWarning, stacklevel=2)