import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from astropy.io import fits
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from baobab.data_augmentation.noise_torch import NoiseModelTorch
from baobab.sim_utils import add_g1g2_columns
from .data_utils import whiten_pixels, plus_1_log, whiten_Y_cols
import h0rton.tdlmc_utils
import h0rton.tdlmc_data

__all__ = ['TDLMCData',]

class TDLMCData(Dataset): # torch.utils.data.Dataset
    """Represents the XYData used to train or validate the BNN

    """
    def __init__(self, data_cfg, rung_i=2):
        """
        Parameters
        ----------
        dataset_dir : str or os.path object
            path to the directory containing the images and metadata
        data_cfg : dict or Dict
            copy of the `data` field of `BNNConfig`

        """
        self.__dict__ = data_cfg
        self.img_dir = os.path.join(h0rton.tdlmc_data.__path__[0], 'rung{:d}'.format(rung_i))
        self.img_paths = np.sort(list(Path(self.img_dir).rglob('*drizzled_image/lens-image.fits')))
        # Rescale pixels, stack filters, and shift/scale pixels on the fly 
        rescale = transforms.Lambda(whiten_pixels)
        log = transforms.Lambda(plus_1_log)
        transforms_list = []
        if self.log_pixels:
            transforms_list.append(log)
        if self.rescale_pixels:
            transforms_list.append(rescale)
        if len(transforms_list) == 0:
            self.X_transform = None
        else:
            self.X_transform = transforms.Compose(transforms_list)
        # Y metadata
        self.cosmo_df = h0rton.tdlmc_utils.convert_to_dataframe(rung=rung_i, save_csv_path=None)
        self.cosmo_df.sort_values('seed', axis=0, inplace=True)
        # Size of dataset
        self.n_data = self.cosmo_df.shape[0]
        # Number of predictive columns
        self.Y_dim = len(self.Y_cols)
        # Adjust exposure time relative to that used to generate the noiseless images
        self.exposure_time_factor = self.noise_kwargs.exposure_time/9600.0
        if self.add_noise:
            self.noise_model = NoiseModelTorch(**self.noise_kwargs)

    def __getitem__(self, index):
        # Image X
        img_path = self.img_paths[index]
        img = fits.getdata(img_path, ext=0)
        img *= self.exposure_time_factor
        img = img[17:-18, 17:-18] # Hacky clipping to preserve pixel scale and resize 99 x 99 to 64 x 64 
        img = torch.as_tensor(img.astype(np.float32)) # np array type must match with default tensor type
        if self.add_noise:
            img += self.noise_model.get_noise_map(img)
        img = self.X_transform(img).unsqueeze(0)
        return img

    def __len__(self):
        return self.n_data