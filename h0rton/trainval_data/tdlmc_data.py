import os
from pathlib import Path
import numpy as np
import glob
import torch
from astropy.io import fits
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from .data_utils import whiten_pixels, rescale_01, plus_1_log
import h0rton.tdlmc_utils as tdlmc_utils
import h0rton.tdlmc_data as tdlmc_data

__all__ = ['TDLMCData',]

class TDLMCData(Dataset): # torch.utils.data.Dataset
    """Represents the XYData used to train or validate the BNN

    Note
    ----
    Meant to evaluate a model trained on 9600s, noised data transformed via
    appropriate rescale and log operations

    """
    def __init__(self, float_type, rescale_pixels, log_pixels, 
                 rescale_pixels_type='whiten_pixels', rung_i=2):
        """
        Parameters
        ----------
        float_type : str
            float type of image and target labels ('double' or 'float')
        log_pixels : bool
            whether to log1p() pixels before rescaling, if any
        rescale_pixels : bool
            whether to rescale pixels
        rescale_pixels_type : str
            specific rescaling operation 
            ('whiten_pixels' or 'rescale_01' supported)
        rung_i : int
            ID of the TDLMC Rung

        """
        self.img_dir = os.path.join(tdlmc_data.__path__[0], 
                                    'rung{:d}'.format(rung_i))
        img_wildcard = '*drizzled_image/lens-image.fits'
        self.img_paths = list(Path(self.img_dir).rglob(img_wildcard))
        self.img_paths.sort()
        self.float_type = float_type
        if 'double' in self.float_type.lower():
            self.float_type_numpy = np.float64
        else:
            self.float_type_numpy = np.float32
        self.rescale_pixels = rescale_pixels
        self.log_pixels = log_pixels
        ################
        # Input images #
        ################
        # Rescale pixels, stack filters, and shift/scale pixels on the fly 
        if rescale_pixels_type == 'rescale_01':
            rescale = transforms.Lambda(rescale_01)
        else:
            rescale = transforms.Lambda(whiten_pixels)
        log = transforms.Lambda(plus_1_log)
        transforms_list = []
        if self.log_pixels:
            transforms_list.append(log)
        if self.rescale_pixels:
            transforms_list.append(rescale)
        if len(transforms_list) == 0:
            self.X_transform = lambda x: x
        else:
            self.X_transform = transforms.Compose(transforms_list)
        # Y metadata
        self.cosmo_df = tdlmc_utils.convert_to_dataframe(rung=rung_i, 
                                                         save_csv_path=None)
        self.cosmo_df.sort_values('seed', axis=0, inplace=True)
        # Size of dataset
        self.n_data = self.cosmo_df.shape[0]

    def __getitem__(self, index):
        # Image X
        img_path = self.img_paths[index]
        img = fits.getdata(img_path, ext=0)
        # Hacky clipping to preserve pixel scale and resize 99 x 99 to 64 x 64 
        img = img[17:-18, 17:-18] 
        img = torch.as_tensor(img.astype(self.float_type_numpy))
        img = self.X_transform(img).unsqueeze(0)
        return img, torch.tensor([0])

    def __len__(self):
        return self.n_data