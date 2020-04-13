import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from baobab.data_augmentation.noise_torch import NoiseModelTorch
from baobab.sim_utils import add_g1g2_columns
from .data_utils import whiten_pixels, plus_1_log, whiten_Y_cols

__all__ = ['XYCosmoData',]

class XYCosmoData(Dataset): # torch.utils.data.Dataset
    """Represents the XYData used to train or validate the BNN

    """
    def __init__(self, dataset_dir, data_cfg):
        """
        Parameters
        ----------
        dataset_dir : str or os.path object
            path to the directory containing the images and metadata
        data_cfg : dict or Dict
            copy of the `data` field of `BNNConfig`

        """
        self.__dict__ = data_cfg
        self.dataset_dir = dataset_dir
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
        metadata_path = os.path.join(self.dataset_dir, 'metadata.csv')
        metadata_df = pd.read_csv(metadata_path, index_col=False, converters={'measured_td': eval})
        metadata_df = add_g1g2_columns(metadata_df)
        # Define source light position as offset from lens mass
        if self.define_src_pos_wrt_lens:
            metadata_df['src_light_center_x'] -= metadata_df['lens_mass_center_x']
            metadata_df['src_light_center_y'] -= metadata_df['lens_mass_center_y']
        # Take only the columns we need
        self.Y_df = metadata_df[self.Y_cols + ['img_filename']].copy()
        # Cosmology-related metadata we need for H0 inference
        #self.cosmo_df = metadata_df[['z_lens', 'z_src', 'H0', 'x_image_0', 'x_image_1', 'x_image_2', 'x_image_3', 'y_image_0', 'y_image_1', 'y_image_2', 'y_image_3', 'true_vd', 'true_td',]].copy()
        self.cosmo_df = metadata_df.copy()
        del metadata_df
        # Size of dataset
        self.n_data = self.Y_df.shape[0]
        # Number of predictive columns
        self.Y_dim = len(self.Y_cols)
        self.Y_df = whiten_Y_cols(self.Y_df, self.train_Y_mean, self.train_Y_std, self.Y_cols)
        # Adjust exposure time relative to that used to generate the noiseless images
        self.exposure_time_factor = self.noise_kwargs.exposure_time/self.noiseless_exposure_time
        if self.add_noise:
            self.noise_model = NoiseModelTorch(**self.noise_kwargs)

    def __getitem__(self, index):
        # Image X
        img_filename = self.Y_df.iloc[index]['img_filename']
        img_path = os.path.join(self.dataset_dir, img_filename)
        img = np.load(img_path)
        img *= self.exposure_time_factor
        img = torch.as_tensor(img.astype(np.float32)) # np array type must match with default tensor type
        if self.add_noise:
            img += self.noise_model.get_noise_map(img)
        img = self.X_transform(img).unsqueeze(0)
        # Label Y
        Y_row = self.Y_df.iloc[index][self.Y_cols].values.astype(np.float32)
        Y_row = torch.as_tensor(Y_row)
        return img, Y_row

    def __len__(self):
        return self.n_data