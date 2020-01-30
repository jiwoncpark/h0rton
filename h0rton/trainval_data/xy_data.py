import os
import numpy as np
import pandas as pd
import astropy.io.fits as pyfits
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from baobab.data_augmentation import NoiseModelTorch
from baobab.sim_utils import add_g1g2_columns
from .data_utils import rescale_01, stack_rgb, log_parameterize_Y_cols, whiten_Y_cols

__all__ = ['XYData', 'XData',]

class XYData(Dataset): # torch.utils.data.Dataset
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
        rescale = transforms.Lambda(rescale_01)
        stack = transforms.Lambda(stack_rgb)
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], inplace=True)
        self.X_transform = transforms.Compose([rescale, stack, normalize])
        #self.Y_transform = torch.Tensor
        # Y metadata
        metadata_path = os.path.join(self.dataset_dir, 'metadata.csv')
        Y_df = pd.read_csv(metadata_path, index_col=False)
        Y_df = add_g1g2_columns(Y_df)
        # Define source light position as offset from lens mass
        Y_df['src_light_center_x'] = Y_df['src_light_center_x'] - Y_df['lens_mass_center_x']
        Y_df['src_light_center_y'] = Y_df['src_light_center_y'] - Y_df['lens_mass_center_y']
        # Take only the columns we need
        self.Y_df = Y_df[self.Y_cols + ['img_filename']].copy()
        # Size of dataset
        self.n_data = self.Y_df.shape[0]
        # Number of predictive columns
        self.Y_dim = len(self.Y_cols)
        # Log parameterizing
        if len(self.Y_cols_to_log_parameterize) > 0:
            self.Y_df = log_parameterize_Y_cols(self.Y_df, self.Y_cols_to_log_parameterize)
        # Whitening
        if len(self.Y_cols_to_whiten) > 0:
            self.Y_df = whiten_Y_cols(self.Y_df, self.Y_cols_to_whiten, self.train_Y_mean, self.train_Y_std)
        if self.add_noise:
            self.noise_model = NoiseModelTorch(**data_cfg.noise_kwargs)

    def __getitem__(self, index):
        # Image X
        img_filename = self.Y_df.iloc[index]['img_filename']
        img_path = os.path.join(self.dataset_dir, img_filename)
        img = np.load(img_path)
        img = torch.as_tensor(img.astype(np.float32)) # np array type must match with default tensor type
        if self.add_noise:
            img += self.noise_model.get_noise_map(img)
        img = self.X_transform(img)
        # Label Y
        Y_row = self.Y_df.iloc[index][self.Y_cols].values.astype(np.float32)
        Y_row = torch.as_tensor(Y_row)
        return img, Y_row

    def __len__(self):
        return self.n_data

class XData(Dataset): # torch.utils.data.Dataset
    """Represents the XData used to test the BNN

    """
    def __init__(self, img_paths, data_cfg):
        """
        Parameters
        ----------
        img_paths : list
            list of image paths. Indexing is based on order in this list.
        data_cfg : dict or Dict
            copy of the `data` field of `BNNConfig`

        """
        self.__dict__ = data_cfg
        self.img_paths = img_paths
        self.X_transform = torch.Tensor

    def __getitem__(self, index):
        hdul = pyfits.open(self.img_paths[index])
        img = hdul['PRIMARY'].data
        img = np.stack([img]*self.n_filters, axis=0).astype(np.float32)
        # Transformations
        img = self.X_transform(img)

        return img

    def __len__(self):
        return self.n_data