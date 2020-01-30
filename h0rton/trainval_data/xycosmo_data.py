import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from baobab.data_augmentation import NoiseModelTorch
from baobab.sim_utils import add_g1g2_columns
from .data_utils import rescale_01, stack_rgb, log_parameterize_Y_cols, whiten_Y_cols

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
        rescale = transforms.Lambda(rescale_01)
        stack = transforms.Lambda(stack_rgb)
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], inplace=True)
        self.X_transform = transforms.Compose([rescale, stack, normalize])
        #self.Y_transform = torch.Tensor
        # Y metadata
        metadata_path = os.path.join(self.dataset_dir, 'metadata.csv')
        metadata_df = pd.read_csv(metadata_path, index_col=False, converters={'measured_td': eval})
        metadata_df = add_g1g2_columns(metadata_df)
        # Define source light position as offset from lens mass
        metadata_df['src_light_center_x'] = metadata_df['src_light_center_x'] - metadata_df['lens_mass_center_x']
        metadata_df['src_light_center_y'] = metadata_df['src_light_center_y'] - metadata_df['lens_mass_center_y']
        # Take only the columns we need
        self.Y_df = metadata_df[self.Y_cols + ['img_filename']].copy()
        # Cosmology-related metadata we need for H0 inference
        self.cosmo_df = metadata_df[['z_lens', 'z_src', 'H0', 'x_image_0', 'x_image_1', 'x_image_2', 'x_image_3', 'y_image_0', 'y_image_1', 'y_image_2', 'y_image_3', 'true_vd', 'true_td']].copy()
        del metadata_df
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