import os, sys
import numpy as np
import pandas as pd
import astropy.io.fits as pyfits
from scipy import ndimage
from torch.utils.data import Dataset
import torchvision.transforms
import torch

__all__ = ['XYData', 'XData', 'get_X_normalizer']

#from PIL import Image

def get_X_normalizer(normalize_pixels, mean_pixels, std_pixels):
    """Instantiate a normalizer for the pixels in images X
    
    Parameters
    ----------
    normalize_pixels : bool
        whether to normalize the pixels
    mean_pixels : array-like
        each element is the mean of pixel values for that filter
    std_pixels : array-like
        each element is the std of pixel values for that filter

    Returns
    -------
    torchvision.transforms.Compose object
        composition of transforms for X normalization

    """
    if normalize_pixels:
        normalize = torchvision.transforms.Normalize(mean=mean_pixels, std=std_pixels)
        X_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), normalize])
    else:
        X_transform = torch.Tensor
    return X_transform

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
        self.X_transform = get_X_normalizer(self.normalize_pixels, self.mean_pixels, self.std_pixels)
        # Y metadata
        metadata_path = os.path.join(self.dataset_dir, 'metadata.csv')
        self.Y_df = pd.read_csv(metadata_path, index_col=False)[self.Y_cols + ['img_filename']]
        if len(self.Y_cols_to_log_parameterize) > 0:
            self.log_parameterize_Y_cols()
        if len(self.Y_cols_to_whiten) > 0:
            self.whiten_Y_cols()
        self.Y_transform = torch.Tensor

    def log_parameterize_Y_cols(self):
        """Parameterize user-defined Y_cols in terms of their log

        """
        self.Y_df.loc[:, self.Y_cols_to_log_parameterize] = np.log(self.Y_df.loc[:, self.Y_cols_to_log_parameterize].values)

    def whiten_Y_cols(self):
        """Whiten user-defined Y_cols, i.e. shift and scale them so their mean is 0 and std is 1

        """
        self.Y_df.loc[:, self.Y_cols_to_whiten] = (self.Y_df.loc[:, self.Y_cols_to_whiten].values - self.Y_mean)/self.Y_std

    def __getitem__(self, index):
        img_filename = self.Y_df.iloc[index]['img_filename']
        img_path = os.path.join(self.dataset_dir, img_filename)
        img = np.load(img_path)
        img = ndimage.zoom(img, self.X_dim/self.raw_X_dim, order=1) # TODO: consider order=3
        img = np.stack([img]*self.n_filters, axis=2).astype(np.float32)
        Y_row = self.Y_df.iloc[index][self.Y_cols].values.astype(np.float32)
        # Transformations
        img = self.X_transform(img)
        Y_row = self.Y_transform(Y_row)

        return img, Y_row

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
        self.X_transform = get_X_normalizer(self.normalize_pixels, self.mean_pixels, self.std_pixels)

    def __getitem__(self, index):
        hdul = pyfits.open(self.img_paths[index])
        img = hdul['PRIMARY'].data
        img = ndimage.zoom(img, self.X_dim/self.raw_X_dim, order=1) # TODO: consider order=3
        img = np.stack([img]*self.n_filters, axis=2).astype(np.float32)
        # Transformations
        img = self.X_transform(img)

        return img

    def __len__(self):
        return self.n_data

if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from torch.utils.data.sampler import SubsetRandomSampler
    import torchvision.transforms as transforms
    import torch
    #from h0rton.trainval_data import train_tdlmc_diagonal_config
    #from baobab import Config as BaobabConfig
    from h0rton.configs import BNNConfig
    from h0rton.example_user_config import cfg as user_cfg

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    X_transform = transforms.Compose([transforms.ToTensor(), normalize])

    #baobab_cfg = BaobabConfig.fromfile(train_tdlmc_diagonal_config.__file__)
    cfg = BNNConfig(user_cfg)
    data = XYData(cfg.data.train_dir, data_cfg=cfg.data)
    loader = DataLoader(data, batch_size=20, shuffle=False, num_workers=0)

    subset_sampler = SubsetRandomSampler(np.arange(20))
    subset_loader = DataLoader(data, batch_size=20,
                               sampler=subset_sampler)

    # Test plotting data loader
    if False:
        for batch_idx, (X_, Y_) in enumerate(subset_loader):
            print(X_.shape)
            print(Y_.shape)

    if True:
        for batch_idx, (X_, Y_) in enumerate(loader):
            #if xy_batch[0].shape[0] != 20:
            #    print(len(xy_batch)) # should be 2, x and y
            #    print(xy_batch[0].shape) # X shape
            #    print(xy_batch[1].shape) # Y shape
            print(X_.shape) # X shape
            print(Y_.shape) # Y shape
            if batch_idx == 3:
                break