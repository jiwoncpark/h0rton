import os
import numpy as np
import pandas as pd
import astropy.io.fits as pyfits
from torch.utils.data import Dataset
import torch
from baobab.data_augmentation import NoiseModelTorch
from baobab.sim_utils import add_g1g2_columns
from .data_utils import log_parameterize_Y_cols, whiten_Y_cols

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
        self.X_transform = torch.Tensor
        # Y metadata
        metadata_path = os.path.join(self.dataset_dir, 'metadata.csv')
        Y_df = pd.read_csv(metadata_path, index_col=False)
        Y_df = add_g1g2_columns(Y_df)
        self.Y_df = Y_df[self.Y_cols + ['img_filename']].copy()
        self.n_data = self.Y_df.shape[0]
        # Log parameterizing
        if len(self.Y_cols_to_log_parameterize) > 0:
            self.Y_df = log_parameterize_Y_cols(self.Y_df, self.Y_cols_to_log_parameterize)
        # Whitening
        if len(self.Y_cols_to_whiten) > 0:
            self.Y_df = whiten_Y_cols(self.Y_df, self.Y_cols_to_whiten, self.train_Y_mean, self.train_Y_std)
        self.Y_transform = torch.Tensor
        self.noise_model = NoiseModelTorch(**data_cfg.noise_kwargs)

    def __getitem__(self, index):
        img_filename = self.Y_df.iloc[index]['img_filename']
        img_path = os.path.join(self.dataset_dir, img_filename)
        img = np.load(img_path)
        img = np.stack([img]*3, axis=0)
        Y_row = self.Y_df.iloc[index][self.Y_cols].values
        # Transformations
        img = self.X_transform(img.astype(np.float32))
        img += self.noise_model.get_noise_map(img)
        Y_row = self.Y_transform(Y_row.astype(np.float32))
        return img, Y_row

    def __len__(self):
        return self.Y_df.shape[0]

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

if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from torch.utils.data.sampler import SubsetRandomSampler
    import torchvision.transforms as transforms
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