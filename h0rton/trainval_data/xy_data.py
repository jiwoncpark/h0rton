import os
import glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from baobab import BaobabConfig
from baobab.data_augmentation.noise_torch import NoiseModelTorch
from baobab.sim_utils import add_g1g2_columns
from .data_utils import whiten_pixels, rescale_01, plus_1_log, whiten_Y_cols

__all__ = ['XYData']

class XYData(Dataset): # torch.utils.data.Dataset
    """Represents the XYData used to train or validate the BNN

    """
    def __init__(self, is_train, Y_cols, float_type, define_src_pos_wrt_lens, rescale_pixels, log_pixels, add_pixel_noise, eff_exposure_time, train_Y_mean=None, train_Y_std=None, train_baobab_cfg_path=None, val_baobab_cfg_path=None, for_cosmology=False, rescale_pixels_type='whiten_pixels'):
        """
        Parameters
        ----------
        dataset_dir : str or os.path object
            path to the directory containing the images and metadata
        data_cfg : dict or Dict
            copy of the `data` field of `BNNConfig`
        for_cosmology : bool
            whether the dataset will be used in cosmological inference 
            (in which case cosmology-related metadata will be stored)

        """
        #self.__dict__ = data_cfg.deepcopy()
        self.is_train = is_train
        if self.is_train:
            self.baobab_cfg = BaobabConfig.from_file(train_baobab_cfg_path)
        else:
            self.baobab_cfg = BaobabConfig.from_file(val_baobab_cfg_path)
        self.dataset_dir = self.baobab_cfg.out_dir
        if not self.is_train:
            if train_Y_mean is None or train_Y_std is None:
                raise ValueError("Mean and std of training set must be provided for whitening.")
        self.train_Y_mean = train_Y_mean
        self.train_Y_std = train_Y_std
        self.Y_cols = Y_cols
        self.float_type = float_type
        self.float_type_numpy = np.float64 if 'Double' in float_type else np.float32
        self.define_src_pos_wrt_lens = define_src_pos_wrt_lens
        self.rescale_pixels = rescale_pixels
        self.log_pixels = log_pixels
        self.add_pixel_noise = add_pixel_noise
        self.eff_exposure_time = eff_exposure_time
        self.bandpass_list = self.baobab_cfg.survey_info.bandpass_list
        self.for_cosmology = for_cosmology
        
        #################
        # Target labels #
        #################
        metadata_path = os.path.join(self.dataset_dir, 'metadata.csv')
        Y_df = pd.read_csv(metadata_path, index_col=False)
        if 'external_shear_gamma1' not in Y_df.columns: # assumes gamma_ext, psi_ext were sampled
            Y_df = add_g1g2_columns(Y_df)
        # Define source light position as offset from lens mass
        if self.define_src_pos_wrt_lens:
            Y_df['src_light_center_x'] -= Y_df['lens_mass_center_x']
            Y_df['src_light_center_y'] -= Y_df['lens_mass_center_y']
        train_Y_to_whiten = Y_df[self.Y_cols].values
        if self.is_train:
            self.train_Y_mean = np.mean(train_Y_to_whiten, axis=0, keepdims=True)
            self.train_Y_std = np.std(train_Y_to_whiten, axis=0, keepdims=True)
        # Store the unwhitened metadata
        if self.for_cosmology:
            self.Y_df = Y_df.copy()        
        # Number of predictive columns
        self.Y_dim = len(self.Y_cols)
        # Whiten the columns
        whiten_Y_cols(Y_df, self.train_Y_mean, self.train_Y_std, self.Y_cols)
        # Convert into array the columns required for training
        self.img_filenames = Y_df['img_filename'].values
        self.Y_array = Y_df[self.Y_cols].values.astype(self.float_type_numpy)
        # Free memory
        if not self.for_cosmology:
            del Y_df
        ################
        # Input images #
        ################
        # Set some metadata
        img_path = glob.glob(os.path.join(self.dataset_dir, '*.npy'))[0]
        img = np.load(img_path)
        self.X_dim = img.shape[0]

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
        # Noise-related kwargs
        self.noise_kwargs = {}
        self.noiseless_exposure_time = {}
        self.noise_model = {}
        self.exposure_time_factor = np.ones([len(self.bandpass_list), 1, 1]) # for broadcasting
        for i, bp in enumerate(self.bandpass_list):
            survey_object = self.baobab_cfg.survey_object_dict[bp]
            # Dictionary of SingleBand kwargs
            self.noise_kwargs[bp] = survey_object.kwargs_single_band()
            # Factor of effective exptime relative to exptime of the noiseless images
            self.exposure_time_factor[i, :, :] = self.eff_exposure_time[bp]/self.noise_kwargs[bp]['exposure_time']
            if self.add_pixel_noise:
                self.noise_kwargs[bp].update(exposure_time=self.eff_exposure_time[bp])
                # Dictionary of noise models
                self.noise_model[bp] = NoiseModelTorch(**self.noise_kwargs[bp])

    def __getitem__(self, index):
        # Image X
        img_filename = self.img_filenames[index]
        img_path = os.path.join(self.dataset_dir, img_filename)
        img = np.load(img_path)
        img *= self.exposure_time_factor
        img = torch.as_tensor(img.astype(self.float_type_numpy)) # np array type must match with default tensor type
        if self.add_pixel_noise:
            for i, bp in enumerate(self.bandpass_list):
                img[i, :, :] += self.noise_model[bp].get_noise_map(img[i, :, :])
        img = self.X_transform(img)
        # Label Y
        Y_row = self.Y_array[index, :]
        Y_row = torch.as_tensor(Y_row)
        return img, Y_row

    def __len__(self):
        return self.Y_array.shape[0]