from torch.utils.data import Dataset
import torch
import numpy as np
import pandas as pd
from scipy import ndimage
from PIL import Image

class XYData(Dataset): # torch.utils.data.Dataset
    def __init__(self, dataset_dir, Y_cols, X_transform=None, Y_transform=None, interpolation=224):
        self.dataset_dir = dataset_dir
        self.Y_cols = Y_cols
        self.X_transform = X_transform
        self.Y_transform = Y_transform
        #self.df = pd.read_csv('../input/clean-full-train/clean_full_data.csv') #+ '/clean_full_data.csv')

        self.df = pd.read_csv(self.dataset_dir + '/metadata.csv')
        self.interpolation = interpolation
        self.n_data = self.df.shape[0]

    def __getitem__(self, index):
        img_path = self.df.iloc[index]['img_path']
        img = np.load(img_path)
        img = ndimage.zoom(img, self.interpolation/100, order=1) # TODO: consider order=3
        img = np.stack([img, img, img], axis=2).astype(np.float32)
        #img = Image.fromarray(img)
        #img = img.transpose(0, 2, 3, 1)
        #img = Image.fromarray(img)

        Y_row = self.df.iloc[index][self.Y_cols].values.astype(np.float32)

        if self.X_transform is not None:
            img = self.X_transform(img)
        if self.Y_transform is not None:
            Y_row = self.Y_transform(Y_row)

        return img, Y_row

    def __len__(self):
        return self.n_data

if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from torch.utils.data.sampler import SubsetRandomSampler
    import torchvision.transforms as transforms
    import torch

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    X_transform = transforms.Compose([transforms.ToTensor(), normalize])

    data = XYData('data/tdlmc_train_DiagonalBNNPrior_seed1113',
                  Y_cols=['lens_mass_theta_E', 'lens_mass_gamma'],
                  X_transform=X_transform, Y_transform=None, interpolation=224)
    loader = DataLoader(data, batch_size=20, shuffle=False)

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
            break