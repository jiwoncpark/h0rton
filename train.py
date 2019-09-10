import random
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import torch
import torch.nn as nn
import torch.nn.functional as F
from  torch.autograd import Variable
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.transforms as transforms
import torchvision.models as models
import torchvision.utils as vutils
#import lenstronomy.Util.image_util as image_util
import os, sys
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import datetime

from data.data_io import XYData
from utils.config import cfg
from utils.loss import GaussianNLL
from utils.plotting import Plotter

# Seed everything for reproducibility
np.random.seed(cfg.global_seed)
random.seed(cfg.global_seed)
torch.manual_seed(cfg.global_seed)
torch.cuda.manual_seed(cfg.global_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if cfg.data.normalize:
    normalize = transforms.Normalize(mean=cfg.data.mean, std=cfg.data.std)
    X_transform = transforms.Compose([transforms.ToTensor(), normalize])
else:
    X_transform = None
Y_transform = torch.Tensor

if cfg.device == 'cuda':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

# Define training data and loader
train_data = XYData(cfg.data.train_dir, Y_cols=cfg.data.Y_cols, X_transform=X_transform, Y_transform=Y_transform, interpolation=cfg.data.X_dim)
train_loader = DataLoader(train_data, batch_size=cfg.optim.batch_size, shuffle=True)
n_train = train_data.n_data

# Define val data and loader
val_data = XYData(cfg.data.val_dir, Y_cols=cfg.data.Y_cols, X_transform=X_transform, Y_transform=Y_transform, interpolation=cfg.data.X_dim)
val_loader = DataLoader(val_data, batch_size=cfg.optim.batch_size, shuffle=True)
n_val = val_data.n_data

# Define plotting data (subset of val data) and loader
plot_data_sampler = SubsetRandomSampler(cfg.data.plot_idx)
plot_data_loader = DataLoader(val_data, batch_size=len(cfg.data.plot_idx), sampler=plot_data_sampler)

# Define plotter object
plotter = Plotter(cfg.model.type, cfg.data.Y_dim, cfg.device)

if __name__ == '__main__':
    # Instantiate model
    net = models.resnet18(pretrained=cfg.model.load_pretrained)
    n_filters = net.fc.in_features # number of output nodes in 2nd-to-last layer
    net.fc = nn.Linear(in_features=n_filters, out_features=cfg.model.out_dim) # replace final layer
    net.cuda()

    # Instantiate loss function
    loss_fn = GaussianNLL(cov_mat=cfg.model.type, Y_dim=cfg.data.Y_dim, out_dim=cfg.model.out_dim, device=cfg.device)

    optimizer = optim.Adam(net.parameters(), lr=cfg.optim.learning_rate, amsgrad=True)
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg.optim.lr_scheduler.milestones, gamma=cfg.optim.lr_scheduler.gamma)
    logger = SummaryWriter()

    if not os.path.exists(cfg.log.checkpoint_dir):
        os.mkdir(cfg.log.checkpoint_dir)
        #net = torch.load('./saved_model/resnet18.mdl')
        #print('loaded mdl!')

    progress = tqdm(range(cfg.optim.n_epochs))
    for epoch in progress:
        net.train()
        total_loss = 0.0

        for batch_idx, (X_, Y_) in enumerate(train_loader):
            X = Variable(torch.FloatTensor(X_)).to(cfg.device)
            Y = Variable(torch.FloatTensor(Y_)).to(cfg.device)

            pred = net(X)
            loss = loss_fn(pred, Y)
            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

        with torch.no_grad():
            net.eval()
            total_val_loss = 0.0

            for batch_idx, (X_, Y_) in enumerate(val_loader):
                X = Variable(torch.FloatTensor(X_)).to(cfg.device)
                Y = Variable(torch.FloatTensor(Y_)).to(cfg.device)

                pred = net(X)
                loss = loss_fn(pred, Y)
                total_val_loss += loss.item()

            epoch_avg_train_loss = total_loss/n_train
            epoch_avg_val_loss = total_val_loss/n_val

            tqdm.write("Epoch [{}/{}]: TRAIN Loss: {:.4f}".format(epoch+1, cfg.optim.n_epochs, epoch_avg_train_loss))
            tqdm.write("Epoch [{}/{}]: VALID Loss: {:.4f}".format(epoch+1, cfg.optim.n_epochs, epoch_avg_val_loss))
            
            if (epoch + 1)%(cfg.log.logging_interval) == 0:
                # Log train and val losses
                logger.add_scalars('metrics/loss',
                                   {'train': epoch_avg_train_loss, 'val': epoch_avg_val_loss},
                                   epoch)

                # Log histograms of named parameters
                for param_name, param in net.named_parameters():
                    logger.add_histogram(param_name, param.clone().cpu().data.numpy(), epoch)

                # Get 1D marginal mapping plots
                # FIXME: operation is subset of the validation loop
                # but we want to free RAM
                for batch_idx, (X_, Y_) in enumerate(plot_data_loader):
                    X_plt = Variable(torch.FloatTensor(X_)).to(cfg.device)
                    Y_plt = Variable(torch.FloatTensor(Y_)).cpu().numpy()
                    pred_plt = net(X_plt).cpu().numpy()
                    break

                plotter.set_normal_mixture_params(pred_plt)
                for param_idx, param_name in enumerate(cfg.data.Y_cols):
                    tag = '1d_mapping/{:s}'.format(param_name)
                    fig = plotter.get_1d_mapping_fig(param_name, param_idx, Y_plt[:, param_idx])
                    logger.add_figure(tag, fig)

            if (epoch + 1)%(cfg.log.checkpoint_interval) == 0:
                time_stamp = str(datetime.date.today())
                torch.save(net, os.path.join(cfg.log.checkpoint_dir, 'resnet18_{:s}.mdl'.format(time_stamp)))

    logger.close()
