### 2019-7-30 neural networks by Joshua Yao-Yu Lin
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import torch
import torch.nn as nn
import torch.nn.functional as F
from  torch.autograd import Variable
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
#import lenstronomy.Util.image_util as image_util
import os, sys
from scipy.ndimage import gaussian_filter, rotate
from tqdm import tqdm
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
import time
import datetime

from data.data_io import XYData
from utils.config import cfg

if cfg.DATA.NORMALIZE:
    normalize = transforms.Normalize(mean=cfg.DATA.MEAN, std=cfg.DATA.STD)
    data_transform = transforms.Compose([transforms.ToTensor(), normalize])
else:
    data_transform = None

train_data = XYData(cfg.DATA.TRAIN, train=True, transform=data_transform, target_transform=torch.Tensor, interpolation=cfg.DATA.X_DIM)
train_loader = DataLoader(train_data, batch_size=cfg.OPTIM.BATCH_SIZE, shuffle=True)
n_train = train_data.n_data

val_data = XYData(cfg.DATA.VAL, train=False, transform=data_transform, target_transform=torch.Tensor, interpolation=cfg.DATA.X_DIM)
val_loader = DataLoader(val_data, batch_size=cfg.OPTIM.BATCH_SIZE, shuffle=True)
n_val = val_data.n_data

if __name__ == '__main__':
    net = models.resnet18(pretrained=cfg.MODEL.LOAD_PRETRAINED)
    n_filters = net.fc.in_features # number of output nodes in 2nd-to-last layer
    net.fc = nn.Linear(in_features=n_filters, out_features=cfg.MODEL.OUT_DIM) # replace final layer
    loss_fn = nn.MSELoss(reduction='sum')

    net.cuda()

    optimizer = optim.Adam(net.parameters(), lr=cfg.OPTIM.LEARNING_RATE)
    logger = SummaryWriter()

    if not os.path.exists(cfg.LOG.CHECKPOINT_DIR):
        os.mkdir(cfg.LOG.CHECKPOINT_DIR)
        #net = torch.load('./saved_model/resnet18.mdl')
        #print('loaded mdl!')

    progress = tqdm(range(cfg.OPTIM.N_EPOCHS))
    for epoch in progress:

        net.train()
        total_loss = 0.0

        for batch_idx, (X, theta_E, gamma, center_x, center_y, e1, e2, source_x, source_y, gamma_ext, psi_ext, source_R_sersic, source_n_sersic, sersic_source_e1, sersic_source_e2, lens_light_e1, lens_light_e2, lens_light_R_sersic, lens_light_n_sersic) in enumerate(train_loader):
            X, theta_E, gamma, center_x, center_y, e1, e2, source_x, source_y, gamma_ext, psi_ext, source_R_sersic, source_n_sersic, sersic_source_e1, sersic_source_e2, lens_light_e1, lens_light_e2, lens_light_R_sersic, lens_light_n_sersic = X.float(), theta_E.float(), gamma.float(), center_x.float(), center_y.float(), e1.float(), e2.float(), source_x.float(), source_y.float(), gamma_ext.float(), psi_ext.float(), source_R_sersic.float(), source_n_sersic.float(), sersic_source_e1.float(), sersic_source_e2.float(), lens_light_e1.float(), lens_light_e2.float(), lens_light_R_sersic.float(), lens_light_n_sersic.float()
            X, theta_E, gamma, center_x, center_y, e1, e2, source_x, source_y, gamma_ext, psi_ext, source_R_sersic, source_n_sersic, sersic_source_e1, sersic_source_e2, lens_light_e1, lens_light_e2, lens_light_R_sersic, lens_light_n_sersic = Variable(X).cuda(), Variable(theta_E).cuda(), Variable(gamma).cuda(), Variable(center_x).cuda(), Variable(center_y).cuda(), Variable(e1).cuda(), Variable(e2).cuda(), Variable(source_x).cuda(), Variable(source_y).cuda(), Variable(gamma_ext).cuda(), Variable(psi_ext).cuda(), Variable(source_R_sersic).cuda(), Variable(source_n_sersic).cuda(), Variable(sersic_source_e1).cuda(), Variable(sersic_source_e2).cuda(), Variable(lens_light_e1).cuda(), Variable(lens_light_e2).cuda(), Variable(lens_light_R_sersic).cuda(), Variable(lens_light_n_sersic).cuda()

            output = net(X)
            #print(output[:, 1].unsqueeze(1).shape, theta_E.shape)
            target = torch.cat((theta_E, gamma, center_x, center_y, e1, e2, source_x, source_y, gamma_ext, psi_ext, source_R_sersic, source_n_sersic, sersic_source_e1, sersic_source_e2, lens_light_e1, lens_light_e2, lens_light_R_sersic, lens_light_n_sersic), dim = 1)
            #target = torch.cat(target, e2)
            #print(output.shape, target.shape)
            #loss_theta_E = loss_fn(output[:, 0].unsqueeze(1), theta_E)
            loss = loss_fn(output, target)
            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        tqdm.write("Epoch [{}/{}]: TRAIN Loss: {:.4f}".format(epoch+1, cfg.OPTIM.N_EPOCHS, total_loss/n_train))

        with torch.no_grad():
            net.eval()
            total_val_loss = 0.0

            for batch_idx, (X, theta_E, gamma, center_x, center_y, e1, e2, source_x, source_y, gamma_ext, psi_ext, source_R_sersic, source_n_sersic, sersic_source_e1, sersic_source_e2, lens_light_e1, lens_light_e2, lens_light_R_sersic, lens_light_n_sersic) in enumerate(val_loader):
                X, theta_E, gamma, center_x, center_y, e1, e2, source_x, source_y, gamma_ext, psi_ext, source_R_sersic, source_n_sersic, sersic_source_e1, sersic_source_e2, lens_light_e1, lens_light_e2, lens_light_R_sersic, lens_light_n_sersic = X.float(), theta_E.float(), gamma.float(), center_x.float(), center_y.float(), e1.float(), e2.float(), source_x.float(), source_y.float(), gamma_ext.float(), psi_ext.float(), source_R_sersic.float(), source_n_sersic.float(), sersic_source_e1.float(), sersic_source_e2.float(), lens_light_e1.float(), lens_light_e2.float(), lens_light_R_sersic.float(), lens_light_n_sersic.float()
                X, theta_E, gamma, center_x, center_y, e1, e2, source_x, source_y, gamma_ext, psi_ext, source_R_sersic, source_n_sersic, sersic_source_e1, sersic_source_e2, lens_light_e1, lens_light_e2, lens_light_R_sersic, lens_light_n_sersic = Variable(X).cuda(), Variable(theta_E).cuda(), Variable(gamma).cuda(), Variable(center_x).cuda(), Variable(center_y).cuda(), Variable(e1).cuda(), Variable(e2).cuda(), Variable(source_x).cuda(), Variable(source_y).cuda(), Variable(gamma_ext).cuda(), Variable(psi_ext).cuda(), Variable(source_R_sersic).cuda(), Variable(source_n_sersic).cuda(), Variable(sersic_source_e1).cuda(), Variable(sersic_source_e2).cuda(), Variable(lens_light_e1).cuda(), Variable(lens_light_e2).cuda(), Variable(lens_light_R_sersic).cuda(), Variable(lens_light_n_sersic).cuda()
                target = torch.cat((theta_E, gamma, center_x, center_y, e1, e2, source_x, source_y, gamma_ext, psi_ext, source_R_sersic, source_n_sersic, sersic_source_e1, sersic_source_e2, lens_light_e1, lens_light_e2, lens_light_R_sersic, lens_light_n_sersic), dim=1)
                #pred [batch, out_caps_num, out_caps_size, 1]
                pred = net(X)

                loss = loss_fn(pred, target)
                total_val_loss += loss

            tqdm.write("Epoch [{}/{}]: VALID Loss: {:.4f}".format(epoch+1, cfg.OPTIM.N_EPOCHS, total_val_loss/n_val))

        if (epoch + 1)%(cfg.LOG.CHECKPOINT_INTERVAL):
            torch.save(net, os.path.join(cfg.LOG.CHECKPOINT_DIR, 'resnet.mdl'))

    logger.close()
