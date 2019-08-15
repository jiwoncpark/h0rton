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
from tqdm import tqdm
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
import datetime

from data.data_io import XYData
from utils.config import cfg
from utils.loss import GaussianNLL

np.random.seed(cfg.GLOBAL_SEED)
torch.manual_seed(cfg.GLOBAL_SEED)
torch.cuda.manual_seed(cfg.GLOBAL_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if cfg.DATA.NORMALIZE:
    normalize = transforms.Normalize(mean=cfg.DATA.MEAN, std=cfg.DATA.STD)
    data_transform = transforms.Compose([transforms.ToTensor(), normalize])
else:
    data_transform = None

if cfg.DEVICE == 'cuda':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

train_data = XYData(cfg.DATA.TRAIN, Y_cols=cfg.DATA.Y_COLS, train=True, transform=data_transform, interpolation=cfg.DATA.X_DIM)
train_loader = DataLoader(train_data, batch_size=cfg.OPTIM.BATCH_SIZE, shuffle=True)
n_train = train_data.n_data

val_data = XYData(cfg.DATA.VAL, Y_cols=cfg.DATA.Y_COLS, train=False, transform=data_transform, interpolation=cfg.DATA.X_DIM)
val_loader = DataLoader(val_data, batch_size=cfg.OPTIM.BATCH_SIZE, shuffle=True)
n_val = val_data.n_data

if __name__ == '__main__':
    net = models.resnet18(pretrained=cfg.MODEL.LOAD_PRETRAINED)
    n_filters = net.fc.in_features # number of output nodes in 2nd-to-last layer
    net.fc = nn.Linear(in_features=n_filters, out_features=cfg.MODEL.OUT_DIM) # replace final layer

    loss_fn = GaussianNLL(cov_mat=cfg.MODEL.TYPE, y_dim=cfg.DATA.Y_DIM, out_dim=cfg.MODEL.OUT_DIM, device=cfg.DEVICE)

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

        for batch_idx, (X_, Y_) in enumerate(train_loader):
            X = Variable(torch.FloatTensor(X_)).to(cfg.DEVICE)
            Y = Variable(torch.FloatTensor(Y_)).to(cfg.DEVICE)

            pred = net(X)
            loss = loss_fn(pred, Y)
            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        tqdm.write("Epoch [{}/{}]: TRAIN Loss: {:.4f}".format(epoch+1, cfg.OPTIM.N_EPOCHS, total_loss/n_train))

        with torch.no_grad():
            net.eval()
            total_val_loss = 0.0

            for batch_idx, (X_, Y_) in enumerate(val_loader):
                X = Variable(torch.FloatTensor(X_)).to(cfg.DEVICE)
                Y = Variable(torch.FloatTensor(Y_)).to(cfg.DEVICE)

                pred = net(X)
                loss = loss_fn(pred, Y)
                total_val_loss += loss.item()

            tqdm.write("Epoch [{}/{}]: VALID Loss: {:.4f}".format(epoch+1, cfg.OPTIM.N_EPOCHS, total_val_loss/n_val))
            logger.add_scalar('val_loss', loss.item(), epoch)

        if (epoch + 1)%(cfg.LOG.CHECKPOINT_INTERVAL):
            time_stamp = str(datetime.date.today())
            torch.save(net, os.path.join(cfg.LOG.CHECKPOINT_DIR, 'resnet18_{:s}.mdl'.format(time_stamp)))

    logger.close()
