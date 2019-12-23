# -*- coding: utf-8 -*-
"""Training the Bayesian neural network (BNN).
This script trains the BNN according to the config specifications.

Example
-------
To run this script, pass in the path to the user-defined training config file as the argument::
    
    $ train h0rton/example_user_config.py

"""

import os, sys
import random
import argparse
from addict import Dict
import numpy as np # linear algebra
from tqdm import tqdm
# torch modules
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.models
from torch.utils.tensorboard import SummaryWriter
# h0rton modules
from h0rton.trainval_data import XYData
from h0rton.configs import BNNConfig
import h0rton.losses
from h0rton.plotting import BNNInterpreter
import h0rton.train_utils as train_utils

def parse_args():
    """Parse command-line arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('user_cfg_path', help='path to the user-defined training config file')
    #parser.add_argument('--n_data', default=None, dest='n_data', type=int,
    #                    help='size of dataset to generate (overrides config file)')
    args = parser.parse_args()
    # sys.argv rerouting for setuptools entry point
    if args is None:
        args = Dict()
        args.user_cfg_path = sys.argv[0]
        #args.n_data = sys.argv[1]
    return args

def seed_everything(global_seed):
    """Seed everything for reproducibility

    global_seed : int
        seed for `np.random`, `random`, and relevant `torch` backends

    """
    np.random.seed(global_seed)
    random.seed(global_seed)
    torch.manual_seed(global_seed)
    torch.cuda.manual_seed(global_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    args = parse_args()
    cfg = BNNConfig.from_file(args.user_cfg_path)
    seed_everything(cfg.global_seed)

    ############
    # Data I/O #
    ############

    # Define training data and loader
    torch.multiprocessing.set_start_method('spawn', force=True)
    train_data = XYData(cfg.data.train_dir, data_cfg=cfg.data)
    train_loader = DataLoader(train_data, batch_size=cfg.optim.batch_size, shuffle=True, drop_last=True, num_workers=4, pin_memory=True)
    n_train = train_data.n_data - (train_data.n_data % cfg.optim.batch_size)

    # Define val data and loader
    val_data = XYData(cfg.data.val_dir, data_cfg=cfg.data)
    val_loader = DataLoader(val_data, batch_size=cfg.optim.batch_size, shuffle=False, drop_last=True, num_workers=4, pin_memory=True)
    n_val = val_data.n_data - (val_data.n_data % cfg.optim.batch_size)

    # Define plotting data (subset of val data) and loader
    if cfg.log.monitor_1d_marginal_mapping:
        plotter = BNNInterpreter(cfg.model.type, cfg.data.Y_dim, cfg.device)

    #########
    # Model #
    #########

    # Instantiate loss function
    loss_fn = getattr(h0rton.losses, cfg.model.likelihood_class)(Y_dim=cfg.data.Y_dim, device=cfg.device)
    # Instantiate model
    net = getattr(torchvision.models, cfg.model.architecture)(pretrained=True)
    n_filters = net.fc.in_features # number of output nodes in 2nd-to-last layer
    net.fc = nn.Linear(in_features=n_filters, out_features=loss_fn.out_dim) # replace final layer
    net.to(cfg.device)

    ################
    # Optimization #
    ################

    # Instantiate optimizer
    optimizer = optim.Adam(net.parameters(), lr=cfg.optim.learning_rate, amsgrad=True, weight_decay=cfg.optim.weight_decay)
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg.optim.lr_scheduler.milestones, gamma=cfg.optim.lr_scheduler.gamma)
    
    # Saving/loading state dicts
    if not os.path.exists(cfg.log.checkpoint_dir):
        os.mkdir(cfg.log.checkpoint_dir)

    if cfg.model.load_state:
        net, optimizer, lr_scheduler, epoch = train_utils.load_state_dict(cfg.model.state_path, net, optimizer, lr_scheduler, cfg.optim.n_epochs, cfg.device)
        epoch += 1 # resume with next epoch
    else:
        epoch = 0

    logger = SummaryWriter()
    model_path = ''
    last_saved_val_loss = np.inf
    print("Training set size: {:d}".format(n_train))
    print("Validation set size: {:d}".format(n_val))
    progress = tqdm(range(epoch, cfg.optim.n_epochs))
    for epoch in progress:
        net.train()
        train_loss = 0.0

        for batch_idx, (X_, Y_) in enumerate(train_loader):
            X = X_.to(cfg.device)
            Y = Y_.to(cfg.device)
            # Update weights
            optimizer.zero_grad()
            pred = net(X)
            loss = loss_fn(pred, Y)
            loss.backward()
            optimizer.step()
            # For logging
            train_loss += (loss.item() - train_loss)/(1 + batch_idx)
        # Step lr_scheduler every epoch
        lr_scheduler.step()

        with torch.no_grad():
            net.eval()
            val_loss = 0.0

            for batch_idx, (X_, Y_) in enumerate(val_loader):
                X = X_.to(cfg.device)
                Y = Y_.to(cfg.device)
                pred = net(X)
                nograd_loss = loss_fn(pred, Y)
                val_loss += (nograd_loss.item() - val_loss)/(1 + batch_idx)

            tqdm.write("Epoch [{}/{}]: TRAIN Loss: {:.4f}".format(epoch+1, cfg.optim.n_epochs, train_loss))
            tqdm.write("Epoch [{}/{}]: VALID Loss: {:.4f}".format(epoch+1, cfg.optim.n_epochs, val_loss))
            
            if (epoch + 1)%(cfg.log.logging_interval) == 0:
                # Subset of validation for plotting
                # TODO: enforce batch_size >= n_plotting
                X_plt = X[:cfg.data.n_plotting].cpu().numpy()
                Y_plt = Y[:cfg.data.n_plotting].cpu().numpy()
                pred_plt = pred[:cfg.data.n_plotting].cpu().numpy()
                pred_dict = train_utils.interpret_pred(pred_plt, Y_dim=cfg.data.Y_dim)
                # Log train and val metrics
                logger.add_scalars('metrics/loss',
                                   {
                                   'train': train_loss, 
                                   'val': val_loss
                                   },
                                   epoch)
                transformed_rmse = train_utils.get_transformed_rmse(pred_dict['mu'], Y_plt)
                logger.add_scalars('metrics/rmse',
                                   {
                                   'transformed_rmse': transformed_rmse,
                                   },
                                   epoch)
                # Log alpha value
                logger.add_histogram('val_pred/weight_gaussian2', pred_dict['w2'], epoch)
                # Log histograms of named parameters
                if cfg.log.monitor_weight_distributions:
                    for param_name, param in net.named_parameters():
                        logger.add_histogram(param_name, param.clone().cpu().data.numpy(), epoch)
                # Log sample images
                if cfg.log.monitor_sample_images:
                    X = X_plt[:3]
                    #pred = pred.cpu().numpy()
                    logger.add_images('val_images', X, epoch, dataformats='NCHW')

                if cfg.log.monitor_1d_marginal_mapping:
                    plotter.set_normal_mixture_params(pred)
                    for param_idx, param_name in enumerate(cfg.data.Y_cols):
                        tag = '1d_mapping/{:s}'.format(param_name)
                        fig = plotter.get_1d_mapping_fig(param_name, param_idx, Y[:, param_idx])
                        logger.add_figure(tag, fig)

            if (epoch + 1)%(cfg.log.checkpoint_interval) == 0:
                # FIXME compare to last saved epoch val loss
                if val_loss < last_saved_val_loss:
                    os.remove(model_path) if os.path.exists(model_path) else None
                    model_path = train_utils.save_state_dict(net, optimizer, lr_scheduler, train_loss, val_loss, cfg.log.checkpoint_dir, cfg.model.architecture, epoch)
                    last_saved_val_loss = val_loss

    logger.close()
    # Save final state dict
    if val_loss < last_saved_val_loss:
        os.remove(model_path) if os.path.exists(model_path) else None
        model_path = train_utils.save_state_dict(net, optimizer, lr_scheduler, train_loss, val_loss, cfg.log.checkpoint_dir, cfg.model.architecture, epoch)
        print("Saved model at {:s}".format(os.path.abspath(model_path)))

if __name__ == '__main__':
    main()