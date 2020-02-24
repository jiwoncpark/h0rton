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
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
# h0rton modules
from h0rton.trainval_data import XYData
from h0rton.configs import TrainValConfig
import h0rton.losses
import h0rton.models
import h0rton.h0_inference
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
    torch.cuda.manual_seed_all(global_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    args = parse_args()
    cfg = TrainValConfig.from_file(args.user_cfg_path)
    # Set device and default data type
    device = torch.device(cfg.device_type)
    if device.type == 'cuda':
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')
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

    if cfg.data.test_dir is not None:
        test_data = XYData(cfg.data.test_dir, data_cfg=cfg.data)
        test_loader = DataLoader(test_data, batch_size=cfg.optim.batch_size, shuffle=False, drop_last=True, num_workers=4, pin_memory=True)
        n_test = test_data.n_data - (test_data.n_data % cfg.optim.batch_size)

    #########
    # Model #
    #########

    # Instantiate loss function
    loss_fn = getattr(h0rton.losses, cfg.model.likelihood_class)(Y_dim=cfg.data.Y_dim, device=device)
    # Instantiate posterior (for logging)
    bnn_post = getattr(h0rton.h0_inference.gaussian_bnn_posterior, loss_fn.posterior_name)(val_data.Y_dim, device, val_data.train_Y_mean, val_data.train_Y_std)
    # Instantiate model
    net = getattr(h0rton.models, cfg.model.architecture)(num_classes=loss_fn.out_dim)
    net.to(device)

    ################
    # Optimization #
    ################

    # Instantiate optimizer
    optimizer = optim.Adam(net.parameters(), lr=cfg.optim.learning_rate, amsgrad=True, weight_decay=cfg.optim.weight_decay)
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg.optim.lr_scheduler.milestones, gamma=cfg.optim.lr_scheduler.gamma)
    
    # Saving/loading state dicts
    checkpoint_dir = cfg.checkpoint.save_dir
    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)

    if cfg.model.load_state:
        epoch, net, optimizer, train_loss, val_loss = train_utils.load_state_dict(cfg.model.state_path, net, optimizer, cfg.optim.n_epochs, device)
        epoch += 1 # resume with next epoch
        last_saved_val_loss = val_loss
        print(lr_scheduler.state_dict())
    else:
        epoch = 0
        last_saved_val_loss = np.inf

    logger = SummaryWriter()
    model_path = ''
    print("Training set size: {:d}".format(n_train))
    print("Validation set size: {:d}".format(n_val))
    if cfg.data.test_dir is not None:
        print("Test set size: {:d}".format(n_test))
    progress = tqdm(range(epoch, cfg.optim.n_epochs))
    for epoch in progress:
        net.train()
        net.apply(h0rton.models.deactivate_batchnorm)
        train_loss = 0.0

        for batch_idx, (X_tr, Y_tr) in enumerate(train_loader):
            X_tr = X_tr.to(device)
            Y_tr = Y_tr.to(device)
            # Update weights
            optimizer.zero_grad()
            pred = net(X_tr)
            loss = loss_fn(pred, Y_tr)
            loss.backward()
            optimizer.step()
            # For logging
            train_loss += (loss.item() - train_loss)/(1 + batch_idx)
        # Step lr_scheduler every epoch
        lr_scheduler.step()

        with torch.no_grad():
            net.eval()
            net.apply(h0rton.models.deactivate_batchnorm)
            val_loss = 0.0
            test_loss = 0.0

            if cfg.data.test_dir is not None:
                for batch_idx, (X_t, Y_t) in enumerate(test_loader):
                    X_t = X_t.to(device)
                    Y_t = Y_t.to(device)
                    pred = net(X_t)
                    nograd_loss = loss_fn(pred, Y_t)
                    test_loss += (nograd_loss.item() - test_loss)/(1 + batch_idx)

            for batch_idx, (X_v, Y_v) in enumerate(val_loader):
                X_v = X_v.to(device)
                Y_v = Y_v.to(device)
                pred = net(X_v)
                nograd_loss = loss_fn(pred, Y_v)
                val_loss += (nograd_loss.item() - val_loss)/(1 + batch_idx)

            tqdm.write("Epoch [{}/{}]: TRAIN Loss: {:.4f}".format(epoch+1, cfg.optim.n_epochs, train_loss))
            tqdm.write("Epoch [{}/{}]: VALID Loss: {:.4f}".format(epoch+1, cfg.optim.n_epochs, val_loss))
            if cfg.data.test_dir is not None:
                tqdm.write("Epoch [{}/{}]: TEST Loss: {:.4f}".format(epoch+1, cfg.optim.n_epochs, test_loss))
            
            if (epoch + 1)%(cfg.monitoring.interval) == 0:
                # Subset of validation for plotting
                n_plotting = cfg.monitoring.n_plotting
                X_plt = X_v[:n_plotting].cpu().numpy()
                #Y_plt = Y[:n_plotting].cpu().numpy()
                Y_plt_orig = bnn_post.transform_back_mu(Y_v[:n_plotting]).cpu().numpy()
                pred_plt = pred[:n_plotting]
                # Slice pred_plt into meaningful Gaussian parameters for this batch
                bnn_post.set_sliced_pred(pred_plt)
                mu_orig = bnn_post.transform_back_mu(bnn_post.mu).cpu().numpy()
                mu2_orig = bnn_post.transform_back_mu(bnn_post.mu2).cpu().numpy()
                # Log train and val metrics
                scalar_dict = {'train': train_loss, 'val': val_loss}
                if cfg.data.test_dir is not None:
                    scalar_dict.update(test=test_loss)
                logger.add_scalars('metrics/loss', scalar_dict, epoch)
                #rmse = train_utils.get_rmse(mu, Y_plt)
                rmse_orig = train_utils.get_rmse(mu_orig, Y_plt_orig)
                rmse_orig2 = train_utils.get_rmse(mu2_orig, Y_plt_orig)
                logger.add_scalars('metrics/rmse',
                                   {
                                   #'rmse': rmse,
                                   'rmse_orig1': rmse_orig,
                                   'rmse_orig2': rmse_orig2,
                                   'rmse_lens_x': train_utils.get_rmse_param(mu_orig, Y_plt_orig, 0),
                                   'rmse_src_x': train_utils.get_rmse_param(mu_orig, Y_plt_orig, 1),
                                   'rmse_lens_y': train_utils.get_rmse_param(mu_orig, Y_plt_orig, 2),
                                   'rmse_src_y': train_utils.get_rmse_param(mu_orig, Y_plt_orig, 3),
                                   'rmse_gamma': train_utils.get_rmse_param(mu_orig, Y_plt_orig, 4),
                                   },
                                   epoch)
                # Log alpha value
                logger.add_histogram('val_pred/weight_gaussian2', bnn_post.w2.cpu().numpy(), epoch)
                # Log histograms of named parameters
                if cfg.monitoring.weight_distributions:
                    for param_name, param in net.named_parameters():
                        logger.add_histogram(param_name, param.clone().cpu().data.numpy(), epoch)
                # Log sample images
                if cfg.monitoring.sample_images:
                    sample_img = X_plt[:5]
                    #pred = pred.cpu().numpy()
                    logger.add_images('val_images', sample_img, epoch, dataformats='NCHW')
                # Log 1D marginal mapping
                if cfg.monitoring.marginal_1d_mapping:
                    for param_idx, param_name in enumerate(cfg.data.Y_cols):
                        tag = '1d_mapping/{:s}'.format(param_name)
                        fig = train_utils.get_1d_mapping_fig(param_name, mu_orig[:, param_idx], Y_plt_orig[:, param_idx])
                        logger.add_figure(tag, fig, global_step=epoch)

            if (epoch + 1)%(cfg.checkpoint.interval) == 0:
                # FIXME compare to last saved epoch val loss
                if val_loss < last_saved_val_loss:
                    #os.remove(model_path) if os.path.exists(model_path) else None
                    model_path = train_utils.save_state_dict(net, optimizer, lr_scheduler, train_loss, val_loss, checkpoint_dir, cfg.model.architecture, epoch)
                    last_saved_val_loss = train_loss

    logger.close()
    # Save final state dict
    if val_loss < last_saved_val_loss:
        #os.remove(model_path) if os.path.exists(model_path) else None
        model_path = train_utils.save_state_dict(net, optimizer, lr_scheduler, train_loss, val_loss, checkpoint_dir, cfg.model.architecture, epoch)
        print("Saved model at {:s}".format(os.path.abspath(model_path)))

if __name__ == '__main__':
    main()