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
from torch.utils.tensorboard import SummaryWriter
# h0rton modules
from h0rton.trainval_data import XYData
from h0rton.configs import TrainValConfig
import h0rton.losses
import h0rton.models
import h0rton.h0_inference
import h0rton.train_utils as train_utils
import h0rton.script_utils as script_utils

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

def main():
    args = parse_args()
    cfg = TrainValConfig.from_file(args.user_cfg_path)
    # Set device and default data type
    device = torch.device(cfg.device_type)
    if device.type == 'cuda':
        torch.set_default_tensor_type('torch.cuda.' + cfg.data.float_type)
    else:
        torch.set_default_tensor_type('torch.' + cfg.data.float_type)
    script_utils.seed_everything(cfg.global_seed)

    ############
    # Data I/O #
    ############

    # Define training data and loader
    #torch.multiprocessing.set_start_method('spawn', force=True)
    train_data = XYData(is_train=True, 
                        Y_cols=cfg.data.Y_cols, 
                        float_type=cfg.data.float_type, 
                        define_src_pos_wrt_lens=cfg.data.define_src_pos_wrt_lens, 
                        rescale_pixels=cfg.data.rescale_pixels, 
                        log_pixels=cfg.data.log_pixels, 
                        add_pixel_noise=cfg.data.add_pixel_noise, 
                        eff_exposure_time=cfg.data.eff_exposure_time, 
                        train_Y_mean=None, 
                        train_Y_std=None, 
                        train_baobab_cfg_path=cfg.data.train_baobab_cfg_path, 
                        val_baobab_cfg_path=cfg.data.val_baobab_cfg_path, 
                        for_cosmology=False)
    train_loader = DataLoader(train_data, batch_size=cfg.optim.batch_size, shuffle=True, drop_last=True)
    n_train = len(train_data) - (len(train_data) % cfg.optim.batch_size)

    # Define val data and loader
    val_data = XYData(is_train=False, 
                      Y_cols=cfg.data.Y_cols, 
                      float_type=cfg.data.float_type, 
                      define_src_pos_wrt_lens=cfg.data.define_src_pos_wrt_lens, 
                      rescale_pixels=cfg.data.rescale_pixels, 
                      log_pixels=cfg.data.log_pixels, 
                      add_pixel_noise=cfg.data.add_pixel_noise, 
                      eff_exposure_time=cfg.data.eff_exposure_time, 
                      train_Y_mean=train_data.train_Y_mean, 
                      train_Y_std=train_data.train_Y_std, 
                      train_baobab_cfg_path=cfg.data.train_baobab_cfg_path, 
                      val_baobab_cfg_path=cfg.data.val_baobab_cfg_path, 
                      for_cosmology=False)
    val_loader = DataLoader(val_data, batch_size=min(len(val_data), cfg.optim.batch_size), shuffle=False, drop_last=True,)
    n_val = len(val_data) - (len(val_data) % min(len(val_data), cfg.optim.batch_size))

    #########
    # Model #
    #########
    Y_dim = val_data.Y_dim
    # Instantiate loss function
    loss_fn = getattr(h0rton.losses, cfg.model.likelihood_class)(Y_dim=Y_dim, device=device)
    # Instantiate posterior (for logging)
    bnn_post = getattr(h0rton.h0_inference.gaussian_bnn_posterior, loss_fn.posterior_name)(val_data.Y_dim, device, val_data.train_Y_mean, val_data.train_Y_std)
    # Instantiate model
    net = getattr(h0rton.models, cfg.model.architecture)(num_classes=loss_fn.out_dim, dropout_rate=cfg.model.dropout_rate)
    net.to(device)

    ################
    # Optimization #
    ################

    # Instantiate optimizer
    optimizer = optim.Adam(net.parameters(), lr=cfg.optim.learning_rate, amsgrad=False, weight_decay=cfg.optim.weight_decay)
    #optimizer = optim.SGD(net.parameters(), lr=cfg.optim.learning_rate, weight_decay=cfg.optim.weight_decay)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.75, patience=50, cooldown=50, min_lr=1e-5, verbose=True)
    #lr_scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=cfg.optim.learning_rate*0.2, max_lr=cfg.optim.learning_rate, step_size_up=cfg.optim.lr_scheduler.step_size_up, step_size_down=None, mode='triangular2', gamma=1.0, scale_fn=None, scale_mode='cycle', cycle_momentum=True, base_momentum=0.8, max_momentum=0.9, last_epoch=-1)
    
    # Saving/loading state dicts
    checkpoint_dir = cfg.checkpoint.save_dir
    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)

    if cfg.model.load_state:
        epoch, net, optimizer, train_loss, val_loss = train_utils.load_state_dict(cfg.model.state_path, net, optimizer, cfg.optim.n_epochs, device)
        epoch += 1 # resume with next epoch
        last_saved_val_loss = val_loss
        print(lr_scheduler.state_dict())
        print(optimizer.state_dict())
    else:
        epoch = 0
        last_saved_val_loss = np.inf

    logger = SummaryWriter()
    model_path = ''
    print("Training set size: {:d}".format(n_train))
    print("Validation set size: {:d}".format(n_val))
    
    progress = tqdm(range(epoch, cfg.optim.n_epochs))
    n_iter = 0
    for epoch in progress:
        #net.apply(h0rton.models.deactivate_batchnorm)
        train_loss = 0.0
        for batch_idx, (X_tr, Y_tr) in enumerate(train_loader):
            n_iter += 1
            net.train()
            X_tr = X_tr.to(device)
            Y_tr = Y_tr.to(device)
            # Update weights
            optimizer.zero_grad()
            pred_tr = net.forward(X_tr)
            loss = loss_fn(pred_tr, Y_tr)
            loss.backward()
            optimizer.step()
            # For logging
            train_loss += (loss.detach().item() - train_loss)/(1 + batch_idx)
            # Step lr_scheduler every batch
            lr_scheduler.step(train_loss)
            tqdm.write("Iter [{}/{}/{}]: TRAIN Loss: {:.4f}".format(n_iter, epoch+1, cfg.optim.n_epochs, train_loss))

            if (n_iter)%(cfg.monitoring.interval) == 0:
                net.eval()         
                with torch.no_grad():
                    #net.apply(h0rton.models.deactivate_batchnorm)
                    val_loss = 0.0
           
                    for batch_idx, (X_v, Y_v) in enumerate(val_loader):
                        X_v = X_v.to(device)
                        Y_v = Y_v.to(device)
                        pred_v = net.forward(X_v)
                        nograd_loss_v = loss_fn(pred_v, Y_v)
                        val_loss += (nograd_loss_v.detach().item() - val_loss)/(1 + batch_idx)

                    tqdm.write("Epoch [{}/{}]: VALID Loss: {:.4f}".format(epoch+1, cfg.optim.n_epochs, val_loss))
                    
                    # Subset of validation for plotting
                    n_plotting = cfg.monitoring.n_plotting
                    #X_plt = X_v[:n_plotting].cpu().numpy()
                    #Y_plt = Y[:n_plotting].cpu().numpy()
                    Y_plt_orig = bnn_post.transform_back_mu(Y_v[:n_plotting]).cpu().numpy()
                    pred_plt = pred_v[:n_plotting]
                    # Slice pred_plt into meaningful Gaussian parameters for this batch
                    bnn_post.set_sliced_pred(pred_plt)
                    mu_orig = bnn_post.transform_back_mu(bnn_post.mu).cpu().numpy()
                    # Log train and val metrics
                    loss_dict = {'train': train_loss, 'val': val_loss}
                    logger.add_scalars('metrics/loss', loss_dict, n_iter)
                    #mae = train_utils.get_mae(mu, Y_plt)
                    mae_dict = train_utils.get_mae(mu_orig, Y_plt_orig, cfg.data.Y_cols)
                    logger.add_scalars('metrics/mae', mae_dict, n_iter)
                    # Log log determinant of the covariance matrix
                    
                    if cfg.model.likelihood_class in ['DoubleGaussianNLL', 'FullRankGaussianNLL']:
                        logdet = train_utils.get_logdet(bnn_post.tril_elements.cpu().numpy(), Y_dim)
                        logger.add_histogram('logdet_cov_mat', logdet, n_iter)
                    # Log second Gaussian stats
                    if cfg.model.likelihood_class in ['DoubleGaussianNLL', 'DoubleLowRankGaussianNLL']:
                        # Log histogram of w2
                        logger.add_histogram('val_pred/weight_gaussian2', bnn_post.w2.cpu().numpy(), n_iter)
                        # Log RMSE of second Gaussian
                        mu2_orig = bnn_post.transform_back_mu(bnn_post.mu2).cpu().numpy()
                        mae2_dict = train_utils.get_mae(mu2_orig, Y_plt_orig, cfg.data.Y_cols)
                        logger.add_scalars('metrics/mae2', mae2_dict, n_iter)
                        # Log logdet of second Gaussian
                        logdet2 = train_utils.get_logdet(bnn_post.tril_elements2.cpu().numpy(), Y_dim)
                        logger.add_histogram('logdet_cov_mat2', logdet2, n_iter)

                    if val_loss < last_saved_val_loss:
                        os.remove(model_path) if os.path.exists(model_path) else None
                        model_path = train_utils.save_state_dict(net, optimizer, lr_scheduler, train_loss, val_loss, checkpoint_dir, cfg.model.architecture, epoch)
                        last_saved_val_loss = val_loss

    logger.close()
    # Save final state dict
    if val_loss < last_saved_val_loss:
        os.remove(model_path) if os.path.exists(model_path) else None
        model_path = train_utils.save_state_dict(net, optimizer, lr_scheduler, train_loss, val_loss, checkpoint_dir, cfg.model.architecture, epoch)
        print("Saved model at {:s}".format(os.path.abspath(model_path)))

if __name__ == '__main__':
    main()