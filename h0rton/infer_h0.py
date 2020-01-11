# -*- coding: utf-8 -*-
"""Training the Bayesian neural network (BNN).
This script performs H0 inference on a test (or validation) sample

Example
-------
To run this script, pass in the path to the user-defined training config file as the argument::
    
    $ infer_h0 h0rton/h0_inference_config.json

"""
import os
import sys
import argparse
import random
from addict import Dict
from tqdm import tqdm
import json
import glob
import numpy as np
import pandas as pd
import scipy.stats as stats
import torch
from torch.utils.data import DataLoader
import torchvision.models
# Baobab modules
import baobab.sim_utils
from baobab import BaobabConfig
# H0rton modules
from h0rton.configs import TrainValConfig, TestConfig
import h0rton.losses
import h0rton.train_utils as train_utils
from h0rton.h0_inference import DoubleGaussianBNNPosterior, H0Posterior, plot_h0_histogram
from h0rton.trainval_data import XYCosmoData

def parse_args():
    """Parse command-line arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('test_config_file_path', help='path to the user-defined test config file')
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

def get_baobab_config(baobab_out_dir):
    """Load the baobab log

    Parameters
    ----------
    baobab_out_dir : str or os.path object
        path to the baobab output directory

    Returns
    -------
    baobab.BaobabConfig object
        log of the baobab-generated dataset, including the input config

    """
    baobab_log_path = glob.glob(os.path.join(baobab_out_dir, 'log_*_baobab.json'))[0]
    with open(baobab_log_path, 'r') as f:
        log_str = f.read()
    baobab_cfg = BaobabConfig(Dict(json.loads(log_str)))
    return baobab_cfg

def main():
    args = parse_args()
    test_cfg = TestConfig.from_file(args.test_config_file_path)
    baobab_cfg = get_baobab_config(test_cfg.data.test_dir)
    train_val_cfg = TrainValConfig.from_file(test_cfg.train_val_config_file_path)
    # Set device and default data type
    device = torch.device(test_cfg.device_type)
    if device.type == 'cuda':
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')
    seed_everything(test_cfg.global_seed)
    
    ############
    # Data I/O #
    ############
    test_data = XYCosmoData(test_cfg.data.test_dir, data_cfg=train_val_cfg.data)
    n_test = test_cfg.data.n_test # number of lenses in the test set
    test_loader = DataLoader(test_data, batch_size=n_test, shuffle=False, drop_last=True)
    cosmo_df = test_data.cosmo_df # cosmography observables
    # Output directory into which the H0 histograms and H0 samples will be saved
    out_dir = test_cfg.out_dir
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        print("Destination folder path: {:s}".format(out_dir))
    else:
        raise OSError("Destination folder already exists.")

    ######################
    # Load trained state #
    ######################
    loss_fn = getattr(h0rton.losses, train_val_cfg.model.likelihood_class)(Y_dim=train_val_cfg.data.Y_dim, device=device)
    net = getattr(torchvision.models, train_val_cfg.model.architecture)(pretrained=False)
    n_filters = net.fc.in_features
    net.fc = torch.nn.Linear(in_features=n_filters, out_features=loss_fn.out_dim) # replace final layer
    net.to(device)
    # Load trained weights from saved state
    net, epoch = train_utils.load_state_dict_test(test_cfg.state_dict_path, net, train_val_cfg.optim.n_epochs, device)
    with torch.no_grad():
        net.eval()
        for _, (X_, Y_) in enumerate(test_loader):
            X = X_.to(device)
            Y = Y_.to(device)
            pred = net(X)
            break

    #################
    # BNN Posterior #
    #################
    bnn_post = DoubleGaussianBNNPosterior(test_data.Y_dim, device, train_val_cfg.data.Y_cols_to_whiten_idx, train_val_cfg.data.train_Y_mean, train_val_cfg.data.train_Y_std, train_val_cfg.data.Y_cols_to_log_parameterize_idx)
    bnn_post.set_sliced_pred(pred)
    # Ground truth
    Y_orig = bnn_post.transform_back(Y).cpu().numpy().reshape(n_test, -1)
    Y_orig_df = pd.DataFrame(Y_orig, columns=train_val_cfg.data.Y_cols)
    if 'external_shear_gamma1' in Y_orig_df.columns:
        Y_orig_df = baobab.sim_utils.add_gamma_psi_ext_columns(Y_orig_df)
    augmented_Y_dim = len(Y_orig_df.columns.values)

    n_samples = test_cfg.h0_posterior.n_samples # number of h0 samples per lens
    bnn_samples = bnn_post.sample(n_samples, sample_seed=test_cfg.global_seed).reshape(-1, test_data.Y_dim)

    ################
    # H0 Posterior #
    ################
    h0_prior = getattr(stats, test_cfg.h0_prior.dist)(**test_cfg.h0_prior.kwargs)
    kappa_ext_prior = getattr(stats, test_cfg.kappa_ext_prior.dist)(**test_cfg.kappa_ext_prior.kwargs)
    aniso_param_prior = getattr(stats, test_cfg.aniso_param_prior.dist)(**test_cfg.aniso_param_prior.kwargs)
    # FIXME: hardcoded
    kwargs_model = dict(
                        lens_model_list=['SPEMD', 'SHEAR_GAMMA_PSI'],
                        lens_light_model_list=['SERSIC_ELLIPSE'],
                        source_light_model_list=['SERSIC_ELLIPSE'],
                        point_source_model_list=['SOURCE_POSITION'],
                       #'point_source_model_list' : ['LENSED_POSITION']
                       )
    h0_post = H0Posterior(
                          H0_prior=h0_prior,
                          kappa_ext_prior=kappa_ext_prior,
                          aniso_param_prior=aniso_param_prior,
                          exclude_vel_disp=test_cfg.h0_posterior.exclude_velocity_dispersion,
                          kwargs_model=kwargs_model,
                          baobab_time_delays=test_cfg.time_delay_likelihood.baobab_time_delays,
                          kinematics=baobab_cfg.bnn_omega.kinematics,
                          Om0=baobab_cfg.bnn_omega.cosmology.Om0
                          )
    # Get H0 samples for each system
    if not test_cfg.time_delay_likelihood.baobab_time_delays:
        if 'abcd_ordering_i' not in cosmo_df:
            raise ValueError("If the time delay measurements were not generated using Baobab, the user must specify the order of image positions in which the time delays are listed, in order of increasing dec.")
    bnn_samples = pd.DataFrame(bnn_samples, columns=train_val_cfg.data.Y_cols)
    # Convert shear and ellipticity to gamma/psi and e1/e2, respectively
    if 'external_shear_gamma1' in bnn_samples.columns:
        bnn_samples = baobab.sim_utils.add_gamma_psi_ext_columns(bnn_samples)
    required_params = h0_post.required_params
    bnn_samples_values = bnn_samples[required_params].values.reshape(n_test, n_samples, -1)
    mean_h0_set = np.zeros(n_test)
    std_h0_set = np.zeros(n_test)
    for lens_i in tqdm(range(n_test)):
        # BNN samples for lens_i
        bnn_sample_df = pd.DataFrame(bnn_samples_values[lens_i, :, :], columns=required_params)
        # Cosmology observables for lens_i
        cosmo = cosmo_df.iloc[lens_i]
        #print("TRUTH", Y_orig[lens_i, :])
        h0_post.set_cosmology_observables(
                                          z_lens=cosmo['z_lens'], 
                                          z_src=cosmo['z_src'], 
                                          measured_vd=cosmo['measured_vd'], 
                                          measured_vd_err=cosmo['measured_vd_err'], 
                                          measured_td=cosmo['measured_td'],
                                          measured_td_err=cosmo['measured_td_err'], 
                                          abcd_ordering_i=np.arange(len(cosmo['measured_td']) + 1)
                                          )
        # Initialize output array
        h0_samples = np.zeros(n_samples)
        h0_weights = np.zeros(n_samples)
        for sample_i in tqdm(range(n_samples)):
            if test_cfg.use_truth_lens_model:
                single_bnn_sample = Y_orig_df.iloc[lens_i] + np.random.randn(augmented_Y_dim)*Y_orig_df.iloc[lens_i]*test_cfg.fractional_error_added_to_truth
            else:
                single_bnn_sample = bnn_sample_df.iloc[sample_i]
            h0_post.set_lens_model(bnn_sample=single_bnn_sample)
            #print("PRED", bnn_sample_df.iloc[sample_i])
            try:
                h0, weight = h0_post.get_sample()
            except:
                continue
            h0_samples[sample_i] = h0
            h0_weights[sample_i] = weight
        # Normalize weights to unity
        h0_weights /= np.sum(h0_weights)
        h0_dict = dict(
                       h0_samples=h0_samples,
                       h0_weights=h0_weights,
                       )
        h0_dict_save_path = os.path.join(out_dir, 'h0_dict_{0:04d}.npy'.format(lens_i))
        np.save(h0_dict_save_path, h0_dict)
        mean_h0, std_h0 = plot_h0_histogram(h0_dict, lens_i, cosmo['H0'], save_dir=out_dir)
        mean_h0_set[lens_i] = mean_h0
        std_h0_set[lens_i] = std_h0
    h0_stats = dict(
                    mean=mean_h0_set,
                    std=std_h0_set,
                    )
    h0_stats_save_path = os.path.join(out_dir, 'h0_stats')
    np.save(h0_stats_save_path, h0_stats)

if __name__ == '__main__':
    main()