# -*- coding: utf-8 -*-
"""Training the Bayesian neural network (BNN).
This script performs H0 inference on a test (or validation) sample using simple MC sampling

Example
-------
To run this script, pass in the path to the user-defined training config file as the argument::
    
    $ infer_h0 h0rton/h0_inference_config.json

"""
import os
import time
from tqdm import tqdm
from ast import literal_eval
import numpy as np
import pandas as pd
import scipy.stats as stats
import torch
from torch.utils.data import DataLoader
import h0rton.models
# Baobab modules
from baobab import BaobabConfig
# H0rton modules
from h0rton.configs import TrainValConfig, TestConfig
import h0rton.losses
import h0rton.script_utils as script_utils
from h0rton.h0_inference import H0Posterior, plot_weighted_h0_histogram
from h0rton.trainval_data import XYData
from astropy.cosmology import FlatLambdaCDM

def main():
    args = script_utils.parse_inference_args()
    test_cfg = TestConfig.from_file(args.test_config_file_path)
    baobab_cfg = BaobabConfig.from_file(test_cfg.data.test_baobab_cfg_path)
    cfg = TrainValConfig.from_file(test_cfg.train_val_config_file_path)
    # Set device and default data type
    device = torch.device(test_cfg.device_type)
    if device.type == 'cuda':
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')
    script_utils.seed_everything(test_cfg.global_seed)
    
    ############
    # Data I/O #
    ############
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
                        val_baobab_cfg_path=test_cfg.data.test_baobab_cfg_path, 
                        for_cosmology=False)
    # Define val data and loader
    test_data = XYData(is_train=False, 
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
                       val_baobab_cfg_path=test_cfg.data.test_baobab_cfg_path, 
                       for_cosmology=True)
    cosmo_df = test_data.Y_df
    if test_cfg.data.lens_indices is None:
        n_test = test_cfg.data.n_test # number of lenses in the test set
        lens_range = range(n_test)
    else: # if specific lenses are specified
        lens_range = test_cfg.data.lens_indices
        n_test = len(lens_range)
        print("Performing H0 inference on {:d} specified lenses...".format(n_test))
    batch_size = max(lens_range) + 1
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, drop_last=True)
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
    # Instantiate loss function
    loss_fn = getattr(h0rton.losses, cfg.model.likelihood_class)(Y_dim=train_data.Y_dim, device=device)
    # Instantiate posterior (for logging)
    bnn_post = getattr(h0rton.h0_inference.gaussian_bnn_posterior, loss_fn.posterior_name)(train_data.Y_dim, device, train_data.train_Y_mean, train_data.train_Y_std)
    with torch.no_grad(): # TODO: skip this if lens_posterior_type == 'truth'
        for X_, Y_ in test_loader:
            X = X_.to(device)
            Y = Y_.to(device)
            break

    # Export the input images X for later error analysis
    if test_cfg.export.images:
        for lens_i in range(n_test):
            X_img_path = os.path.join(out_dir, 'X_{0:04d}.npy'.format(lens_i))
            np.save(X_img_path, X[lens_i, 0, :, :].cpu().numpy())

    ################
    # H0 Posterior #
    ################
    h0_prior = getattr(stats, test_cfg.h0_prior.dist)(**test_cfg.h0_prior.kwargs)
    kappa_ext_prior = getattr(stats, test_cfg.kappa_ext_prior.dist)(**test_cfg.kappa_ext_prior.kwargs)
    aniso_param_prior = getattr(stats, test_cfg.aniso_param_prior.dist)(**test_cfg.aniso_param_prior.kwargs)
    # FIXME: hardcoded
    kwargs_model = dict(
                        lens_model_list=['PEMD', 'SHEAR'],
                        lens_light_model_list=['SERSIC_ELLIPSE'],
                        source_light_model_list=['SERSIC_ELLIPSE'],
                        point_source_model_list=['SOURCE_POSITION'],
                        cosmo=FlatLambdaCDM(H0=70.0, Om0=0.3)
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
                          kappa_transformed=test_cfg.kappa_ext_prior.transformed,
                          Om0=baobab_cfg.bnn_omega.cosmology.Om0,
                          define_src_pos_wrt_lens=cfg.data.define_src_pos_wrt_lens,
                          kwargs_lens_eqn_solver={'min_distance': 0.05, 'search_window': baobab_cfg.instrument['pixel_scale']*baobab_cfg.image['num_pix'], 'num_iter_max': 100}
                          )
    # Get H0 samples for each system
    if not test_cfg.time_delay_likelihood.baobab_time_delays:
        if 'abcd_ordering_i' not in cosmo_df:
            raise ValueError("If the time delay measurements were not generated using Baobab, the user must specify the order of image positions in which the time delays are listed, in order of increasing dec.")
    required_params = h0_post.required_params

    ########################
    # Lens Model Posterior #
    ########################
    n_samples = test_cfg.h0_posterior.n_samples # number of h0 samples per lens
    sampling_buffer = test_cfg.h0_posterior.sampling_buffer # FIXME: dynamically sample more if we run out of samples
    actual_n_samples = int(n_samples*sampling_buffer)

    # Add artificial noise around the truth values
    Y_orig = bnn_post.transform_back_mu(Y).cpu().numpy().reshape(batch_size, test_data.Y_dim)
    Y_orig_df = pd.DataFrame(Y_orig, columns=cfg.data.Y_cols)
    Y_orig_values = Y_orig_df[required_params].values[:, np.newaxis, :] # [n_test, 1, Y_dim]
    artificial_noise = np.random.randn(batch_size, actual_n_samples, test_data.Y_dim)*Y_orig_values*test_cfg.fractional_error_added_to_truth # [n_test, buffer*n_samples, Y_dim]
    lens_model_samples_values = Y_orig_values + artificial_noise # [n_test, buffer*n_samples, Y_dim]

    # Placeholders for mean and std of H0 samples per system
    mean_h0_set = np.zeros(n_test)
    std_h0_set = np.zeros(n_test)
    inference_time_set = np.zeros(n_test)
    # For each lens system...
    total_progress = tqdm(total=n_test)
    sampling_progress = tqdm(total=n_samples)
    prerealized_time_delays = test_cfg.error_model.prerealized_time_delays
    if prerealized_time_delays:
        realized_time_delays = pd.read_csv(test_cfg.error_model.realized_time_delays_path, index_col=None)
    else:
        realized_time_delays = pd.DataFrame()
        realized_time_delays['measured_td_wrt0'] = [[]]*len(lens_range)
    for i, lens_i in enumerate(lens_range):
        lens_i_start_time = time.time()
        # Each lens gets a unique random state for td and vd measurement error realizations.
        rs_lens = np.random.RandomState(lens_i)
        # BNN samples for lens_i
        bnn_sample_df = pd.DataFrame(lens_model_samples_values[lens_i, :, :], columns=required_params)
        # Cosmology observables for lens_i
        cosmo = cosmo_df.iloc[lens_i]
        true_td = np.array(literal_eval(cosmo['true_td']))
        true_img_dec = np.array(literal_eval(cosmo['y_image']))
        true_img_ra = np.array(literal_eval(cosmo['x_image']))
        increasing_dec_i = np.argsort(true_img_dec)
        true_img_dec = true_img_dec[increasing_dec_i]
        true_img_ra = true_img_ra[increasing_dec_i]        
        measured_vd = cosmo['true_vd']*(1.0 + rs_lens.randn()*test_cfg.error_model.velocity_dispersion_frac_error)
        if prerealized_time_delays:
            measured_td_wrt0 = np.array(literal_eval(realized_time_delays.iloc[lens_i]['measured_td_wrt0']))
        else:
            true_td = true_td[increasing_dec_i]
            true_td = true_td[1:] - true_td[0]
            measured_td_wrt0 = true_td + rs_lens.randn(*true_td.shape)*test_cfg.error_model.time_delay_error # [n_img -1,]
            realized_time_delays.at[lens_i, 'measured_td_wrt0'] = list(measured_td_wrt0)
        #print("True: ", true_td)
        #print("True img: ", true_img_dec)
        #print("measured td: ", measured_td_wrt0)
        h0_post.set_cosmology_observables(
                                          z_lens=cosmo['z_lens'], 
                                          z_src=cosmo['z_src'], 
                                          measured_vd=measured_vd, 
                                          measured_vd_err=test_cfg.velocity_dispersion_likelihood.sigma, 
                                          measured_td_wrt0=measured_td_wrt0,
                                          measured_td_err=test_cfg.time_delay_likelihood.sigma, 
                                          abcd_ordering_i=np.arange(len(true_td) + 1),
                                          true_img_dec=true_img_dec,
                                          true_img_ra=true_img_ra,
                                          kappa_ext=cosmo['kappa_ext'], # not necessary
                                          )
        h0_post.set_truth_lens_model(sampled_lens_model_raw=bnn_sample_df.iloc[0])
        # Initialize output array
        h0_samples = np.full(n_samples, np.nan)
        h0_weights = np.zeros(n_samples)
        # For each sample from the lens model posterior of this lens system...
        sampling_progress.reset()
        valid_sample_i = 0
        sample_i = 0
        while valid_sample_i < n_samples:
            if sample_i > actual_n_samples - 1:
                break
            #try:
            # Each sample for a given lens gets a unique random state for H0, k_ext, and aniso_param realizations.
            rs_sample = np.random.RandomState(int(str(lens_i) + str(sample_i).zfill(5)))
            h0, weight = h0_post.get_h0_sample_truth(rs_sample)
            h0_samples[valid_sample_i] = h0
            h0_weights[valid_sample_i] = weight
            sampling_progress.update(1)
            time.sleep(0.001)
            valid_sample_i += 1
            sample_i += 1
            #except:
            #    sample_i += 1
            #    continue
        sampling_progress.refresh()
        lens_i_end_time = time.time()
        inference_time = (lens_i_end_time - lens_i_start_time)/60.0 # min
        h0_dict = dict(
                       h0_samples=h0_samples,
                       h0_weights=h0_weights,
                       n_sampling_attempts=sample_i,
                       measured_td_wrt0=measured_td_wrt0,
                       inference_time=inference_time
                       )
        h0_dict_save_path = os.path.join(out_dir, 'h0_dict_{0:04d}.npy'.format(lens_i))
        np.save(h0_dict_save_path, h0_dict)
        h0_stats = plot_weighted_h0_histogram(h0_samples, h0_weights, lens_i, cosmo['H0'], include_fit_gaussian=test_cfg.plotting.include_fit_gaussian, save_dir=out_dir)
        mean_h0_set[i] = h0_stats['mean']
        std_h0_set[i] = h0_stats['std']
        inference_time_set[i] = inference_time
        total_progress.update(1)
    total_progress.close()
    if not prerealized_time_delays:
        realized_time_delays.to_csv(os.path.join(out_dir, 'realized_time_delays.csv'), index=None)
    h0_stats = dict(
                    mean=mean_h0_set,
                    std=std_h0_set,
                    inference_time=inference_time_set,
                    )
    h0_stats_save_path = os.path.join(out_dir, 'h0_stats')
    np.save(h0_stats_save_path, h0_stats)

if __name__ == '__main__':
    main()