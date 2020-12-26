"""Script to run an MCMC afterburner for the BNN posterior

It borrows heavily from the `catalogue modelling.ipynb` notebook in Lenstronomy 
Extensions, which you can find `here <https://github.com/sibirrer/lenstronomy_\
extensions/blob/master/lenstronomy_extensions/Notebooks/\
catalogue%20modelling.ipynb>`_.

"""
import argparse
import os
import sys
import gc
import time
from tqdm import tqdm
from ast import literal_eval
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from lenstronomy.Workflow.fitting_sequence import FittingSequence
from lenstronomy.Cosmo.lcdm import LCDM
from baobab import BaobabConfig
from h0rton.script_utils import seed_everything, get_batch_size, infer_bnn
import h0rton.models
from h0rton.configs import TrainValConfig, TestConfig
import h0rton.losses
import h0rton.train_utils as train_utils
from h0rton.h0_inference import h0_utils, plotting_utils, mcmc_utils
from h0rton.trainval_data import TDLMCData, XYData

def parse_args():
    """Parse command-line arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('test_config_file_path', 
                        help='path to the user-defined test config file')
    parser.add_argument('rung_idx', 
                        help='TLDMC rung number', type=int)
    parser.add_argument('--lens_indices_path', 
                        default=None, dest='lens_indices_path', type=str,
                        help='path to a text file with specific lens indices \
                        to test on (Default: None)')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    test_cfg = TestConfig.from_file(args.test_config_file_path)
    cfg = TrainValConfig.from_file(test_cfg.train_val_config_file_path)
    baobab_cfg = BaobabConfig.from_file(test_cfg.data.test_baobab_cfg_path)
    # Set device and default data type
    device = torch.device(test_cfg.device_type)
    if device.type == 'cuda':
        torch.set_default_tensor_type('torch.cuda.' + cfg.data.float_type)
    else:
        torch.set_default_tensor_type('torch.' + cfg.data.float_type)
    seed_everything(test_cfg.global_seed)
    
    ############
    # Data I/O #
    ############
    train_data = XYData(is_train=True, 
                        Y_cols=cfg.data.Y_cols, 
                        float_type=cfg.data.float_type, 
                        define_src_pos_wrt_lens=cfg.data.define_src_pos_wrt_lens, 
                        rescale_pixels=cfg.data.rescale_pixels, 
                        rescale_pixels_type=cfg.data.rescale_pixels_type,
                        log_pixels=cfg.data.log_pixels, 
                        add_pixel_noise=cfg.data.add_pixel_noise, 
                        eff_exposure_time=cfg.data.eff_exposure_time, 
                        train_Y_mean=None, 
                        train_Y_std=None, 
                        train_baobab_cfg_path=cfg.data.train_baobab_cfg_path, 
                        val_baobab_cfg_path=None, 
                        for_cosmology=False)
    test_data = TDLMCData(float_type=cfg.data.float_type,
                          rescale_pixels=cfg.data.rescale_pixels, 
                          rescale_pixels_type=cfg.data.rescale_pixels_type,
                          log_pixels=cfg.data.log_pixels, 
                          rung_i=args.rung_idx)
    master_truth = test_data.cosmo_df
    batch_size, n_test, lens_range = get_batch_size(
                                                    test_cfg.data.lens_indices, 
                                                    test_cfg.data.n_test,
                                                    args.lens_indices_path,
                                                    )
    test_loader = DataLoader(test_data, 
                             batch_size=batch_size, 
                             shuffle=False, 
                             drop_last=True)
    # Output directory into which the H0 histograms and H0 samples will be saved
    out_dir = test_cfg.out_dir
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        print("Destination folder path: {:s}".format(out_dir))
    else:
        raise OSError("Destination folder already exists.")

    #####################
    # Parameter penalty #
    #####################
    # Instantiate original loss function with all BNN-predicted params
    orig_Y_cols = cfg.data.Y_cols
    loss_fn = getattr(h0rton.losses, cfg.model.likelihood_class)(Y_dim=train_data.Y_dim, 
                                                                 device=device)
    param_logL = mcmc_utils.HybridBNNPenalty(orig_Y_cols, 
                                             cfg.model.likelihood_class, 
                                             train_data.train_Y_mean, 
                                             train_data.train_Y_std, 
                                             True, # exclude vel_disp
                                             device)
    param_logL.remove_params(['lens_light_R_sersic']) # 

    ###################
    # BNN predictions #
    ###################
    # Instantiate BNN model
    net = getattr(
                  h0rton.models, cfg.model.architecture
                  )(
                  num_classes=loss_fn.out_dim, 
                  dropout_rate=cfg.model.dropout_rate
                  )
    net.to(device)
    # Load trained weights from saved state
    net, epoch = train_utils.load_state_dict_test(test_cfg.state_dict_path, 
                                                  net, 
                                                  cfg.optim.n_epochs, 
                                                  device)
    # When only generating BNN predictions (and not running MCMC), we can afford 
    # more n_dropout. Otherwise, we fix n_dropout = mcmc_Y_dim + 1
    if test_cfg.export.pred:
        n_dropout = 20
        n_samples_per_dropout = test_cfg.numerics.mcmc.walkerRatio
    else:
        # # (BNN params + D_dt) times walker ratio
        n_walkers = test_cfg.numerics.mcmc.walkerRatio*(param_logL.Y_dim + 1) 
        n_dropout = n_walkers//test_cfg.numerics.mcmc.walkerRatio
        n_samples_per_dropout = test_cfg.numerics.mcmc.walkerRatio
    # Initialize arrays that will store samples and BNN predictions
    bnn_post = getattr(
                       h0rton.h0_inference.gaussian_bnn_posterior_cpu, 
                       loss_fn.posterior_name + 'CPU'
                       )(
                       param_logL.Y_dim,  
                       param_logL.mcmc_train_Y_mean, 
                       param_logL.mcmc_train_Y_std)
    init_pos, mcmc_pred = infer_bnn(
                                    net, bnn_post, param_logL, test_loader, 
                                    batch_size, n_dropout, n_samples_per_dropout, 
                                    device, test_cfg.global_seed,
                                    test_cfg.lens_posterior_type
                                    )
    # Terminate right after generating BNN predictions (no MCMC)
    if test_cfg.export.pred:
        samples_path = os.path.join(out_dir, 'samples.npy')
        np.save(samples_path, init_pos)
        sys.exit()

    #############
    # MCMC loop #
    #############
    # Convolve MC dropout iterates with aleatoric samples
    init_pos = init_pos.transpose(0, 3, 1, 2).reshape([batch_size, param_logL.Y_dim, -1]).transpose(0, 2, 1) # [batch_size, n_samples, mcmc_Y_dim]
    init_D_dt = np.random.uniform(0.0, 15000.0, size=(batch_size, n_walkers, 1))
    pred_mean = np.mean(init_pos, axis=1) # [batch_size, mcmc_Y_dim]
    # Define assumed model profiles
    kwargs_model = dict(lens_model_list=['PEMD', 'SHEAR'],
                        point_source_model_list=['SOURCE_POSITION'],
                        source_light_model_list=['SERSIC_ELLIPSE'])
    astro_sig = test_cfg.image_position_likelihood.sigma # astrometric uncertainty
    # Get H0 samples for each system
    if not test_cfg.time_delay_likelihood.baobab_time_delays:
        if 'abcd_ordering_i' not in master_truth:
            raise ValueError("If the time delay measurements were not generated using Baobab, the user must specify the order of image positions in which the time delays are listed, in order of increasing dec.")
    kwargs_lens_eqn_solver = {'min_distance': 0.05, 'search_window': baobab_cfg.instrument['pixel_scale']*baobab_cfg.image['num_pix'], 'num_iter_max': 200}

    total_progress = tqdm(total=n_test)
    # For each lens system...
    for i, lens_i in enumerate(lens_range):
        rs_lens = np.random.RandomState(lens_i) # replaced with externally rendered time delays
        ###########################
        # Relevant data and prior #
        ###########################
        data_i = master_truth.iloc[lens_i].copy()
        param_logL.set_bnn_post_params(mcmc_pred[lens_i, :]) # set the BNN parameters
        # Init values for the lens model params
        init_info = dict(zip(param_logL.Y_cols, pred_mean[lens_i, :]*param_logL.mcmc_train_Y_std + param_logL.mcmc_train_Y_mean))
        lcdm = LCDM(z_lens=data_i['z_lens'], z_source=data_i['z_src'], flat=True)
        # Data is BCD - A with a certain ABCD ordering, so inferred time delays should follow this convention.
        measured_td_wrt0 = np.array(data_i['measured_td']) # [n_img - 1,]
        measured_td_sig = np.array(data_i['measured_td_err']) # [n_img - 1,]
        abcd_ordering_i = np.array(data_i['abcd_ordering_i'])
        n_img = len(abcd_ordering_i)
        kwargs_data_joint = dict(time_delays_measured=measured_td_wrt0,
                                 time_delays_uncertainties=measured_td_sig,
                                 abcd_ordering_i=abcd_ordering_i,
                                 #vel_disp_measured=measured_vd, # TODO: optionally exclude
                                 #vel_disp_uncertainty=vel_disp_sig,
                                 )
        if not test_cfg.h0_posterior.exclude_velocity_dispersion:
            measured_vd = data_i['true_vd']*(1.0 + rs_lens.randn()*test_cfg.error_model.velocity_dispersion_frac_error)
            kwargs_data_joint['vel_disp_measured'] = measured_vd
            kwargs_data_joint['vel_disp_sig'] = test_cfg.velocity_dispersion_likelihood.sigma

        #############################
        # Parameter init and bounds #
        #############################
        lens_kwargs = mcmc_utils.get_lens_kwargs(init_info)
        ps_kwargs = mcmc_utils.get_ps_kwargs_src_plane(init_info, astro_sig)
        src_light_kwargs = mcmc_utils.get_light_kwargs(init_info['src_light_R_sersic'])
        special_kwargs = mcmc_utils.get_special_kwargs(n_img, astro_sig) # image position offset and time delay distance, aka the "special" parameters
        kwargs_params = {'lens_model': lens_kwargs,
                         'point_source_model': ps_kwargs,
                         'source_model': src_light_kwargs,
                         'special': special_kwargs,}
        if test_cfg.numerics.solver_type == 'NONE':
            solver_type = 'NONE'
        else:
            solver_type = 'PROFILE_SHEAR' if n_img == 4 else 'CENTER'
        #solver_type = 'NONE'
        kwargs_constraints = {'num_point_source_list': [n_img],  
                              'Ddt_sampling': True,
                              'solver_type': solver_type,}

        kwargs_likelihood = {'time_delay_likelihood': True,
                             'sort_images_by_dec': True,
                             'prior_lens': [],
                             'prior_special': [],
                             'check_bounds': True, 
                             'check_matched_source_position': False,
                             'source_position_tolerance': 0.01,
                             'source_position_sigma': 0.01,
                             'source_position_likelihood': False,
                             'custom_logL_addition': param_logL.evaluate,
                             'kwargs_lens_eqn_solver': kwargs_lens_eqn_solver}

        ###########################
        # MCMC posterior sampling #
        ###########################
        fitting_seq = FittingSequence(kwargs_data_joint, kwargs_model, 
                                      kwargs_constraints, kwargs_likelihood, 
                                      kwargs_params, verbose=False, mpi=False)
        if i == 0:
            param_class = fitting_seq._updateManager.param_class
            n_params, param_class_Y_cols = param_class.num_param()
            init_pos = mcmc_utils.reorder_to_param_class(param_logL.Y_cols, 
                                                         param_class_Y_cols, 
                                                         init_pos, init_D_dt)
        # MCMC sample from the post-processed BNN posterior jointly with cosmology
        lens_i_start_time = time.time()
        if test_cfg.lens_posterior_type == 'default':
            test_cfg.numerics.mcmc.update(init_samples=init_pos[lens_i, :, :])
        fitting_kwargs_list_mcmc = [['MCMC', test_cfg.numerics.mcmc]]
        #try:
        #with script_utils.HiddenPrints():
        chain_list_mcmc = fitting_seq.fit_sequence(fitting_kwargs_list_mcmc)
        kwargs_result_mcmc = fitting_seq.best_fit()
        lens_i_end_time = time.time()
        inference_time = (lens_i_end_time - lens_i_start_time)/60.0 # min

        #############################
        # Plotting the MCMC samples #
        #############################
        # sampler_type : 'EMCEE'
        # samples_mcmc : np.array of shape `[n_mcmc_eval, n_params]`
        # param_mcmc : list of str of length n_params, the parameter names
        sampler_type, samples_mcmc, param_mcmc, _  = chain_list_mcmc[0]
        new_samples_mcmc = mcmc_utils.postprocess_mcmc_chain(kwargs_result_mcmc, samples_mcmc, kwargs_model, lens_kwargs[2], ps_kwargs[2], src_light_kwargs[2], special_kwargs[2], kwargs_constraints)
        # Plot D_dt histogram
        D_dt_samples = new_samples_mcmc['D_dt'].values
        true_D_dt = lcdm.D_dt(H_0=data_i['H0'], Om0=0.27)
        data_i['D_dt'] = true_D_dt
        # Export D_dt samples for this lens
        lens_inference_dict = dict(
                                   D_dt_samples=D_dt_samples, # kappa_ext=0 for these samples
                                   inference_time=inference_time,
                                   true_D_dt=true_D_dt, 
                                   )
        lens_inference_dict_save_path = os.path.join(out_dir, 'D_dt_dict_{0:04d}.npy'.format(lens_i))
        np.save(lens_inference_dict_save_path, lens_inference_dict)
        # Optionally export the MCMC samples
        if test_cfg.export.mcmc_samples:
            mcmc_samples_path = os.path.join(out_dir, 'mcmc_samples_{0:04d}.csv'.format(lens_i))
            new_samples_mcmc.to_csv(mcmc_samples_path, index=None)
        # Optionally export the D_dt histogram
        if test_cfg.export.D_dt_histogram:
            cleaned_D_dt_samples = h0_utils.remove_outliers_from_lognormal(D_dt_samples, 3)
            _ = plotting_utils.plot_D_dt_histogram(cleaned_D_dt_samples, lens_i, true_D_dt, save_dir=out_dir)
        # Optionally export the plot of MCMC chain
        if test_cfg.export.mcmc_chain:
            mcmc_chain_path = os.path.join(out_dir, 'mcmc_chain_{0:04d}.png'.format(lens_i))
            plotting_utils.plot_mcmc_chain(chain_list_mcmc, mcmc_chain_path)
        # Optionally export posterior cornerplot of select lens model parameters with D_dt
        if test_cfg.export.mcmc_corner:
            mcmc_corner_path = os.path.join(out_dir, 'mcmc_corner_{0:04d}.png'.format(lens_i))
            plotting_utils.plot_mcmc_corner(new_samples_mcmc[test_cfg.export.mcmc_cols], None, test_cfg.export.mcmc_col_labels, mcmc_corner_path)
        total_progress.update(1)
    total_progress.close()

if __name__ == '__main__':
    #import cProfile
    #pr = cProfile.Profile()
    #pr.enable()
    main()
    #pr.disable()
    #pr.print_stats(sort='cumtime')