"""Script to run MCMC cosmological sampling for individual lenses, using the BNN posterior

It borrows heavily from the `catalogue modelling.ipynb` notebook in Lenstronomy Extensions, which you can find `here <https://github.com/sibirrer/lenstronomy_extensions/blob/master/lenstronomy_extensions/Notebooks/catalogue%20modelling.ipynb>`_.

Example
-------
To run this script, pass in the path to the user-defined inference config file as the argument::
    
    $ python h0rton/infer_h0_mcmc_default.py mcmc_default.json

"""
import os
import time
from tqdm import tqdm
import gc
from ast import literal_eval
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from lenstronomy.Workflow.fitting_sequence import FittingSequence
from lenstronomy.Cosmo.lcdm import LCDM
import baobab.sim_utils.metadata_utils as metadata_utils
from baobab import BaobabConfig
import h0rton.models
from h0rton.configs import TrainValConfig, TestConfig
import h0rton.losses
import h0rton.train_utils as train_utils
import h0rton.script_utils as script_utils
from h0rton.h0_inference import h0_utils, plotting_utils, mcmc_utils
from h0rton.trainval_data import XYData

def main():
    args = script_utils.parse_inference_args()
    test_cfg = TestConfig.from_file(args.test_config_file_path)
    baobab_cfg = BaobabConfig.from_file(test_cfg.data.test_baobab_cfg_path)
    cfg = TrainValConfig.from_file(test_cfg.train_val_config_file_path)
    # Set device and default data type
    device = torch.device(test_cfg.device_type)
    if device.type == 'cuda':
        torch.set_default_tensor_type('torch.cuda.' + cfg.data.float_type)
    else:
        torch.set_default_tensor_type('torch.' + cfg.data.float_type)
    script_utils.seed_everything(test_cfg.global_seed)
    
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
    # Define val data and loader
    test_data = XYData(is_train=False, 
                       Y_cols=cfg.data.Y_cols, 
                       float_type=cfg.data.float_type, 
                       define_src_pos_wrt_lens=cfg.data.define_src_pos_wrt_lens, 
                       rescale_pixels=cfg.data.rescale_pixels, 
                       rescale_pixels_type=cfg.data.rescale_pixels_type,
                       log_pixels=cfg.data.log_pixels, 
                       add_pixel_noise=cfg.data.add_pixel_noise, 
                       eff_exposure_time=cfg.data.eff_exposure_time, 
                       train_Y_mean=train_data.train_Y_mean, 
                       train_Y_std=train_data.train_Y_std, 
                       train_baobab_cfg_path=cfg.data.train_baobab_cfg_path, 
                       val_baobab_cfg_path=test_cfg.data.test_baobab_cfg_path, 
                       for_cosmology=True)
    master_truth = test_data.Y_df
    master_truth = metadata_utils.add_qphi_columns(master_truth)
    master_truth = metadata_utils.add_gamma_psi_ext_columns(master_truth)
    # Figure out how many lenses BNN will predict on (must be consecutive)
    if test_cfg.data.lens_indices is None:
        if args.lens_indices_path is None:
            # Test on all n_test lenses in the test set
            n_test = test_cfg.data.n_test 
            lens_range = range(n_test)
        else:
            # Test on the lens indices in a text file at the specified path
            lens_range = []
            with open(args.lens_indices_path, "r") as f:
                for line in f:
                    lens_range.append(int(line.strip()))
            n_test = len(lens_range)
            print("Performing H0 inference on {:d} specified lenses...".format(n_test))
    else:
        if args.lens_indices_path is None:
            # Test on the lens indices specified in the test config file
            lens_range = test_cfg.data.lens_indices
            n_test = len(lens_range)
            print("Performing H0 inference on {:d} specified lenses...".format(n_test))
        else:
            raise ValueError("Specific lens indices were specified in both the test config file and the command-line argument.")
    batch_size = max(lens_range) + 1
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, drop_last=True)
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
    loss_fn = getattr(h0rton.losses, cfg.model.likelihood_class)(Y_dim=test_data.Y_dim, 
                                                                 device=device)
    # Not all predicted params will be sampled via MCMC
    params_to_remove = [] #'lens_light_R_sersic', 'src_light_R_sersic'] 
    mcmc_Y_cols = [col for col in orig_Y_cols if col not in params_to_remove]
    mcmc_Y_dim = len(mcmc_Y_cols)
    # Instantiate loss function with just the MCMC params
    mcmc_loss_fn = getattr(h0rton.losses, cfg.model.likelihood_class)(Y_dim=test_data.Y_dim - len(params_to_remove), 
                                                                      device=device)
    remove_param_idx, remove_idx = mcmc_utils.get_idx_for_params(mcmc_loss_fn.out_dim, 
                                                                 orig_Y_cols, 
                                                                 params_to_remove, 
                                                                 cfg.model.likelihood_class)
    mcmc_train_Y_mean = np.delete(train_data.train_Y_mean, remove_param_idx)
    mcmc_train_Y_std = np.delete(train_data.train_Y_std, remove_param_idx)
    parameter_penalty = mcmc_utils.HybridBNNPenalty(mcmc_Y_cols, cfg.model.likelihood_class, mcmc_train_Y_mean, mcmc_train_Y_std, test_cfg.h0_posterior.exclude_velocity_dispersion, device)
    custom_logL_addition = parameter_penalty.evaluate
    null_spread = False

    ###################
    # BNN predictions #
    ###################
    # Instantiate BNN model
    net = getattr(h0rton.models, cfg.model.architecture)(num_classes=loss_fn.out_dim, dropout_rate=cfg.model.dropout_rate)
    net.to(device)
    # Load trained weights from saved state
    net, epoch = train_utils.load_state_dict_test(test_cfg.state_dict_path, net, cfg.optim.n_epochs, device)
    # When only generating BNN predictions (and not running MCMC), we can afford more n_dropout
    # otherwise, we fix n_dropout = mcmc_Y_dim + 1
    if test_cfg.export.pred:
        n_dropout = 20
        n_samples_per_dropout = test_cfg.numerics.mcmc.walkerRatio
    else:
        n_walkers = test_cfg.numerics.mcmc.walkerRatio*(mcmc_Y_dim + 1) # (BNN params + D_dt) times walker ratio
        n_dropout = n_walkers//test_cfg.numerics.mcmc.walkerRatio
        n_samples_per_dropout = test_cfg.numerics.mcmc.walkerRatio
    # Initialize arrays that will store samples and BNN predictions
    init_pos = np.empty([batch_size, n_dropout, n_samples_per_dropout, mcmc_Y_dim])
    mcmc_pred = np.empty([batch_size, n_dropout, mcmc_loss_fn.out_dim])
    with torch.no_grad():
        net.train()
        # Send some empty forward passes through the test data without backprop to adjust batchnorm weights
        # (This is often not necessary. Beware if using for just 1 lens.)
        for nograd_pass in range(5):
            for X_, Y_ in test_loader:
                X = X_.to(device)
                _ = net(X)
        # Obtain MC dropout samples
        for d in range(n_dropout):
            net.eval()
            for X_, Y_ in test_loader:
                X = X_.to(device)
                Y = Y_.to(device)
                pred = net(X)
                break
            mcmc_pred_d = pred.cpu().numpy()
            # Replace BNN posterior's primary gaussian mean with truth values
            if test_cfg.lens_posterior_type == 'default_with_truth_mean':
                mcmc_pred_d[:, :len(mcmc_Y_cols)] = Y[:, :len(mcmc_Y_cols)].cpu().numpy()
            # Leave only the MCMC parameters in pred
            mcmc_pred_d = mcmc_utils.remove_parameters_from_pred(mcmc_pred_d, remove_idx, return_as_tensor=False)
            # Populate pred that will define the MCMC penalty function
            mcmc_pred[:, d, :] = mcmc_pred_d
            # Instantiate posterior to generate BNN samples, which will serve as initial positions for walkers
            bnn_post = getattr(h0rton.h0_inference.gaussian_bnn_posterior_cpu, loss_fn.posterior_name + 'CPU')(mcmc_Y_dim,  mcmc_train_Y_mean, mcmc_train_Y_std)
            bnn_post.set_sliced_pred(mcmc_pred_d)
            init_pos[:, d, :, :] = bnn_post.sample(n_samples_per_dropout, sample_seed=test_cfg.global_seed+d) # contains just the lens model params, no D_dt
            gc.collect()
    # Terminate right after generating BNN predictions (no MCMC)
    if test_cfg.export.pred:
        import sys
        samples_path = os.path.join(out_dir, 'samples.npy')
        np.save(samples_path, init_pos)
        sys.exit()

    #############
    # MCMC loop #
    #############
    # Convolve MC dropout iterates with aleatoric samples
    init_pos = init_pos.transpose(0, 3, 1, 2).reshape([batch_size, mcmc_Y_dim, -1]).transpose(0, 2, 1) # [batch_size, n_samples, mcmc_Y_dim]
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
    realized_time_delays = pd.read_csv(test_cfg.error_model.realized_time_delays, index_col=None)
    # For each lens system...
    for i, lens_i in enumerate(lens_range):
        # Each lens gets a unique random state for time delay measurement error realizations.
        #rs_lens = np.random.RandomState(lens_i) # replaced with externally rendered time delays
        ###########################
        # Relevant data and prior #
        ###########################
        data_i = master_truth.iloc[lens_i].copy()
        # Set BNN pred defining parameter penalty for this lens, batch processes across n_dropout
        parameter_penalty.set_bnn_post_params(mcmc_pred[lens_i, :, :])
        # Initialize lens model params walkers at the predictive mean
        init_info = dict(zip(mcmc_Y_cols, pred_mean[lens_i, :]*mcmc_train_Y_std + mcmc_train_Y_mean))
        lcdm = LCDM(z_lens=data_i['z_lens'], z_source=data_i['z_src'], flat=True)
        true_img_dec = literal_eval(data_i['y_image'])
        n_img = len(true_img_dec)
        measured_td_sig = test_cfg.time_delay_likelihood.sigma
        measured_td_wrt0 = np.array(literal_eval(realized_time_delays.iloc[lens_i]['measured_td_wrt0']))
        kwargs_data_joint = dict(
                                 time_delays_measured=measured_td_wrt0,
                                 time_delays_uncertainties=measured_td_sig,
                                 )

        #############################
        # Parameter init and bounds #
        #############################
        lens_kwargs = mcmc_utils.get_lens_kwargs(init_info, null_spread=null_spread)
        ps_kwargs = mcmc_utils.get_ps_kwargs_src_plane(init_info, astro_sig)
        src_light_kwargs = mcmc_utils.get_light_kwargs(init_info['src_light_R_sersic'], null_spread=null_spread)
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
                             'custom_logL_addition': custom_logL_addition,
                             'kwargs_lens_eqn_solver': kwargs_lens_eqn_solver}

        ###########################
        # MCMC posterior sampling #
        ###########################
        fitting_seq = FittingSequence(kwargs_data_joint, kwargs_model, kwargs_constraints, kwargs_likelihood, kwargs_params, verbose=False, mpi=False)
        if i == 0:
            param_class = fitting_seq._updateManager.param_class
            n_params, param_class_Y_cols = param_class.num_param()
            init_pos = mcmc_utils.reorder_to_param_class(mcmc_Y_cols, param_class_Y_cols, init_pos, init_D_dt)
        # MCMC sample from the post-processed BNN posterior jointly with cosmology
        lens_i_start_time = time.time()
        if test_cfg.lens_posterior_type == 'default':
            test_cfg.numerics.mcmc.update(init_samples=init_pos[lens_i, :, :])
        fitting_kwargs_list_mcmc = [['MCMC', test_cfg.numerics.mcmc]]
        #try:
        with script_utils.HiddenPrints():
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
        true_D_dt = lcdm.D_dt(H_0=data_i['H0'], Om0=0.3)
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
            plotting_utils.plot_mcmc_corner(new_samples_mcmc[test_cfg.export.mcmc_cols], data_i[test_cfg.export.mcmc_cols], test_cfg.export.mcmc_col_labels, mcmc_corner_path)
        total_progress.update(1)
        gc.collect()
    realized_time_delays.to_csv(os.path.join(out_dir, 'realized_time_delays.csv'), index=None)
    total_progress.close()

if __name__ == '__main__':
    #import cProfile
    #pr = cProfile.Profile()
    #pr.enable()
    main()
    #pr.disable()
    #pr.print_stats(sort='cumtime')