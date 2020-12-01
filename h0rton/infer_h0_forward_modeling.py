"""Script to run MCMC cosmological sampling for individual lenses, using the BNN posterior

It borrows heavily from the `time-delay cosmography.ipynb` notebook in Lenstronomy Extensions, which you can find `here <https://github.com/sibirrer/lenstronomy_extensions/blob/master/lenstronomy_extensions/Notebooks/time-delay%20cosmography.ipynb>`_.

Example
-------
To run this script, pass in the path to the user-defined inference config file as the argument::
    
    $ python h0rton/infer_h0_forward_modeling.py forward_modeling.json

"""

import time
import warnings
import os
from tqdm import tqdm
import gc
from ast import literal_eval
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from lenstronomy.Cosmo.lcdm import LCDM
from lenstronomy.Plots.model_plot import ModelPlot
import baobab.sim_utils.metadata_utils as metadata_utils
from baobab import BaobabConfig
from h0rton.configs import TrainValConfig, TestConfig
import h0rton.script_utils as script_utils
from h0rton.h0_inference import h0_utils, plotting_utils, mcmc_utils
from h0rton.h0_inference.forward_modeling_posterior import ForwardModelingPosterior
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
    # Define val data and loader
    test_data = XYData(is_train=False, 
                       Y_cols=cfg.data.Y_cols, 
                       float_type=cfg.data.float_type, 
                       define_src_pos_wrt_lens=cfg.data.define_src_pos_wrt_lens, 
                       rescale_pixels=False, 
                       rescale_pixels_type=None,
                       log_pixels=False, 
                       add_pixel_noise=cfg.data.add_pixel_noise, 
                       eff_exposure_time={"TDLMC_F160W": test_cfg.data.eff_exposure_time}, 
                       train_Y_mean=np.zeros((1, len(cfg.data.Y_cols))), 
                       train_Y_std=np.ones((1, len(cfg.data.Y_cols))), 
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
        warnings.warn("Destination folder already exists.")

    ################
    # Compile data #
    ################
    # Image data
    with torch.no_grad():
        for X_, Y_ in test_loader:
            X = X_.to(device)
            break
    X = X.detach().cpu().numpy()

    #############
    # MCMC loop #
    #############
    kwargs_lens_eqn_solver = dict(
                                  min_distance=0.05,
                                  search_window=baobab_cfg.instrument['pixel_scale']*baobab_cfg.image['num_pix'],
                                  num_iter_max=200
                                  )
    fm_posterior = ForwardModelingPosterior(kwargs_lens_eqn_solver=kwargs_lens_eqn_solver,
                                            astrometric_sigma=test_cfg.image_position_likelihood.sigma,
                                            supersampling_factor=baobab_cfg.numerics.supersampling_factor)
    # Get H0 samples for each system
    if not test_cfg.time_delay_likelihood.baobab_time_delays:
        if 'abcd_ordering_i' not in master_truth:
            raise ValueError("If the time delay measurements were not generated using Baobab, the user must specify the order of image positions in which the time delays are listed, in order of increasing dec.")

    total_progress = tqdm(total=n_test)
    realized_time_delays = pd.read_csv(test_cfg.error_model.realized_time_delays, index_col=None)
    # For each lens system...
    for i, lens_i in enumerate(lens_range):
        ###########################
        # Relevant data and prior #
        ###########################
        data_i = master_truth.iloc[lens_i].copy()
        lcdm = LCDM(z_lens=data_i['z_lens'], z_source=data_i['z_src'], flat=True)
        measured_td_wrt0 = np.array(literal_eval(realized_time_delays.iloc[lens_i]['measured_td_wrt0']))
        n_img = len(measured_td_wrt0) + 1
        #print(baobab_cfg.survey_object_dict)
        fm_posterior.set_kwargs_data_joint(
                                           image=X[lens_i, 0, :, :],
                                           measured_td=measured_td_wrt0,
                                           measured_td_sigma=test_cfg.time_delay_likelihood.sigma,
                                           survey_object_dict=baobab_cfg.survey_object_dict,
                                           eff_exposure_time=test_cfg.data.eff_exposure_time,
                                           )
        # Update solver according to number of lensed images 
        if test_cfg.numerics.solver_type == 'NONE':
            fm_posterior.kwargs_constraints['solver_type'] = 'NONE'
        else:
            fm_posterior.kwargs_constraints['solver_type'] = 'PROFILE_SHEAR' if n_img == 4 else 'ELLIPSE'
        fm_posterior.kwargs_constraints['num_point_source_list'] = [n_img]
        #print(fm_posterior.kwargs_params['point_source_model'][0][0])
        true_D_dt = lcdm.D_dt(H_0=data_i['H0'], Om0=0.3)
        # Pull truth param values and initialize walkers there
        if test_cfg.numerics.initialize_walkers_to_truth:
            fm_posterior.kwargs_lens_init = metadata_utils.get_kwargs_lens_mass(data_i)
            fm_posterior.kwargs_lens_light_init = metadata_utils.get_kwargs_lens_light(data_i)
            fm_posterior.kwargs_source_init = metadata_utils.get_kwargs_src_light(data_i)
            fm_posterior.kwargs_ps_init = metadata_utils.get_kwargs_ps_lensed(data_i)
            fm_posterior.kwargs_special_init = dict(D_dt=true_D_dt)

        ###########################
        # MCMC posterior sampling #
        ###########################
        lens_i_start_time = time.time()
        #with script_utils.HiddenPrints():
        chain_list_mcmc, kwargs_result_mcmc = fm_posterior.run_mcmc(test_cfg.numerics.mcmc)
        lens_i_end_time = time.time()
        inference_time = (lens_i_end_time - lens_i_start_time)/60.0 # min

        #############################
        # Plotting the MCMC samples #
        #############################
        # sampler_type : 'EMCEE'
        # samples_mcmc : np.array of shape `[n_mcmc_eval, n_params]`
        # param_mcmc : list of str of length n_params, the parameter names
        sampler_type, samples_mcmc, param_mcmc, _  = chain_list_mcmc[0]
        new_samples_mcmc = mcmc_utils.postprocess_mcmc_chain(kwargs_result_mcmc, 
                                                             samples_mcmc, 
                                                             fm_posterior.kwargs_model, 
                                                             fm_posterior.kwargs_params['lens_model'][2], 
                                                             fm_posterior.kwargs_params['point_source_model'][2], 
                                                             fm_posterior.kwargs_params['source_model'][2], 
                                                             fm_posterior.kwargs_params['special'][2], 
                                                             fm_posterior.kwargs_constraints,
                                                             kwargs_fixed_lens_light=fm_posterior.kwargs_params['lens_light_model'][2],
                                                             verbose=False
                                                             )
        #from lenstronomy.Plots import chain_plot
        model_plot = ModelPlot(fm_posterior.multi_band_list, 
                              fm_posterior.kwargs_model, kwargs_result_mcmc, arrow_size=0.02, cmap_string="gist_heat")
        plotting_utils.plot_forward_modeling_comparisons(model_plot, out_dir)

        # Plot D_dt histogram
        D_dt_samples = new_samples_mcmc['D_dt'].values # may contain negative values
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