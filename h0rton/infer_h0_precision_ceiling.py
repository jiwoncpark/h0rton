"""Script to run an MCMC afterburner for the BNN posterior

It borrows heavily from the `catalogue modelling.ipynb` notebook in Lenstronomy Extensions, which you can find `here <https://github.com/sibirrer/lenstronomy_extensions/blob/master/lenstronomy_extensions/Notebooks/catalogue%20modelling.ipynb>`_.

"""
import os
import time
from tqdm import tqdm
import glob
import json
from baobab import BaobabConfig
from addict import Dict
from ast import literal_eval
import numpy as np
import gc
import torch
from lenstronomy.Workflow.fitting_sequence import FittingSequence
from lenstronomy.Cosmo.lcdm import LCDM
import baobab.sim_utils.metadata_utils as metadata_utils
from h0rton.script_utils import parse_args, seed_everything, HiddenPrints
from h0rton.configs import TrainValConfig, TestConfig
from h0rton.h0_inference import h0_utils, plotting_utils, mcmc_utils
from h0rton.trainval_data import XYCosmoData

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
    train_val_cfg = TrainValConfig.from_file(test_cfg.train_val_config_file_path)
    baobab_cfg = get_baobab_config(test_cfg.data.test_dir)
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
    master_truth = test_data.cosmo_df
    master_truth = metadata_utils.add_qphi_columns(master_truth)
    master_truth = metadata_utils.add_gamma_psi_ext_columns(master_truth)
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
    # Instantiate loss function, to append to the MCMC objective as the prior
    orig_Y_cols = train_val_cfg.data.Y_cols
    # Instantiate MCMC parameter penalty function
    params_to_remove = ['lens_light_R_sersic'] #'src_light_R_sersic'] 
    mcmc_Y_cols = [col for col in orig_Y_cols if col not in params_to_remove]
    mcmc_Y_dim = len(mcmc_Y_cols)
    null_spread = True
    #init_D_dt = np.random.uniform(0.0, 10000.0, size=(batch_size, n_walkers, 1)) # FIXME: init H0 hardcoded

    kwargs_model = dict(lens_model_list=['PEMD', 'SHEAR'],
                        point_source_model_list=['SOURCE_POSITION'],
                        source_light_model_list=['SERSIC_ELLIPSE'])
    astro_sig = test_cfg.image_position_likelihood.sigma
    # Get H0 samples for each system
    if not test_cfg.time_delay_likelihood.baobab_time_delays:
        if 'abcd_ordering_i' not in master_truth:
            raise ValueError("If the time delay measurements were not generated using Baobab, the user must specify the order of image positions in which the time delays are listed, in order of increasing dec.")
    kwargs_lens_eq_solver = {'min_distance': 0.05, 'search_window': baobab_cfg.instrument.pixel_scale*baobab_cfg.image.num_pix, 'num_iter_max': 100}
    #n_walkers = test_cfg.numerics.mcmc.walkerRatio*(mcmc_Y_dim + 1) # BNN params + H0 times walker ratio
    #init_pos = np.tile(master_truth[mcmc_Y_cols].iloc[:batch_size].values[:, np.newaxis, :], [1, n_walkers, 1])
    #init_D_dt = np.random.uniform(0.0, 10000.0, size=(batch_size, n_walkers, 1))
    #print(init_pos.shape, init_D_dt.shape)

    total_progress = tqdm(total=n_test)
    # For each lens system...
    for i, lens_i in enumerate(lens_range):
        # Each lens gets a unique random state for td and vd measurement error realizations.
        rs_lens = np.random.RandomState(lens_i)
        ###########################
        # Relevant data and prior #
        ###########################
        data_i = master_truth.iloc[lens_i].copy()
        # Init values for the lens model params
        init_info = dict(zip(mcmc_Y_cols, data_i[mcmc_Y_cols].values)) # truth params
        lcdm = LCDM(z_lens=data_i['z_lens'], z_source=data_i['z_src'], flat=True)
        true_img_dec = np.array(literal_eval(data_i['y_image']))
        n_img = len(true_img_dec)
        true_td = np.array(literal_eval(data_i['true_td']))
        measured_td = true_td + rs_lens.randn(*true_td.shape)*test_cfg.error_model.time_delay_error
        measured_td_sig = test_cfg.time_delay_likelihood.sigma # np.ones(n_img - 1)*
        measured_img_dec = true_img_dec + rs_lens.randn(n_img)*astro_sig
        increasing_dec_i = np.argsort(true_img_dec) #np.argsort(measured_img_dec)
        measured_td = h0_utils.reorder_to_tdlmc(measured_td, increasing_dec_i, range(n_img)) # need to use measured dec to order
        measured_img_dec = h0_utils.reorder_to_tdlmc(measured_img_dec, increasing_dec_i, range(n_img))
        measured_td_wrt0 = measured_td[1:] - measured_td[0]   
        kwargs_data_joint = dict(time_delays_measured=measured_td_wrt0,
                                 time_delays_uncertainties=measured_td_sig,
                                 )

        #############################
        # Parameter init and bounds #
        #############################
        lens_kwargs = mcmc_utils.get_lens_kwargs(init_info, null_spread=null_spread)
        ps_kwargs = mcmc_utils.get_ps_kwargs_src_plane(init_info, astro_sig, null_spread=null_spread)
        src_light_kwargs = mcmc_utils.get_light_kwargs(init_info['src_light_R_sersic'], null_spread=null_spread)
        special_kwargs = mcmc_utils.get_special_kwargs(n_img, astro_sig, D_dt_sigma=2000, null_spread=null_spread) # image position offset and time delay distance, aka the "special" parameters
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
                             'custom_logL_addition': None,
                             'kwargs_lens_eq_solver': kwargs_lens_eq_solver}

        ###########################
        # MCMC posterior sampling #
        ###########################
        fitting_seq = FittingSequence(kwargs_data_joint, kwargs_model, kwargs_constraints, kwargs_likelihood, kwargs_params, verbose=False, mpi=False)
        if i == 0:
            param_class = fitting_seq._updateManager.param_class
            n_params, param_class_Y_cols = param_class.num_param()
            #init_pos = mcmc_utils.reorder_to_param_class(mcmc_Y_cols, param_class_Y_cols, init_pos, init_D_dt)
        # MCMC sample from the post-processed BNN posterior jointly with cosmology
        lens_i_start_time = time.time()
        #test_cfg.numerics.mcmc.update(init_samples=init_pos[lens_i, :, :])
        fitting_kwargs_list_mcmc = [['MCMC', test_cfg.numerics.mcmc]]
        #with HiddenPrints():
        #try:
        chain_list_mcmc = fitting_seq.fit_sequence(fitting_kwargs_list_mcmc)
        kwargs_result_mcmc = fitting_seq.best_fit()
        #except:
        #    print("lens {:d} skipped".format(lens_i))
        #    total_progress.update(1)
        #    continue
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
    total_progress.close()

if __name__ == '__main__':
    #import cProfile
    #pr = cProfile.Profile()
    #pr.enable()
    main()
    #pr.disable()
    #pr.print_stats(sort='cumtime')