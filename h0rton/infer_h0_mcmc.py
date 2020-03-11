"""Script to run an MCMC afterburner for the BNN posterior

It borrows heavily from the `catalogue modelling.ipynb` notebook in Lenstronomy Extensions, which you can find `here <https://github.com/sibirrer/lenstronomy_extensions/blob/master/lenstronomy_extensions/Notebooks/catalogue%20modelling.ipynb>`_.

"""

import os
import sys
import argparse
import random
import corner
import time
from addict import Dict
from tqdm import tqdm
from ast import literal_eval
import numpy as np
import torch
from torch.utils.data import DataLoader
sys.path.insert(0, '/home/jwp/stage/sl/lenstronomy')
import lenstronomy
print(lenstronomy.__path__)
from lenstronomy.Plots import chain_plot as chain_plot
from lenstronomy.Sampling.parameters import Param
from lenstronomy.Util import param_util
from lenstronomy.Workflow.fitting_sequence import FittingSequence
from lenstronomy.Cosmo.lcdm import LCDM
# H0rton modules
import h0rton.models
from h0rton.configs import TrainValConfig, TestConfig
import h0rton.losses
import h0rton.train_utils as train_utils
from h0rton.h0_inference import h0_utils, plotting_utils
from h0rton.trainval_data import XYCosmoData
import matplotlib.pyplot as plt

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

class HiddenPrints:
    """Hide standard output

    """
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

def main():
    args = parse_args()
    test_cfg = TestConfig.from_file(args.test_config_file_path)
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
    master_truth = test_data.cosmo_df
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
    orig_Y_cols = train_val_cfg.data.Y_cols
    loss_fn = getattr(h0rton.losses, train_val_cfg.model.likelihood_class)(Y_dim=train_val_cfg.data.Y_dim, device=device)
    # Instantiate MCMC parameter penalty function
    params_to_remove = ['src_light_center_x', 'src_light_center_y', 'lens_light_R_sersic', 'src_light_R_sersic'] # must be removed, as the post-processing scheme only involves image positions
    mcmc_Y_cols = [col for col in orig_Y_cols if col not in params_to_remove]
    mcmc_loss_fn = getattr(h0rton.losses, train_val_cfg.model.likelihood_class)(Y_dim=train_val_cfg.data.Y_dim - len(params_to_remove), device=device)
    remove_param_idx, remove_idx = h0_utils.get_idx_for_params(mcmc_loss_fn.out_dim, orig_Y_cols, params_to_remove)
    mcmc_train_Y_mean = np.delete(train_val_cfg.data.train_Y_mean, remove_param_idx)
    mcmc_train_Y_std = np.delete(train_val_cfg.data.train_Y_std, remove_param_idx)
    parameter_penalty = h0_utils.HybridBNNPenalty(mcmc_Y_cols, train_val_cfg.model.likelihood_class, mcmc_train_Y_mean, mcmc_train_Y_std, test_cfg.h0_posterior.exclude_velocity_dispersion, device)
    # Instantiate model
    net = getattr(h0rton.models, train_val_cfg.model.architecture)(num_classes=loss_fn.out_dim)
    net.to(device)
    # Load trained weights from saved state
    net, epoch = train_utils.load_state_dict_test(test_cfg.state_dict_path, net, train_val_cfg.optim.n_epochs, device)
    with torch.no_grad():
        net.eval()
        for X_, Y_ in test_loader:
            X = X_.to(device)
            Y = Y_.to(device) # TODO: compare master_truth with reverse-transformed Y
            pred = net(X)
            break
    mcmc_pred = h0_utils.remove_parameters_from_pred(pred.cpu().numpy(), remove_idx, return_as_tensor=True, device=device)
    
    # FIXME: hardcoded
    kwargs_model = dict(lens_model_list=['SPEMD', 'SHEAR'],
                        point_source_model_list=['LENSED_POSITION'],)
    astrometry_sigma = test_cfg.image_position_likelihood.sigma
    # Get H0 samples for each system
    if not test_cfg.time_delay_likelihood.baobab_time_delays:
        if 'abcd_ordering_i' not in master_truth:
            raise ValueError("If the time delay measurements were not generated using Baobab, the user must specify the order of image positions in which the time delays are listed, in order of increasing dec.")

    # Placeholders for mean and std of D_dt samples per system
    mean_D_dt_set = np.zeros(n_test)
    std_D_dt_set = np.zeros(n_test)
    inference_time_set = np.zeros(n_test)
    # For each lens system...
    total_progress = tqdm(total=n_test)
    lens_i_start_time = time.time()
    for i, lens_i in enumerate(lens_range):
        # Each lens gets a unique random state for td and vd measurement error realizations.
        rs_lens = np.random.RandomState(lens_i)

        ###########################
        # Relevant data and prior #
        ###########################
        data_i = master_truth.iloc[lens_i]
        parameter_penalty.set_bnn_post_params(mcmc_pred[lens_i, :]) # set the BNN parameters
        mu = dict(zip(mcmc_Y_cols, mcmc_pred.cpu().numpy()[lens_i, :len(mcmc_Y_cols)]*mcmc_train_Y_std + mcmc_train_Y_mean)) # mean of primary Gaussian in the BNN posterior will be used to initialize
        if not test_cfg.h0_posterior.exclude_velocity_dispersion:
            parameter_penalty.set_vel_disp_params()
            raise NotImplementedError
        lcdm = LCDM(z_lens=data_i['z_lens'], z_source=data_i['z_src'], flat=True)

        # Data accessible to likelihood function
        true_img_dec = np.trim_zeros(data_i[['y_image_0', 'y_image_1', 'y_image_2', 'y_image_3']].values, 'b')
        true_img_ra = np.trim_zeros(data_i[['x_image_0', 'x_image_1', 'x_image_2', 'x_image_3']].values, 'b')
        n_img = len(true_img_dec)
        true_td = np.array(literal_eval(data_i['true_td']))
        measured_td = true_td + rs_lens.randn(*true_td.shape)*test_cfg.error_model.time_delay_error
        measured_td_sig = np.ones(n_img - 1)*test_cfg.time_delay_likelihood.sigma
        measured_img_dec = true_img_dec + rs_lens.randn(n_img)*astrometry_sigma
        measured_img_ra = true_img_ra + rs_lens.randn(n_img)*astrometry_sigma
        reordered_measured_td = h0_utils.reorder_to_tdlmc(measured_td, np.argsort(measured_img_dec), range(n_img)) # need to use measured dec to order
        measured_td_wrt0 = reordered_measured_td[1:] - reordered_measured_td[0]   
        kwargs_data_joint = dict(time_delays_measured=measured_td_wrt0,
                                 time_delays_uncertainties=measured_td_sig,
                                 #vel_disp_measured=measured_vd, # TODO: optionally exclude
                                 #vel_disp_uncertainty=vel_disp_sig,
                                 ra_image_list=[measured_img_ra],
                                 dec_image_list=[measured_img_dec],)
        if not test_cfg.h0_posterior.exclude_velocity_dispersion:
            measured_vd = data_i['true_vd']*(1.0 + rs_lens.randn()*test_cfg.error_model.velocity_dispersion_frac_error)
            kwargs_data_joint['vel_disp_measured'] = measured_vd
            kwargs_data_joint['vel_disp_sig'] = test_cfg.velocity_dispersion_likelihood.sigma

        #############################
        # Parameter init and bounds #
        #############################
        # Lens parameters
        kwargs_init_lens = [{'theta_E': mu['lens_mass_theta_E'], 'gamma': mu['lens_mass_gamma'], 'center_x': mu['lens_mass_center_x'], 'center_y': mu['lens_mass_center_y'], 'e1': mu['lens_mass_e1'], 'e2': mu['lens_mass_e2']}, {'gamma1': mu['external_shear_gamma2'], 'gamma2': mu['external_shear_gamma2']}]
        kwargs_sigma_lens, kwargs_lower_lens, kwargs_upper_lens, kwargs_fixed_lens = h0_utils.get_misc_kwargs_lens()
        # AGN light parameters
        fixed_ps = [{}] 
        kwargs_ps_init = [{'ra_image': measured_img_ra, 'dec_image': measured_img_dec}]
        kwargs_ps_sigma = [{'ra_image': astrometry_sigma*np.ones(n_img), 'dec_image': astrometry_sigma*np.ones(n_img)}]
        kwargs_lower_ps = [{'ra_image': -10*np.ones(n_img), 'dec_image': -10*np.ones(n_img)}]
        kwargs_upper_ps = [{'ra_image': 10*np.ones(n_img), 'dec_image': 10*np.ones(n_img)}]
        # Image position offset and time delay distance, aka the "special" parameters
        fixed_special = {}
        kwargs_special_init = {'delta_x_image': np.zeros(n_img), 'delta_y_image': np.zeros(n_img), 'D_dt': 5000}
        kwargs_special_sigma = {'delta_x_image': np.ones(n_img)*astrometry_sigma, 'delta_y_image': np.ones(n_img)*astrometry_sigma, 'D_dt': 3000}
        kwargs_lower_special = {'delta_x_image': np.ones(n_img)*(-1.0), 'delta_y_image': np.ones(n_img)*(-1.0), 'D_dt': 0}
        kwargs_upper_special = {'delta_x_image': np.ones(n_img), 'delta_y_image': np.ones(n_img), 'D_dt': 10000}
        # Put all components together
        kwargs_params = {'lens_model': [kwargs_init_lens, kwargs_sigma_lens, kwargs_fixed_lens, kwargs_lower_lens, kwargs_upper_lens],
                         'point_source_model': [kwargs_ps_init, kwargs_ps_sigma, fixed_ps, kwargs_lower_ps, kwargs_upper_ps],
                         'special': [kwargs_special_init, kwargs_special_sigma, fixed_special, kwargs_lower_special, kwargs_upper_special],}
        kwargs_constraints = {'num_point_source_list': [n_img],  
                              'Ddt_sampling': True,
                              'solver_type': 'NONE',
                              #'joint_lens_with_lens': [[0, 1, ['center_x', 'ra_0']], [0, 1, ['center_y', 'dec_0']]],
                             }
        kwargs_likelihood = {'image_position_uncertainty': astrometry_sigma,
                             'image_position_likelihood': True,
                             'time_delay_likelihood': True,
                             'prior_lens': [],
                             'prior_special': [],
                             'check_bounds': True, 
                             'check_matched_source_position': True,
                             'source_position_tolerance': 0.01,
                             'source_position_sigma': 0.001,
                             'source_position_likelihood': False,
                             'custom_logL_addition': parameter_penalty.evaluate,}

        ###########################
        # MCMC posterior sampling #
        ###########################
        #kwargs_data_joint : dictionary of data, e.g. measured_td, measured_td_err, measured_img_ra, measured_img_dec
        #kwargs_model : dictionary of list of models
        #kwargs_constraints : dictionary of solver configs
        #kwargs_likelihood : parameters of likelihood function
        #kwargs_params : dictionary of components as keys and list of min, max, ? init dictionaries as values
        fitting_seq = FittingSequence(kwargs_data_joint, kwargs_model, kwargs_constraints, kwargs_likelihood, kwargs_params, verbose=False)
        # MCMC sample from the post-processed BNN posterior jointly with cosmology
        lens_i_start_time = time.time()
        fitting_kwargs_list_mcmc = [['MCMC', test_cfg.numerics.mcmc]]
        with HiddenPrints():
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
        # Convert D_dt into H0
        sampler_type, samples_mcmc, param_mcmc, _  = chain_list_mcmc[0]
        D_dt_i = param_mcmc.index('D_dt')
        D_dt_samples = samples_mcmc[:, D_dt_i]
        true_D_dt = lcdm.D_dt(H_0=data_i['H0'], Om0=0.3)
        mean_D_dt, std_D_dt = plotting_utils.plot_D_dt_histogram(D_dt_samples, lens_i, true_D_dt, include_fit_gaussian=test_cfg.plotting.include_fit_gaussian, save_dir=out_dir)
        lens_inference_dict = dict(
                                   D_dt_mcmc_samples=D_dt_samples,
                                   inference_time=inference_time
                                   )
        lens_inference_dict_save_path = os.path.join(out_dir, 'inference_dict_{0:04d}.npy'.format(lens_i))
        np.save(lens_inference_dict_save_path, lens_inference_dict)
        mean_D_dt_set[i] = mean_D_dt
        std_D_dt_set[i] = std_D_dt
        inference_time_set[i] = inference_time
        total_progress.update(1)
        # Optionally export the plot of MCMC chain
        if test_cfg.export.mcmc_chain:
            fig, ax = chain_plot.plot_chain_list(chain_list_mcmc)
            fig.savefig(os.path.join(out_dir, 'mcmc_chain_{0:04d}.png'.format(lens_i)), dpi=100)
            plt.close()
        # Optionally export posterior cornerplot of select lens model parameters with D_dt
        if test_cfg.export.mcmc_corner:
            labels_new = [r"$\theta_E$", r"$\gamma$", r"$\phi_{lens}$", r"$q$", r"$\phi_{ext}$", r"$\gamma_{ext}$", r"$D_{\Delta t}$"]
            param = Param(kwargs_model, kwargs_fixed_lens, kwargs_fixed_source=kwargs_fixed_src_light, kwargs_fixed_lens_light=kwargs_fixed_lens_light, kwargs_fixed_ps=fixed_ps, kwargs_fixed_special=fixed_special, kwargs_lens_init=kwargs_result_mcmc['kwargs_lens'], **kwargs_constraints)
            mcmc_new_list = []
            for i in range(len(samples_mcmc)):
                kwargs_out = param.args2kwargs(samples_mcmc[i])
                kwargs_lens_out, kwargs_special_out, _ = kwargs_out['kwargs_lens'], kwargs_out['kwargs_special'], kwargs_out['kwargs_ps']
                theta_E = kwargs_lens_out[0]['theta_E']
                gamma = kwargs_lens_out[0]['gamma']
                e1, e2 = kwargs_lens_out[0]['e1'], kwargs_lens_out[0]['e2']
                phi, q = param_util.ellipticity2phi_q(e1, e2)
                gamma1, gamma2 = kwargs_lens_out[1]['gamma1'], kwargs_lens_out[1]['gamma2']
                phi_ext, gamma_ext = param_util.shear_cartesian2polar(gamma1, gamma2)
                D_dt = kwargs_special_out['D_dt']
                new_chain = [theta_E, gamma, phi, q, phi_ext, gamma_ext, D_dt]
                mcmc_new_list.append(np.array(new_chain))
            fig = corner.corner(mcmc_new_list, labels=labels_new, show_titles=True)
            fig.savefig(os.path.join(out_dir, 'mcmc_corner_{0:04d}.png'.format(lens_i)), dpi=100)
            plt.close()
    total_progress.close()
    inference_stats = dict(
                    name='rung1_seed{:d}'.format(lens_i),
                    mean=mean_D_dt_set,
                    std=std_D_dt_set,
                    inference_time=inference_time_set,
                    )
    h0_stats_save_path = os.path.join(out_dir, 'inference_stats')
    np.save(h0_stats_save_path, inference_stats)


if __name__ == '__main__':
    main()