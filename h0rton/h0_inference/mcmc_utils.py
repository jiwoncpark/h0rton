import numpy as np
import pandas as pd
import torch
from lenstronomy.Sampling.parameters import Param
import baobab.sim_utils.metadata_utils as metadata_utils
import h0rton.losses

__all__ = ['get_lens_kwargs', 'get_ps_kwargs', 'get_ps_kwargs_src_plane', 'get_light_kwargs', 'get_special_kwargs', 'postprocess_mcmc_chain', "HybridBNNPenalty", "get_idx_for_params", "remove_parameters_from_pred", "split_component_param", "dict_to_array"]

# Conversion from param to BNN column naming
baobab_to_param = dict(zip(['lens_mass_theta_E', 'lens_mass_gamma', 'lens_mass_e1', 'lens_mass_e2', 'lens_mass_center_x', 'lens_mass_center_y', 'external_shear_gamma1', 'external_shear_gamma2', 'src_light_R_sersic', 'src_light_center_x', 'src_light_center_y', 'D_dt'],
                           ['theta_E_lens0', 'gamma_lens0', 'e1_lens0', 'e2_lens0', 'center_x_lens0', 'center_y_lens0', 'gamma1_lens1', 'gamma2_lens1', 'R_sersic_source_light0', 'ra_source', 'dec_source', 'D_dt']))
param_to_baobab = dict(zip(['theta_E_lens0', 'gamma_lens0', 'e1_lens0', 'e2_lens0', 'center_x_lens0', 'center_y_lens0', 'gamma1_lens1', 'gamma2_lens1', 'R_sersic_source_light0', 'ra_source', 'dec_source', 'D_dt'], 
                           ['lens_mass_theta_E', 'lens_mass_gamma', 'lens_mass_e1', 'lens_mass_e2', 'lens_mass_center_x', 'lens_mass_center_y', 'external_shear_gamma1', 'external_shear_gamma2', 'src_light_R_sersic', 'src_light_center_x', 'src_light_center_y', 'D_dt']))

def get_lens_kwargs(init_dict, null_spread=False):
    """Get the SPEMD, SHEAR kwargs with the provided init values and some conservative sigma, fixed, lower, and upper

    Note
    ----
    The sigma, fixed, lower, and upper kwargs are hardcoded.

    Parameters
    ----------
    init_dict : dict
        the init values for each of the parameters in SPEMD and SHEAR

    """
    eps = 1.e-7 # some small fractional value
    kwargs_init_lens = [{'theta_E': init_dict['lens_mass_theta_E'], 'gamma': init_dict['lens_mass_gamma'], 'center_x': init_dict['lens_mass_center_x'], 'center_y': init_dict['lens_mass_center_y'], 'e1': init_dict['lens_mass_e1'], 'e2': init_dict['lens_mass_e2']}, {'gamma1': init_dict['external_shear_gamma1'], 'gamma2': init_dict['external_shear_gamma2']}]
    if null_spread:
        kwargs_sigma_lens = [{k: eps*v for k, v in kwargs_init_lens[0].items()}, {k: eps*v for k, v in kwargs_init_lens[1].items()}]
        kwargs_fixed_lens = [{}, {'ra_0': init_dict['lens_mass_center_x'], 'dec_0': init_dict['lens_mass_center_y']}] # FIXME: fixing shear at the wrong position won't affect time delays but caution in case you add likelihoods affected by shear
    else:
        kwargs_sigma_lens = [{'theta_E': 0.05, 'e1': 0.05, 'e2': 0.05, 'gamma': 0.05, 'center_x': 0.02, 'center_y': 0.02}, {'gamma1': 0.05, 'gamma2': 0.05}]
        kwargs_fixed_lens = [{}, {'ra_0': 0.0, 'dec_0': 0.0}]
    kwargs_lower_lens = [{'theta_E': 0.01, 'e1': -1, 'e2': -1, 'gamma': 1, 'center_x': -10, 'center_y': -10}, {'gamma1': -1, 'gamma2': -1}]
    kwargs_upper_lens = [{'theta_E': 10, 'e1': 1, 'e2': 1, 'gamma': 4, 'center_x': 10, 'center_y': 10}, {'gamma1': 1, 'gamma2': 1,}]
    return [kwargs_init_lens, kwargs_sigma_lens, kwargs_fixed_lens, kwargs_lower_lens, kwargs_upper_lens]

def get_ps_kwargs(measured_img_ra, measured_img_dec, astrometry_sigma, hard_bound=30.0):
    """Get the point source kwargs for the image positions

    Parameters
    ----------
    measured_img_ra : np.array
        measured ra of the images
    measured_img_dec : np.array
        measured dec of the images
    astrometry_sigma : float
        astrometric uncertainty in arcsec
    hard_bound : float
        hard bound of the image positions around zero in arcsec

    Returns
    -------
    list of dict
        list of init, sigma, fixed, lower, and upper kwargs

    """
    n_img = len(measured_img_dec)
    ones = np.ones(n_img)
    kwargs_ps_init = [{'ra_image': measured_img_ra, 'dec_image': measured_img_dec}]
    kwargs_ps_sigma = [{'ra_image': astrometry_sigma*ones, 'dec_image': astrometry_sigma*ones}]
    fixed_ps = [{}] 
    kwargs_lower_ps = [{'ra_image': -hard_bound*ones, 'dec_image': -hard_bound*ones}]
    kwargs_upper_ps = [{'ra_image': hard_bound*ones, 'dec_image': hard_bound*ones}]
    return [kwargs_ps_init, kwargs_ps_sigma, fixed_ps, kwargs_lower_ps, kwargs_upper_ps]

def get_ps_kwargs_src_plane(init_dict, astrometry_sigma, hard_bound=5.0):
    """Get the point source kwargs for the source plane

    Parameters
    ----------
    measured_img_ra : np.array
        measured ra of the images
    measured_img_dec : np.array
        measured dec of the images
    astrometry_sigma : float
        astrometric uncertainty in arcsec
    hard_bound : float
        hard bound of the image positions around zero in arcsec

    Returns
    -------
    list of dict
        list of init, sigma, fixed, lower, and upper kwargs

    """
    kwargs_ps_init = [{'ra_source': init_dict['src_light_center_x'] + init_dict['lens_mass_center_x'], 'dec_source': init_dict['src_light_center_y'] + init_dict['lens_mass_center_y']}]
    kwargs_ps_sigma = [{'ra_source': astrometry_sigma, 'dec_source': astrometry_sigma}]
    fixed_ps = [{}] 
    kwargs_lower_ps = [{'ra_source': -hard_bound, 'dec_source': -hard_bound}]
    kwargs_upper_ps = [{'ra_source': hard_bound, 'dec_source': hard_bound}]
    return [kwargs_ps_init, kwargs_ps_sigma, fixed_ps, kwargs_lower_ps, kwargs_upper_ps]

def get_light_kwargs(init_R, null_spread=False):
    """Get the usual sigma, lower, upper, and fixed kwargs for a SERSIC_ELLIPSE light with only the R_sersic allowed to vary

    """
    eps = 1.e-7 # some small fractional value
    kwargs_light_init = [{'R_sersic': init_R}]
    if null_spread:
        kwargs_light_sigma = [{k: v*eps for k, v in kwargs_light_init[0].items()}]
    else:
        kwargs_light_sigma = [{'R_sersic': 0.05}]
    kwargs_light_fixed = [{'n_sersic': None, 'e1': None, 'e2': None, 'center_x': None, 'center_y': None}]
    kwargs_light_lower = [{'R_sersic': 0.01}]
    kwargs_light_upper = [{'R_sersic': 10.0}]
    return [kwargs_light_init, kwargs_light_sigma, kwargs_light_fixed, kwargs_light_lower, kwargs_light_upper]

def get_special_kwargs(n_img, astrometry_sigma, delta_pos_hard_bound=5.0, D_dt_init=5000.0, D_dt_sigma=1000.0, D_dt_lower=0.0, D_dt_upper=15000.0):
    """Get the point source kwargs for the image positions

    Parameters
    ----------
    measured_img_ra : np.array
        measured ra of the images
    measured_img_dec : np.array
        measured dec of the images
    astrometry_sigma : float
        astrometric uncertainty in arcsec
    hard_bound : float
        hard bound of the image positions around zero in arcsec

    Returns
    -------
    list of dict
        list of init, sigma, fixed, lower, and upper kwargs

    """
    zeros = np.zeros(n_img)
    ones = np.ones(n_img)
    kwargs_special_init = {'delta_x_image': zeros, 'delta_y_image': zeros, 'D_dt': D_dt_init}
    kwargs_special_sigma = {'delta_x_image': ones*astrometry_sigma, 'delta_y_image': ones*astrometry_sigma, 'D_dt': D_dt_sigma}
    fixed_special = {}
    kwargs_lower_special = {'delta_x_image': -ones*delta_pos_hard_bound, 'delta_y_image': -ones*delta_pos_hard_bound, 'D_dt': D_dt_lower}
    kwargs_upper_special = {'delta_x_image': ones*delta_pos_hard_bound, 'delta_y_image': ones*delta_pos_hard_bound, 'D_dt': D_dt_upper}
    return [kwargs_special_init, kwargs_special_sigma, fixed_special, kwargs_lower_special, kwargs_upper_special]

def postprocess_mcmc_chain(kwargs_result, samples, kwargs_model, fixed_lens_kwargs, fixed_ps_kwargs, fixed_src_light_kwargs, fixed_special_kwargs, kwargs_constraints, kwargs_fixed_lens_light=None, verbose=False, forward_modeling=False):
    """Postprocess the MCMC chain for making the chains consistent with the optimized lens model and converting parameters

    Returns
    -------
    pandas.DataFrame
        processed MCMC chain, where each row is a sample

    """
    param = Param(kwargs_model, fixed_lens_kwargs, kwargs_fixed_ps=fixed_ps_kwargs, kwargs_fixed_source=fixed_src_light_kwargs, kwargs_fixed_special=fixed_special_kwargs, kwargs_fixed_lens_light=kwargs_fixed_lens_light, kwargs_lens_init=kwargs_result['kwargs_lens'], **kwargs_constraints)
    if verbose:
        param.print_setting()
    n_samples = len(samples)
    processed = []
    for i in range(n_samples):
        kwargs = {}
        kwargs_out = param.args2kwargs(samples[i])
        kwargs_lens_out, kwargs_special_out, kwargs_ps_out, kwargs_source_out, kwargs_lens_light_out = kwargs_out['kwargs_lens'], kwargs_out['kwargs_special'], kwargs_out['kwargs_ps'], kwargs_out['kwargs_source'], kwargs_out['kwargs_lens_light']
        for k, v in kwargs_lens_out[0].items():
            kwargs['lens_mass_{:s}'.format(k)] = v
        for k, v in kwargs_lens_out[1].items():
            kwargs['external_shear_{:s}'.format(k)] = v
        if forward_modeling:
            for k, v in kwargs_source_out[0].items():
                kwargs['src_light_{:s}'.format(k)] = v
            for k, v in kwargs_lens_light_out[0].items():
                kwargs['lens_light_{:s}'.format(k)] = v
        else:
            kwargs['src_light_R_sersic'] = kwargs_source_out[0]['R_sersic']
        if 'ra_source' in kwargs_ps_out[0]:
            kwargs['src_light_center_x'] = kwargs_ps_out[0]['ra_source'] - kwargs_lens_out[0]['center_x']
            kwargs['src_light_center_y'] = kwargs_ps_out[0]['dec_source'] - kwargs_lens_out[0]['center_y']
        for k, v in kwargs_special_out.items():
            kwargs[k] = v
        processed.append(kwargs)
    processed_df = pd.DataFrame(processed)
    processed_df = metadata_utils.add_qphi_columns(processed_df)
    processed_df = metadata_utils.add_gamma_psi_ext_columns(processed_df)
    return processed_df

class HybridBNNPenalty:
    """Wrapper for subclasses of BaseGaussianNLL that allows MCMC methods to appropriately penalize parameter samples

    """
    def __init__(self, Y_cols, likelihood_class, mcmc_train_Y_mean, mcmc_train_Y_std, exclude_vel_disp, device):
        """
        Parameters
        ----------
        Y_dim : int
            number of parameters subject to penalty function
        likelihood_class : str
            name of subclass of BaseGaussianNLL to wrap around
        exclude_vel_disp : bool
            whether to add the NLL of velocity dispersion (not used)
        device : str

        """
        self.Y_cols = Y_cols
        self.Y_dim = len(self.Y_cols)
        self.mcmc_train_Y_mean = mcmc_train_Y_mean
        self.mcmc_train_Y_std = mcmc_train_Y_std
        self.constant_term = np.log(2*np.pi)*self.Y_dim*0.5
        #self.device = device
        self.exclude_vel_disp = exclude_vel_disp
        self.nll = getattr(h0rton.losses, '{:s}CPU'.format(likelihood_class))(Y_dim=self.Y_dim)

    def set_bnn_post_params(self, bnn_post_params):
        """Set BNN posterior parameters, which define the penaty function

        """
        self.bnn_post_params = bnn_post_params#.reshape(1, -1)
        self.n_dropout = bnn_post_params.shape[0]

    def evaluate(self, kwargs_lens, kwargs_source, kwargs_lens_light=None, kwargs_ps=None, kwargs_special=None, kwargs_extinction=None):
        #kwargs_lens[1]['ra_0'] = kwargs_lens[0]['center_x']
        #kwargs_lens[1]['dec_0'] = kwargs_lens[0]['center_y']
        to_eval = dict_to_array(self.Y_cols, kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_ps) # shape [1, self.Y_dim]
        # Whiten the mcmc array
        to_eval = (to_eval - self.mcmc_train_Y_mean)/self.mcmc_train_Y_std
        #print(self.bnn_post_params[:, :self.Y_dim], to_eval)
        to_eval = np.repeat(to_eval, repeats=self.n_dropout, axis=0)
        ll = -self.nll(self.bnn_post_params, to_eval)
        return ll #+ self.constant_term

def get_idx_for_params(out_dim, Y_cols, params_to_remove, likelihood_class, debug=False):
    """Get columns corresponding to certain parameters from the BNN output

    Parameters
    ----------
    out_dim : int
    Y_cols : list of str
    
    Returns
    -------
    remove_idx : list
        indices of the columns removed in orig_pred
    remove_param_idx : list
        indices of the parameters in Y_cols

    """
    Y_dim = len(Y_cols)
    col_to_idx = dict(zip(Y_cols, range(Y_dim)))
    param_idx = np.array([col_to_idx[i] for i in params_to_remove]) # indices corresponding to primary mean
    if likelihood_class in ['FullRankGaussianNLL', 'DoubleGaussianNLL']:
        tril_idx = np.tril_indices(Y_dim)
        tril_idx_dim0 = tril_idx[0]
        tril_idx_dim1 = tril_idx[1]
        tril_len = len(tril_idx_dim0)
        tril_mask = np.logical_or(np.isin(tril_idx_dim0, param_idx), np.isin(tril_idx_dim1, param_idx)).nonzero()[0]
        idx_within_tril1 = list(Y_dim + tril_mask)
        idx_within_tril2 = list(2*Y_dim + tril_len + tril_mask)
        idx = list(param_idx) + idx_within_tril1 + list(Y_dim + tril_len + param_idx) + idx_within_tril2
        if debug:
            to_test = dict(
                           tril_mask=tril_mask,
                           idx_within_tril1=idx_within_tril1,
                           idx_within_tril2=idx_within_tril2,
                           param_idx=param_idx,
                           idx=idx,
                           )
            return to_test
    else: # tested for 'DoubleLowRankGaussianNLL':
        tiling_by_Y_dim = np.arange(out_dim//Y_dim)*Y_dim
        idx = [tile + i for tile in tiling_by_Y_dim for i in param_idx]
    param_idx = np.array(param_idx, dtype=int)
    idx = np.array(idx, dtype=int)
    return param_idx, idx 

def remove_parameters_from_pred(orig_pred, remove_idx, return_as_tensor=True, device='cpu'):
    """Remove columns corresponding to certain parameters from the BNN output

    Parameters
    ----------
    orig_pred : np.array of shape `[n_lenses, out_dim]`
        the BNN output
    orig_Y_cols : list of str
        the original list of Y columns
    params_to_remove : list of str
        list of colums to remove. They must belong to orig_Y_cols

    Returns
    -------
    new_pred : np.array of shape `[n_lenses, out_dim - len(params_to_remove)]`
    
    """
    new_pred = np.delete(orig_pred, remove_idx, axis=1)
    if return_as_tensor:
        new_pred = torch.as_tensor(new_pred, device=device)
    return new_pred

def split_component_param(string, sep='_', pos=2):
    """Split the component, e.g. lens_mass, from the parameter, e.g. center_x, from the paramter names under the Baobab convention

    Parameters
    ----------
    string : str
        the Baobab parameter name (column name)
    sep : str
        separation character between the component and the parameter. Default: '_' (Baobab convention)
    pos : int
        position of the component when split by the separation character. Default: 2 (Baobab convention)

    Returns
    -------
    tuple of str
        component, parameter

    """
    substring_list = string.split(sep)
    return sep.join(substring_list[:pos]), sep.join(substring_list[pos:])

def dict_to_array(Y_cols, kwargs_lens, kwargs_source, kwargs_lens_light=None, kwargs_ps=None,):
    """Reformat kwargs into np array. Used to feed the current iteration of MCMC kwargs into the BNN posterior evaluation.

    """
    return_array = np.ones((len(Y_cols)))
    for i, col_name in enumerate(Y_cols):
        component, param = split_component_param(col_name, '_', 2)
        if component == 'lens_mass':
            return_array[i] = kwargs_lens[0][param]
        elif component == 'external_shear':
            return_array[i] = kwargs_lens[1][param]
        elif component == 'src_light':
            if param == 'center_x':
                return_array[i] = kwargs_ps[0]['ra_source'] - kwargs_lens[0]['center_x']
            elif param == 'center_y':
                return_array[i] = kwargs_ps[0]['dec_source'] - kwargs_lens[0]['center_y']
            else:
                return_array[i] = kwargs_source[0][param]
        #elif component == 'lens_light':
        #    return_array[i] = kwargs_lens_light[0][param]
        else:
            # Ignore kwargs_ps since image position likelihood is separate.
            raise ValueError("Component doesn't exist.")
    return return_array.reshape(1, -1)

def reorder_to_param_class(bnn_Y_cols, param_class_Y_cols, bnn_array, D_dt_array):
    """Reorder an array with the given axis ordered according to the BNN's bnn_Y_cols to the Lenstronomy Param's param_class_Y_cols convention

    Parameters
    ----------
    bnn_Y_cols : list of str
    param_class_Y_cols : list of str
    bnn_array : np.array 
    D_dt_array : np.array 
    axis : int
    
    """
    baobab_cols = bnn_Y_cols + ['D_dt']
    baobab_col_to_idx = dict(zip(baobab_cols, range(len(baobab_cols))))
    #param_col_to_idx = dict(zip(param_class_Y_cols, range(len(param_class_Y_cols))))
    # Store the absolute source position instead
    bnn_array[:, :, baobab_col_to_idx['src_light_center_x']] += bnn_array[:, :, baobab_col_to_idx['lens_mass_center_x']]
    bnn_array[:, :, baobab_col_to_idx['src_light_center_y']] += bnn_array[:, :, baobab_col_to_idx['lens_mass_center_y']]
    baobab_array = np.concatenate([bnn_array, D_dt_array], axis=-1) # [n_lenses, n_samples, bnn_Y_dim + 1]
    ordering = [baobab_col_to_idx[param_to_baobab[param_col]] for param_col in param_class_Y_cols]
    mcmc_array = baobab_array[:, :, ordering] # [n_lenses, n_samples, len(param_class_Y_cols)]
    return mcmc_array