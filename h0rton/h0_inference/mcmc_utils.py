import numpy as np
import pandas as pd
import torch
from lenstronomy.Sampling.parameters import Param
import baobab.sim_utils.metadata_utils as metadata_utils
import h0rton.losses

__all__ = ['get_lens_kwargs', 'get_ps_kwargs', 'get_special_kwargs', "HybridBNNPenalty", "get_idx_for_params", "remove_parameters_from_pred", "split_component_param", "dict_to_array"]

def get_lens_kwargs(init_dict):
    """Get the SPEMD, SHEAR kwargs with the provided init values and some conservative sigma, fixed, lower, and upper

    Note
    ----
    The sigma, fixed, lower, and upper kwargs are hardcoded.

    Parameters
    ----------
    init_dict : dict
        the init values for each of the parameters in SPEMD and SHEAR

    """
    kwargs_init_lens = [{'theta_E': init_dict['lens_mass_theta_E'], 'gamma': init_dict['lens_mass_gamma'], 'center_x': init_dict['lens_mass_center_x'], 'center_y': init_dict['lens_mass_center_y'], 'e1': init_dict['lens_mass_e1'], 'e2': init_dict['lens_mass_e2']}, {'gamma1': init_dict['external_shear_gamma2'], 'gamma2': init_dict['external_shear_gamma2']}]
    kwargs_sigma_lens = [{'theta_E': 0.1, 'e1': 0.1, 'e2': 0.1, 'gamma': 0.1, 'center_x': 0.1, 'center_y': 0.1}, {'gamma1': 0.1, 'gamma2': 0.1}]
    kwargs_fixed_lens = [{}, {'ra_0': 0.0, 'dec_0': 0.0}] # FIXME: fixing shear at the wrong position won't affect time delays but caution in case you add likelihoods affected by shear
    kwargs_lower_lens = [{'theta_E': 0.01, 'e1': -0.5, 'e2': -0.5, 'gamma': 1.5, 'center_x': -10, 'center_y': -10}, {'gamma1': -0.3, 'gamma2': -0.3}]
    kwargs_upper_lens = [{'theta_E': 10, 'e1': 0.5, 'e2': 0.5, 'gamma': 2.5, 'center_x': 10, 'center_y': 10}, {'gamma1': 0.3, 'gamma2': 0.3,}]
    return [kwargs_init_lens, kwargs_sigma_lens, kwargs_fixed_lens, kwargs_lower_lens, kwargs_upper_lens]

def get_ps_kwargs(measured_img_ra, measured_img_dec, astrometry_sigma, hard_bound=10.0):
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

def get_special_kwargs(n_img, astrometry_sigma, delta_pos_hard_bound=1.0, D_dt_init=5000.0, D_dt_sigma=1000.0, D_dt_lower=0.0, D_dt_upper=10000.0):
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

def get_light_kwargs():
    """Get the usual sigma, lower, upper, and fixed kwargs for a SERSIC_ELLIPSE light with only the R_sersic allowed to vary

    """
    kwargs_sigma_light = [{'R_sersic': 0.1}]
    kwargs_fixed_light = [{'n_sersic': None, 'e1': None, 'e2': None, 'center_x': None, 'center_y': None}]
    kwargs_lower_light = [{'R_sersic': 0.01}]
    kwargs_upper_light = [{'R_sersic': 10.0}]
    return kwargs_sigma_light, kwargs_fixed_light, kwargs_lower_light, kwargs_upper_light

def postprocess_mcmc_chain(kwargs_result, samples, kwargs_model, fixed_lens_kwargs, fixed_ps_kwargs, fixed_special_kwargs, kwargs_constraints):
    """Postprocess the MCMC chain for making the chains consistent with the optimized lens model and converting parameters

    Returns
    -------
    pandas.DataFrame
        processed MCMC chain, where each row is a sample

    """
    param = Param(kwargs_model, fixed_lens_kwargs, kwargs_fixed_ps=fixed_ps_kwargs, kwargs_fixed_special=fixed_special_kwargs, kwargs_lens_init=kwargs_result['kwargs_lens'], **kwargs_constraints)
    n_samples = len(samples)
    processed = []
    for i in range(n_samples):
        kwargs = {}
        kwargs_out = param.args2kwargs(samples[i])
        kwargs_lens_out, kwargs_special_out, _ = kwargs_out['kwargs_lens'], kwargs_out['kwargs_special'], kwargs_out['kwargs_ps']
        for k, v in kwargs_lens_out[0].items():
            kwargs['lens_mass_{:s}'.format(k)] = v
        for k, v in kwargs_lens_out[1].items():
            kwargs['external_shear_{:s}'.format(k)] = v
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
            whether to add the NLL of velocity dispersion
        device : str

        """
        self.Y_cols = Y_cols
        self.Y_dim = len(self.Y_cols)
        self.mcmc_train_Y_mean = mcmc_train_Y_mean
        self.mcmc_train_Y_std = mcmc_train_Y_std
        self.device = device
        self.exclude_vel_disp = exclude_vel_disp
        self.nll = getattr(h0rton.losses, likelihood_class)(Y_dim=self.Y_dim, device=self.device)

    def set_bnn_post_params(self, bnn_post_params):
        """Set BNN posterior parameters, which define the penaty function

        """
        self.bnn_post_params = bnn_post_params.reshape(1, -1)

    def evaluate(self, kwargs_lens, kwargs_source, kwargs_lens_light=None, kwargs_ps=None, kwargs_special=None, kwargs_extinction=None):
        #kwargs_lens[1]['ra_0'] = kwargs_lens[0]['center_x']
        #kwargs_lens[1]['dec_0'] = kwargs_lens[0]['center_y']
        to_eval = dict_to_array(self.Y_cols, kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_ps) # shape [1, self.Y_dim]
        # Whiten the mcmc array
        to_eval = to_eval*self.mcmc_train_Y_std + self.mcmc_train_Y_mean
        to_eval = torch.as_tensor(to_eval, dtype=torch.get_default_dtype(), device=self.device)
        print(self.bnn_post_params[:, :self.Y_cols])
        print(to_eval)
        nll = self.nll(self.bnn_post_params, to_eval).cpu().item()
        if not self.exclude_vel_disp:
            vel_disp_nll = 0.0
            nll += vel_disp_nll
            raise NotImplementedError
        return nll

def get_idx_for_params(out_dim, Y_cols, params_to_remove):
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
    tiling_by_Y_dim = np.arange(out_dim//Y_dim)*Y_dim
    col_to_idx = dict(zip(Y_cols, range(Y_dim)))
    remove_param_idx = [col_to_idx[i] for i in params_to_remove] # indices corresponding to primary mean
    remove_idx = [tile + i for tile in tiling_by_Y_dim for i in remove_param_idx]
    return remove_param_idx, remove_idx, 

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
    """Reformat kwargs into np array

    """
    return_array = np.ones((len(Y_cols)))
    for i, col_name in enumerate(Y_cols):
        component, param = split_component_param(col_name, '_', 2)
        if component == 'lens_mass':
            return_array[i] = kwargs_lens[0][param]
        elif component == 'external_shear':
            return_array[i] = kwargs_lens[1][param]
        #elif component == 'src_light':
        #    return_array[i] = kwargs_source[0][param]
        #elif component == 'lens_light':
        #    return_array[i] = kwargs_lens_light[0][param]
        else:
            # Ignore kwargs_ps since image position likelihood is separate.
            raise ValueError("Component doesn't exist.")
    return return_array.reshape(1, -1)