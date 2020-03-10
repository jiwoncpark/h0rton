import numpy as np
import torch
import h0rton.losses

def reorder_to_tdlmc(img_array, increasing_dec_i, abcd_ordering_i):
    """Apply the permutation scheme for reordering the list of ra, dec, and time delays to conform to the order in the TDLMC challenge

    Parameters
    ----------
    img_array : array-like
        array of properties corresponding to the AGN images

    Returns
    -------
    array-like
        `img_array` reordered to the TDLMC order
    """
    #print(img_array.shape, self.increasing_dec_i.shape, self.abcd_ordering_i.shape)
    img_array = np.array(img_array)[increasing_dec_i][abcd_ordering_i]
    return img_array

def pred_to_natural_gaussian(pred_mu, pred_cov_mat, shift, scale):
    """Convert the BNN-predicted multivariate Gaussian parameters into the natural space counterparts by reverse transformation

    Parameters
    ----------
    pred_mu : np.array of shape `[Y_dim,]`
    pred_cov_mat : np.array of shape `[Y_dim, Y_dim]`
    scale : np.array of shape `[Y_dim,]`
        vector by which the features were scaled, e.g. the training-set feature standard deviations
    shift : np.array of shape `[Y_dim,]`
        vector by which the features were shifted, e.g. the training-set feature means

    Returns
    -------
    mu : np.array of shape `[Y_dim,]`
        mu in natural space
    cov_mat : np.array of shape `[Y_dim, Y_dim]`
        covariance matrix in natural space

    """
    mu = pred_mu*scale + shift
    A = np.diagflat(scale)
    cov_mat = np.matmul(np.matmul(A, pred_cov_mat), A.T) # There is a better way to do this...
    return mu, cov_mat

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
        kwargs_lens[1]['ra_0'] = kwargs_lens[0]['center_x']
        kwargs_lens[1]['dec_0'] = kwargs_lens[0]['center_y']
        to_eval = dict_to_array(self.Y_cols, kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_ps) # shape [1, self.Y_dim]
        # Whiten the mcmc array
        to_eval = to_eval*self.mcmc_train_Y_std + self.mcmc_train_Y_mean
        to_eval = torch.as_tensor(to_eval, dtype=torch.get_default_dtype(), device=self.device)
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
        elif component == 'src_light':
            return_array[i] = kwargs_source[0][param]
        elif component == 'lens_light':
            return_array[i] = kwargs_lens_light[0][param]
        else:
            # Ignore kwargs_ps since image position likelihood is separate.
            raise ValueError("Component doesn't exist.")
    return return_array.reshape(1, -1)

def get_misc_kwargs_lens():
    """Get the usual sigma, lower, upper, and fixed kwargs for a SPEMD, SHEAR mass

    """
    kwargs_sigma_lens = [{'theta_E': .1, 'e1': 0.1, 'e2': 0.1, 'gamma': 0.1, 'center_x': 0.1, 'center_y': 0.1}, {'gamma1': 0.1, 'gamma2': 0.1}]
    # hard bound lower limit of parameters
    kwargs_lower_lens = [{'theta_E': 0.01, 'e1': -0.5, 'e2': -0.5, 'gamma': 1.5, 'center_x': -10, 'center_y': -10}, {'gamma1': -0.3, 'gamma2': -0.3}]
    kwargs_upper_lens = [{'theta_E': 10, 'e1': 0.5, 'e2': 0.5, 'gamma': 2.5, 'center_x': 10, 'center_y': 10}, {'gamma1': 0.3, 'gamma2': 0.3,}]
    kwargs_fixed_lens = [{}, {'ra_0': 0.0, 'dec_0': 0.0}] # FIXME: fixing shear at the wrong position won't affect time delays but caution in case you add likelihoods affected by shear
    return kwargs_sigma_lens, kwargs_lower_lens, kwargs_upper_lens, kwargs_fixed_lens

def get_misc_kwargs_light():
    """Get the usual sigma, lower, upper, and fixed kwargs for a SERSIC_ELLIPSE light with only the R_sersic allowed to vary

    """
    kwargs_sigma_light = [{'R_sersic': 0.1}]
    kwargs_fixed_light = [{'n_sersic': None, 'e1': None, 'e2': None, 'center_x': None, 'center_y': None}]
    kwargs_lower_light = [{'R_sersic': 0.01}]
    kwargs_upper_light = [{'R_sersic': 10.0}]
    return kwargs_sigma_light, kwargs_fixed_light, kwargs_lower_light, kwargs_upper_light
