import numpy as np
import torch
__all__ = ['rescale_01', 'stack_rgb', 'log_parameterize_Y_cols', 'whiten_Y_cols']

def rescale_01(unscaled):
    """Rescale an image of unknown range to values between 0 and 1

    Parameters
    ----------
    unscaled : torch.Tensor of shape `[X_dim, X_dim]`

    Returns
    -------
    torch.Tensor
        the image of the same input shape, with values now scaled between 0 and 1

    """
    return (unscaled - unscaled.min())/(unscaled.max() - unscaled.min())

def stack_rgb(single_channel):
    """Stack a 1-channel images into a 3-channel one

    Parameters
    ----------
    single_channel : torch.Tensor of shape `[X_dim, X_dim]`

    Returns
    -------
    torch.Tensor
        the image duplicated 3 times along the channel axis, of shape `[3, X_dim, X_dim]`

    """
    return torch.stack([single_channel]*3, dim=0)

def log_parameterize_Y_cols(df, col_names):
    """Log parameterize select columns in the given dataframe

    Parameters
    ----------
    df : pd.DataFrame
    col_names : list
        names of columns to log parameterize

    Returns
    -------
    pd.DataFrame
        a version of the `df` with the specified columns log-parameterized

    """
    df.loc[:, col_names] = np.log(df.loc[:, col_names].values)
    return df

def whiten_Y_cols(df, mean, std, col_names):
    """Whiten select columns in the given dataframe, i.e. shift and scale then so that they have the desired mean and std

    Parameters
    ----------
    df : pd.DataFrame
    mean : array-like
        target mean
    std : array-like
        target std
    col_names : list
        names of columns to whiten

    Returns
    -------
    pd.DataFrame
        a version of the `df` with the specified columns whitened

    """
    df.loc[:, col_names] = (df.loc[:, col_names].values - mean)/std
    return df