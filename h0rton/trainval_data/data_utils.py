import numpy as np
import torch
__all__ = ['rescale_01', 'whiten_Y_cols', 'plus_1_log', 'asinh', 'whiten_pixels', 'log_parameterize_Y_cols']

def whiten_pixels(pixels):
    return (pixels - torch.mean(pixels))/torch.std(pixels)

def asinh(x):
    return torch.log(x+(x**2+1)**0.5)

def plus_1_log(linear):
    """Add 1 and take the log10 of an image

    Parameters
    ----------
    linear : torch.Tensor of shape `[X_dim, X_dim]`

    Returns
    -------
    torch.Tensor
        the image of the same input shape, with values now logged

    """
    return torch.log1p(linear)

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

def whiten_Y_cols(df, mean, std, col_names):
    """Whiten (in place) select columns in the given dataframe, i.e. shift and scale then so that they have the desired mean and std

    Parameters
    ----------
    df : pd.DataFrame
    mean : array-like
        target mean
    std : array-like
        target std
    col_names : list
        names of columns to whiten

    """
    df.loc[:, col_names] = (df.loc[:, col_names].values - mean)/std
    #return df

def log_parameterize_Y_cols(df, col_names):
    """Whiten (in place) select columns in the given dataframe, i.e. shift and scale then so that they have the desired mean and std

    Parameters
    ----------
    df : pd.DataFrame
    mean : array-like
        target mean
    std : array-like
        target std
    col_names : list
        names of columns to whiten

    """
    df.loc[:, col_names] = np.log(df.loc[:, col_names].values)