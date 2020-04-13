import numpy as np
import torch
__all__ = ['rescale_01', 'stack_rgb', 'log_parameterize_Y_cols', 'whiten_Y_cols', 'plus_1_log', 'asinh', 'whiten_pixels', 'calculate_sky_brightness']

def calculate_sky_brightness(flux_density=7.26*1e-19, lam_eff=15269.1):
    """Calculate the sky brightness in mag with our zeropoint

    Parameters
    ----------
    flux_density: float
        the flux density in cgs units, defined per wavelength in angstroms (Default: 7.26*1e-19, estimated for 12500 ang and taken from Givialisco et al 2002)
    lam_eff : float
        the effective filter wavelength in angstroms (Default: for WFC3/IR F160W, 15279.1 ang). Taken from http://svo2.cab.inta-csic.es/svo/theory/fps3/index.php?id=HST/WFC3_IR.F160W

    Note
    ----
    Zeropoint not necessary with absolute flux density values. Only when using cps.

    """
    # All spectral flux density units are per asec^2
    # Zodiacal bg at the North Ecliptic Pole for WFC3, from Giavalisco et al 2002
    flux_density_cgs_wave = flux_density #1.81*1e-18 # erg/cm^2/s^2/ang^1
    # Convert spectral flux density for unit wavelength to that for unit frequency
    flux_density_Jy = flux_density_cgs_wave*(3.34*1e4)*lam_eff**2.0 # Jy
    mag_AB = -2.5*np.log10(flux_density_Jy) + 8.90
    return mag_AB

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