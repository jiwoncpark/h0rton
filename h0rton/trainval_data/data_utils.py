import numpy as np
__all__ = ['log_parameterize_Y_cols', 'whiten_Y_cols']

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

def whiten_Y_cols(df, col_names, mean, std):
    """Whiten select columns in the given dataframe, i.e. shift and scale then so that they have the desired mean and std

    Parameters
    ----------
    df : pd.DataFrame
    col_names : list
        names of columns to whiten
    mean : array-like
        target mean
    std : array-like
        target std

    Returns
    -------
    pd.DataFrame
        a version of the `df` with the specified columns whitened

    """
    df.loc[:, col_names] = (df.loc[:, col_names].values - mean)/std
    return df