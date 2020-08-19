import numpy as np

__all__ = ['get_goodness', 'get_precision', 'get_accuracy', 'format_submission']

def format_submission(summary):
    """Format the summary into submission form for getting the TDLMC metrics cornerplot

    """
    pass

def get_goodness(h0_means, h0_errors, true_h0):
    """Get the goodness of fit (chi square)

    Parameters
    ----------
    h0_means : np.array
        central estimate of H0 for each lens
    h0_errors : np.array
        errors corresponding to the `h0_means`
    true_h0 : np.array or float
        the true H0

    Returns
    -------
    float
        the goodness of fit metric

    """
    chi_sq = np.mean(((h0_means - true_h0)/h0_errors)**2.0)
    return chi_sq

def get_precision(h0_errors, true_h0):
    """Get the precision, i.e. how well-constrained were the estimates on average?

    Parameters
    ----------
    h0_errors : np.array
        errors corresponding to the `h0_means`
    true_h0 : np.array or float
        the true H0

    Returns
    -------
    float
        the precision metric

    """
    precision = np.mean(h0_errors/true_h0)
    return precision

def get_accuracy(h0_means, true_h0):
    """Get the accuracy, i.e. how close were the central estimates to the truth?

    Parameters
    ----------
    h0_means : np.array
        central estimate of H0 for each lens
    true_h0 : np.array or float
        the true H0

    Returns
    -------
    float
        the accuracy metric

    """
    precision = np.mean((h0_means - true_h0)/true_h0)
    return precision