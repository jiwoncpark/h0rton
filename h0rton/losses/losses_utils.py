import numpy as np

__all__ = ['sigmoid']

def sigmoid(x):
    """Evaluate the sigmoid function 

    Note
    ----
    Numerically stable version pending.

    x : float or array-like
        value to evaluate on

    """
    return 1.0/(1.0 + np.exp(-x))