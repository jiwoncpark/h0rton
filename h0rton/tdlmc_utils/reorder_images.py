import numpy as np
__all__ = ['reorder_to_tdlmc']

def reorder_to_tdlmc(abcd_ordering_i, ra_img, dec_img, time_delays):
    """Reorder the list of ra, dec, and time delays to conform to the
    order in the TDLMC challenge

    Parameters
    ----------
    abcd_ordering_i : array-like
        ABCD in an increasing dec order if the keys ABCD mapped to values 0123, respectively, e.g. [3, 1, 0, 2] if D (value 3) is lowest, B (value 1) is second lowest
    ra_img : array-like
        list of ra from lenstronomy
    dec_img : array-like
        list of dec from lenstronomy, in the order specified by `ra_img`
    time_delays : array-like
        list of time delays from lenstronomy, in the order specified by `ra_img`

    Returns
    -------
    tuple
        tuple of (reordered ra, reordered_dec, reordered time delays)

    """
    ra_img = np.array(ra_img)
    dec_img = np.array(dec_img)
    time_delays = np.array(time_delays)
    # Order ra_pos, dec_pos, time_delays in increasing dec order 
    increasing_dec_i = np.argsort(dec_img)
    ra_img = ra_img[increasing_dec_i]
    dec_img = dec_img[increasing_dec_i]
    time_delays = time_delays[increasing_dec_i]
    # Reorder to get it in ABCD
    ra_img = ra_img[abcd_ordering_i]
    dec_img = dec_img[abcd_ordering_i]
    time_delays = time_delays[abcd_ordering_i]

    return (ra_img, dec_img, time_delays)


