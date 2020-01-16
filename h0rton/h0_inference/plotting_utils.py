import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

__all__ = ['plot_h0_histogram']

def gaussian(x, mean, amplitude, standard_deviation):
    return amplitude * np.exp( - ((x - mean) / standard_deviation) ** 2)

def plot_h0_histogram(samples, weights, lens_i, true_h0, save_dir='.'):
    """Plot the histogram of H0 samples, overlaid with a Gaussian fit and truth H0

    """
    bin_heights, bin_borders, _ = plt.hist(samples, weights=weights, bins=40, alpha=0.5, density=True, edgecolor='k', color='tab:blue', range=[50.0, 100.0])
    plt.axvline(x=true_h0, linestyle='--', color='red', label='truth')
    bin_centers = bin_borders[:-1] + np.diff(bin_borders) / 2
    best_guess_mean = bin_centers[np.argmax(bin_heights)]
    #print(best_guess_mean)
    popt, _ = curve_fit(gaussian, bin_centers, bin_heights, p0=[best_guess_mean, 0.3, 3.0], maxfev=10000)
    x_interval_for_fit = np.linspace(bin_borders[0], bin_borders[-1], 10000)
    mean = popt[0]
    std = popt[-1]
    #print(popt)
    plt.plot(x_interval_for_fit, gaussian(x_interval_for_fit, *popt), color='k', label='fit: mu={:0.1f}, sig={:0.1f}'.format(mean, std))
    plt.xlabel('H0 (km/Mpc/s)')
    plt.ylabel('density')
    plt.title('h0_histogram_{0:04d}'.format(lens_i))
    plt.legend()
    save_path = os.path.join(save_dir, 'h0_histogram_{0:04d}.png'.format(lens_i))
    plt.savefig(save_path)
    plt.close()
    return mean, std