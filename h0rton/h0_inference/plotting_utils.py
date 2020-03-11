import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import corner
from lenstronomy.Plots import chain_plot

__all__ = ['plot_h0_histogram', "plot_D_dt_histogram", "plot_mcmc_corner"]

def gaussian(x, mean, standard_deviation, amplitude):
    return amplitude * np.exp( - ((x - mean) / standard_deviation) ** 2)

def plot_h0_histogram(all_samples, all_weights, lens_i=0, true_h0=None, include_fit_gaussian=True, save_dir='.'):
    """Plot the histogram of H0 samples, overlaid with a Gaussian fit and truth H0

    all_samples : np.array
        H0 samples
    all_weights : np.array
        H0 weights corresponding to `all_samples`, possibly including nan values

    """
    # Normalize weights to unity
    is_nan_mask = np.logical_or(np.isnan(all_weights), ~np.isfinite(all_weights))
    all_weights[~is_nan_mask] = all_weights[~is_nan_mask]/np.sum(all_weights[~is_nan_mask])
    samples = all_samples[~is_nan_mask]
    weights = all_weights[~is_nan_mask]
    bin_heights, bin_borders, _ = plt.hist(samples, weights=weights, bins=40, alpha=0.5, density=True, edgecolor='k', color='tab:blue', range=[40.0, 100.0])
    bin_centers = bin_borders[:-1] + np.diff(bin_borders) / 2
    if include_fit_gaussian:
        # Fit a gaussian
        best_guess_mean = bin_centers[np.argmax(bin_heights)]
        popt, _ = curve_fit(gaussian, bin_centers, bin_heights, p0=[best_guess_mean, 0.3, 3.0], maxfev=10000)
        mean = popt[0]
        std = popt[1]
    else:
        # Compute the weighted mean and std analytically
        mean = np.average(samples, weights=weights)
        std = np.average((samples - mean)**2.0, weights=weights)**0.5
        #print(mean, std)
        popt = [mean, std, 1.0/std/np.sqrt(2*np.pi)]
    #x_interval_for_fit = np.linspace(bin_borders[0], bin_borders[-1], 10000)
    x_interval_for_fit = np.linspace(bin_centers[0], bin_centers[-1], 1000) 
    # Overlay the fit gaussian pdf
    plt.plot(x_interval_for_fit, gaussian(x_interval_for_fit, *popt), color='k', label='fit: mu={:0.1f}, sig={:0.1f}'.format(mean, std))
    #if std < 1.0:
    #    bin_heights, bin_borders, _ = plt.hist(samples, weights=weights, bins=80, alpha=0.5, density=True, edgecolor='k', color='tab:blue', range=[mean - 5, mean + 5])
    #    bin_centers = bin_borders[:-1] + np.diff(bin_borders) / 2
    #    best_guess_mean = bin_centers[np.argmax(bin_heights)]
    #    popt, _ = curve_fit(gaussian, bin_centers, bin_heights, p0=[mean, 0.3, 1.0], maxfev=10000)
    #    mean = popt[0]
    #    std = popt[-1]
    #print(popt)
    if save_dir is not None:
        if true_h0 is not None:
            plt.axvline(x=true_h0, linestyle='--', color='red', label='truth')
        plt.xlabel('H0 (km/Mpc/s)')
        plt.ylabel('density')
        plt.title('H0 posterior for lens {0:04d}'.format(lens_i))
        plt.legend()
        save_path = os.path.join(save_dir, 'h0_histogram_{0:04d}.png'.format(lens_i))
        plt.savefig(save_path)
        plt.close()
    return mean, std

def plot_D_dt_histogram(all_samples, lens_i=0, true_D_dt=None, include_fit_gaussian=True, save_dir='.'):
    """Plot the histogram of D_dt samples, overlaid with a Gaussian fit and truth D_dt

    all_samples : np.array
        D_dt MCMC samples

    """
    bin_heights, bin_borders, _ = plt.hist(all_samples, bins=100, alpha=0.5, density=True, edgecolor='k', color='tab:blue', range=[0.0, 10000.0])
    bin_centers = bin_borders[:-1] + np.diff(bin_borders) / 2
    if include_fit_gaussian:
        # Fit a gaussian
        best_guess_mean = bin_centers[np.argmax(bin_heights)]
        popt, _ = curve_fit(gaussian, bin_centers, bin_heights, p0=[best_guess_mean, 0.3, 3.0], maxfev=10000)
        mean = popt[0]
        std = popt[1]
    else:
        # Compute the weighted mean and std analytically
        mean = np.average(all_samples)
        std = np.average((all_samples - mean)**2.0)**0.5
        popt = [mean, std, 1.0/std/np.sqrt(2*np.pi)]
    #x_interval_for_fit = np.linspace(bin_borders[0], bin_borders[-1], 10000)
    x_interval_for_fit = np.linspace(bin_centers[0], bin_centers[-1], 1000) 
    # Overlay the fit gaussian pdf
    plt.plot(x_interval_for_fit, gaussian(x_interval_for_fit, *popt), color='k', label='fit: mu={:0.1f}, sig={:0.1f}'.format(mean, std))
    if save_dir is not None:
        if true_D_dt is not None:
            plt.axvline(x=true_D_dt, linestyle='--', color='red', label='truth')
        plt.xlabel('D_dt (Mpc)')
        plt.ylabel('density')
        plt.title('D_dt posterior for lens {0:04d}'.format(lens_i))
        plt.legend()
        save_path = os.path.join(save_dir, 'D_dt_histogram_{0:04d}.png'.format(lens_i))
        plt.savefig(save_path)
        plt.close()
    return mean, std

def plot_mcmc_chain(chain_list_mcmc, save_path):
    fig, ax = chain_plot.plot_chain_list(chain_list_mcmc)
    fig.savefig(save_path, dpi=100)
    plt.close()

def plot_mcmc_corner(mcmc_samples, truth, col_labels, save_path):
    fig = corner.corner(mcmc_samples, truths=truth, truth_color='r', labels=col_labels, show_titles=True, quiet=True)
    fig.savefig(save_path, dpi=100)
    plt.close()