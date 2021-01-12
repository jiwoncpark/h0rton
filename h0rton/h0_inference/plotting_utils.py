import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import corner
from lenstronomy.Plots import chain_plot
from h0rton.h0_inference import h0_utils
from scipy.stats import norm#, median_absolute_deviation
from lenstronomy.LensModel.lens_model_extensions import LensModelExtensions
from lenstronomy.LensModel.Solver.lens_equation_solver import LensEquationSolver
#import lenstronomy.Util.util as util
import lenstronomy.Util.simulation_util as sim_util
from lenstronomy.Data.imaging_data import ImageData
from lenstronomy.Plots import plot_util
#import scipy.ndimage as ndimage

__all__ = ["plot_weighted_h0_histogram", 'plot_h0_histogram', "plot_D_dt_histogram", "plot_mcmc_corner", "gaussian", "plot_forward_modeling_comparisons"]

def gaussian(x, mean, standard_deviation, amplitude):
    return amplitude * np.exp( - ((x - mean) / standard_deviation) ** 2)

def lognormal(x, mu, sig):
    return np.exp(-0.5*(np.log(x) - mu)**2.0/sig**2.0)/(x*sig*(2.0*np.pi)**0.5)

def plot_weighted_h0_histogram(all_samples, all_weights, lens_i=0, true_h0=None, include_fit_gaussian=True, save_dir='.'):
    """Plot the histogram of H0 samples, overlaid with a Gaussian fit and truth H0

    all_samples : np.array
        H0 samples
    all_weights : np.array
        H0 weights corresponding to `all_samples`, possibly including nan values

    """
    stats = h0_utils.get_normal_stats_naive(all_samples, all_weights)
    _ = plt.hist(stats['samples'], weights=stats['weights'], bins=290, alpha=0.5, density=True, edgecolor='k', color='tab:blue', range=[10.0, 300.0])
    #print(mean, std)
    x_interval_for_fit = np.linspace(10, 300, 1000)
    # Overlay the fit gaussian pdf
    plt.plot(x_interval_for_fit, norm.pdf(x_interval_for_fit, stats['mean'], stats['std']), color='k', label='fit: mu={:0.1f}, sig={:0.1f}'.format(stats['mean'], stats['std']))
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
    return stats

def plot_weighted_D_dt_histogram(all_samples, all_weights, lens_i=0, true_D_dt=None, save_dir='.'):
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
    bin_heights, bin_borders, _ = plt.hist(samples, weights=weights, bins=200, alpha=0.5, density=True, edgecolor='k', color='tab:blue', range=[0.0, 15000.0])
    bin_centers = bin_borders[:-1] + np.diff(bin_borders) / 2

    # Compute the weighted mean and std analytically
    lognorm_stats = h0_utils.get_lognormal_stats_naive(samples, weights)
    mu = lognorm_stats['mu']
    sigma = lognorm_stats['sigma']
    mode = lognorm_stats['mode']
    std = lognorm_stats['std']
    popt = [mu, sigma]
    #x_interval_for_fit = np.linspace(bin_borders[0], bin_borders[-1], 10000)
    x_interval_for_fit = np.linspace(bin_centers[0], bin_centers[-1], 1000)
    # Overlay the fit gaussian pdf
    plt.plot(x_interval_for_fit, lognormal(x_interval_for_fit, *popt), color='k', label='fit: mode={:0.1f}, std={:0.1f}'.format(mode, std))
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
    return mu, sigma

def plot_h0_histogram(samples, lens_i=0, true_h0=None, include_fit_gaussian=True, save_dir='.'):
    """Plot the histogram of H0 samples, overlaid with a Gaussian fit and truth H0

    all_samples : np.array
        H0 samples
    all_weights : np.array
        H0 weights corresponding to `all_samples`, possibly including nan values

    """
    # Normalize weights to unity
    bin_heights, bin_borders, _ = plt.hist(samples, bins=80, alpha=0.5, density=True, edgecolor='k', color='tab:blue', range=[40.0, 100.0])
    bin_centers = bin_borders[:-1] + np.diff(bin_borders) / 2
    if include_fit_gaussian:
        # Fit a gaussian
        best_guess_mean = bin_centers[np.argmax(bin_heights)]
        popt, _ = curve_fit(gaussian, bin_centers, bin_heights, p0=[best_guess_mean, 0.3, 3.0], maxfev=10000)
        mean = popt[0]
        std = popt[1]
    else:
        # Compute the weighted mean and std analytically
        mean = np.median(samples)
        std = np.median_absolute_deviation(samples, axis=None)
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

def plot_D_dt_histogram(all_samples, lens_i=0, true_D_dt=None, save_dir='.'):
    """Plot the histogram of D_dt samples, overlaid with a Gaussian fit and truth D_dt

    all_samples : np.array
        D_dt MCMC samples

    """
    bin_heights, bin_borders, _ = plt.hist(all_samples, bins=200, alpha=0.5, density=True, edgecolor='k', color='tab:blue', range=[0.0, 15000.0])
    bin_centers = bin_borders[:-1] + np.diff(bin_borders) / 2

    # Compute the mode and std for lognormal
    lognorm_stats = h0_utils.get_lognormal_stats(all_samples)
    mu = lognorm_stats['mu']
    sigma = lognorm_stats['sigma']
    mode = lognorm_stats['mode']
    std = lognorm_stats['std']
    popt = [mu, sigma]

    #x_interval_for_fit = np.linspace(bin_borders[0], bin_borders[-1], 10000)
    x_interval_for_fit = np.linspace(bin_centers[0], bin_centers[-1], 1000)
    # Overlay the fit gaussian pdf
    plt.plot(x_interval_for_fit, lognormal(x_interval_for_fit, *popt), color='k', label='fit: mode={:0.1f}, std={:0.1f}'.format(mode, std))
    if save_dir is not None:
        if true_D_dt is not None:
            plt.axvline(x=true_D_dt, linestyle='--', color='red', label='truth')
        plt.xlabel(r'$D_{{\Delta t}}$ (Mpc)')
        plt.ylabel('density')
        plt.title(r'$D_{{\Delta t}}$ posterior for lens {0:04d}'.format(lens_i))
        plt.legend()
        save_path = os.path.join(save_dir, 'D_dt_histogram_{0:04d}.png'.format(lens_i))
        plt.savefig(save_path)
        plt.close()
    return mu, sigma

def plot_mcmc_chain(chain_list_mcmc, save_path):
    fig, ax = chain_plot.plot_chain_list(chain_list_mcmc)
    fig.savefig(save_path, dpi=100)
    plt.close()

def plot_mcmc_corner(mcmc_samples, truth, col_labels, save_path):
    fig = corner.corner(mcmc_samples,
                        truths=truth,
                        truth_color='r',
                        labels=col_labels,
                        smooth=1.0,
                        no_fill_contours=True,
                        plot_datapoints=False,
                        show_titles=True,
                        quiet=True,
                        plot_contours=True,
                        use_math_text=True,
                        contour_kwargs=dict(linestyles='solid'),
                        levels=[0.68, 0.95],)
    fig.savefig(save_path, dpi=100)
    plt.close()

def plot_forward_modeling_comparisons(model_plot_instance, out_dir):
    """Plot the data vs. model comparisons using the Lenstronomy modelPlot tool

    Parameters
    ----------
    model_plot_instance : lenstronomy.Plots.model_plot.ModelPlot object
    out_dir : directory in which the plots will be saved

    """
    f, axes = model_plot_instance.plot_main()
    f.savefig(os.path.join(out_dir, 'main_plot_lenstronomy.png'))
    plt.close()
    f, axes = model_plot_instance.plot_separate()
    f.savefig(os.path.join(out_dir, 'separate_plot_lenstronomy.png'))
    plt.close()
    f, axes = model_plot_instance.plot_subtract_from_data_all()
    f.savefig(os.path.join(out_dir, 'subtract_plot_lenstronomy.png'))
    plt.close('all')

# TODO define coordinate grid beforehand, e.g. kwargs_data
def lens_model_plot_custom(image, ax, lensModel, kwargs_lens, numPix=500, deltaPix=0.01, sourcePos_x=0, sourcePos_y=0, point_source=False, with_caustics=False):
    """
    plots a lens model (convergence) and the critical curves and caustics

    :param ax:
    :param kwargs_lens:
    :param numPix:
    :param deltaPix:
    :return:
    """
    kwargs_data = sim_util.data_configure_simple(numPix, deltaPix)
    data = ImageData(**kwargs_data)
    _coords = data
    _frame_size = numPix * deltaPix
    x_grid, y_grid = data.pixel_coordinates
    lensModelExt = LensModelExtensions(lensModel)
    #ra_crit_list, dec_crit_list, ra_caustic_list, dec_caustic_list = lensModelExt.critical_curve_caustics(
    #    kwargs_lens, compute_window=_frame_size, grid_scale=deltaPix/2.)
    #x_grid1d = util.image2array(x_grid)
    #y_grid1d = util.image2array(y_grid)
    #kappa_result = lensModel.kappa(x_grid1d, y_grid1d, kwargs_lens)
    #kappa_result = util.array2image(kappa_result)
    #im = ax.matshow(np.log10(kappa_result), origin='lower', extent=[0, _frame_size, 0, _frame_size], cmap='Greys',vmin=-1, vmax=1) #, cmap=self._cmap, vmin=v_min, vmax=v_max)
    im = ax.matshow(image, origin='lower', extent=[0, _frame_size, 0, _frame_size])
    if with_caustics is True:
        ra_crit_list, dec_crit_list = lensModelExt.critical_curve_tiling(kwargs_lens, compute_window=_frame_size, start_scale=deltaPix, max_order=20)
        ra_caustic_list, dec_caustic_list = lensModel.ray_shooting(ra_crit_list, dec_crit_list, kwargs_lens)
        plot_util.plot_line_set(ax, _coords, ra_caustic_list, dec_caustic_list,
                                color='tab:red')
        plot_util.plot_line_set(ax, _coords, ra_crit_list, dec_crit_list,
                                color='yellow')
    if point_source:
        solver = LensEquationSolver(lensModel)
        theta_x, theta_y = solver.image_position_from_source(sourcePos_x,
                                                             sourcePos_y,
                                                             kwargs_lens,
                                                             min_distance=deltaPix,
                                                             search_window=deltaPix*numPix)
        mag_images = lensModel.magnification(theta_x, theta_y, kwargs_lens)
        x_image, y_image = _coords.map_coord2pix(theta_x, theta_y)
        abc_list = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K']
        for i in range(len(x_image)):
            x_ = (x_image[i] + 0.5) * deltaPix
            y_ = (y_image[i] + 0.5) * deltaPix
            ax.plot(x_, y_, 'dk',
                    markersize=4*(1 + np.log(np.abs(mag_images[i]))),
                    markerfacecolor='none')
            ax.text(x_+0.1, y_+0.1, abc_list[i], fontsize=15, color='white')
        x_source, y_source = _coords.map_coord2pix(sourcePos_x, sourcePos_y)
        ax.plot((x_source + 0.5) * deltaPix, (y_source + 0.5) * deltaPix,
                marker='*',
                color='tab:red',
                markersize=7.5)
    #ax.plot(numPix * deltaPix*0.5 + pred['lens_mass_center_x'] + pred['src_light_center_x'], numPix * deltaPix*0.5 + pred['lens_mass_center_y'] + pred['src_light_center_y'], '*k', markersize=5)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.autoscale(False)
    return ax