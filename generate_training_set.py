import os
import sys
import time
from tqdm import tqdm
import astropy.io.fits as pyfits
import argparse
import gc
from collections import OrderedDict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# Lenstronomy modules
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LensModel.Solver.lens_equation_solver import LensEquationSolver
from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.PointSource.point_source import PointSource
from lenstronomy.ImSim.image_model import ImageModel
import lenstronomy.Util.param_util as param_util
import lenstronomy.Util.simulation_util as sim_util
import lenstronomy.Util.image_util as image_util
from lenstronomy.Util import kernel_util
from lenstronomy.Data.imaging_data import ImageData
from lenstronomy.Data.psf import PSF

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', dest='is_train', action='store_true')
    parser.add_argument('--test', dest='is_train', action='store_false')
    parser.add_argument('--n_data', type=int, default=100, help="number of examples to generate")
    parser.add_argument('--seed', type=int, default=123, help="random seed for BNN prior")
    args = parser.parse_args()
    return args

# Define cosmology
z_lens = 0.5 # TODO: sample
z_source = 1.5 # TODO: sample
true_H0 = 70.0 # TODO: sample from U(50.0, 90.0)
from astropy.cosmology import FlatLambdaCDM # TODO: replace with lenstronomy cosmo
cosmo = FlatLambdaCDM(H0=true_H0, Om0=0.27, Ob0=0.0)

# Read in PSF kernel maps
data_dir = 'data' # TODO: draw tree diagram in README
code1_psf_seed_list = [101, 102, 103, 104, 105, 107, 108, 109] 
code2_psf_seed_list = [110, 111, 113, 114, 115, 116, 117, 118]
n_psf = len(code1_psf_seed_list) + len(code2_psf_seed_list)
psf_dict = OrderedDict()
for psf_i, psf_seed in enumerate(code1_psf_seed_list):
    psf_path = os.path.join(data_dir, 'rung1/code1/f160w-seed{:d}/drizzled_image/psf.fits'.format(psf_seed))
    psf_dict[psf_i] = pyfits.getdata(psf_path)
for psf_i, psf_seed in enumerate(code2_psf_seed_list):
    psf_path = os.path.join(data_dir, 'rung1/code2/f160w-seed{:d}/drizzled_image/psf.fits'.format(psf_seed))
    psf_dict[psf_i + len(code1_psf_seed_list)] = pyfits.getdata(psf_path)

# Image specifics
sigma_bkg = 0.05  #  background noise per pixel (Gaussian)
exp_time = 100.0  #  exposure time (arbitrary units, flux per pixel is in units #photons/exp_time unit)
numPix = 100  # cutout pixel size
deltaPix = 0.05  #  pixel size in arcsec (area per pixel = deltaPix**2)
fwhm = 0.1  # full width half max of PSF (only valid when psf_type='gaussian')
psf_type = 'PIXEL'  # 'gaussian', 'pixel', 'NONE'
kernel_size = 91

# initial input simulation
if __name__ == "__main__":
    args = parse_args()

    print("Simulation started.")
    show_img = False

    if args.is_train:
        dest_dir = os.path.join(data_dir, 'train_seed{:d}'.format(args.seed))
    else:
        dest_dir = os.path.join(data_dir, 'test_seed{:d}'.format(args.seed))

    if not os.path.exists(dest_dir):
        os.mkdir(dest_dir)
        print("Destination folder: {:s}".format(dest_dir))

    for i in tqdm(range(args.n_data)):
        # Generate the coordinate grid and image properties kwargs
        kwargs_data = sim_util.data_configure_simple(numPix, deltaPix, exp_time, sigma_bkg)
        image_data_class = ImageData(**kwargs_data)
        # Generate the PSF
        kernel_cut = kernel_util.cut_psf(psf_dict[i % n_psf], kernel_size)
        kwargs_psf = {'psf_type': psf_type, 'pixel_size': deltaPix, 'kernel_point_source': kernel_cut}
        #kwargs_psf = sim_util.psf_configure_simple(psf_type=psf_type, fwhm=fwhm, kernelsize=kernel_size, deltaPix=deltaPix, kernel=kernel)
        psf_class = PSF(**kwargs_psf)

        # Mean of the BNN prior
        gamma_ext_mu, theta_E_mu, gamma_mu, lens_center_mu, lens_e_mu = 0.015, 1.25, 2.0, 0.0, 0.0
        gamma_ext_sigma, theta_E_sigma, gamma_sigma, lens_center_sigma, lens_e_sigma= 0.005, 0.4, 0.05, 0.2, 0.2

        gamma_ext = np.maximum(np.random.normal(gamma_ext_mu, gamma_ext_sigma), 0)
        psi_ext = np.random.uniform(0.0, 2* np.pi)
        theta_E = np.maximum(np.random.normal(loc=theta_E_mu, scale=theta_E_sigma), 0.1)
        gamma = np.maximum(np.random.normal(gamma_mu, gamma_sigma), 1.85)
        lens_center_x = np.random.normal(lens_center_mu, lens_center_sigma)
        lens_center_y = np.random.normal(lens_center_mu, lens_center_sigma)
        lens_e1 = np.minimum(np.random.normal(lens_e_mu, lens_e_sigma), 0.9)
        lens_e2 = np.minimum(np.random.normal(lens_e_mu, lens_e_sigma), 0.9)

        kwargs_shear = {'gamma_ext': gamma_ext, 'psi_ext': psi_ext}  # shear values to the source plane
        kwargs_spemd = {'theta_E': theta_E, 'gamma': gamma, 'center_x': lens_center_x, 'center_y': lens_center_y, 'e1': lens_e1, 'e2': lens_e2}  # parameters of the deflector lens model

        # the lens model is a supperposition of an elliptical lens model with external shear
        lens_model_list = ['SPEMD', 'SHEAR_GAMMA_PSI']
        kwargs_lens = [kwargs_spemd, kwargs_shear]
        lens_model_class = LensModel(lens_model_list=lens_model_list)

        # choice of source type
        source_type = 'SERSIC'  # 'SERSIC' or 'SHAPELETS'
        source_position_mu = 0.0
        source_position_sigma = 0.1
        #sigma_source_position = 0.1
        source_x = np.random.normal(source_position_mu, source_position_sigma)
        source_y = np.random.normal(source_position_mu, source_position_sigma)

        # Sersic parameters in the initial simulation
        phi_G, q = 0.5, 0.8
        sersic_source_e1, sersic_source_e2 = param_util.phi_q2_ellipticity(phi_G, q)
        source_R_sersic_mu, source_R_sersic_sigma = 0.2, 0.1
        source_n_sersic_mu, source_n_sersic_sigma = 1.0, 0.1
        source_R_sersic = np.random.normal(source_R_sersic_mu, source_R_sersic_sigma)
        source_n_sersic = np.random.normal(source_n_sersic_mu, source_n_sersic_sigma)

        kwargs_sersic_source = {'amp': 4000, 'R_sersic': source_R_sersic, 'n_sersic': source_n_sersic, 'e1': sersic_source_e1, 'e2': sersic_source_e2, 'center_x': source_x, 'center_y': source_y}
        #kwargs_else = {'sourcePos_x': source_x, 'sourcePos_y': source_y, 'quasar_amp': 400., 'gamma1_foreground': 0.0, 'gamma2_foreground':-0.0}
        source_model_list = ['SERSIC_ELLIPSE']
        kwargs_source = [kwargs_sersic_source]
        source_model_class = LightModel(light_model_list=source_model_list)

        # lens light model
        phi_G, q = 0.9, 0.9
        lens_light_e1, lens_light_e2 = param_util.phi_q2_ellipticity(phi_G, q)
        lens_light_R_sersic_mu, lens_light_R_sersic_sigma = 0.3, 0.1
        lens_light_n_sersic_mu, lens_light_n_sersic_sigma = 1.0, 0.1
        lens_light_R_sersic = np.random.normal(lens_light_R_sersic_mu, lens_light_R_sersic_sigma)
        lens_light_n_sersic = np.random.normal(lens_light_n_sersic_mu, lens_light_n_sersic_sigma)
        kwargs_sersic_lens = {'amp': 8000, 'R_sersic': lens_light_R_sersic, 'n_sersic': lens_light_n_sersic , 'e1': lens_light_e1, 'e2': lens_light_e2, 'center_x': lens_center_x, 'center_y': lens_center_y}
        lens_light_model_list = ['SERSIC_ELLIPSE']
        kwargs_lens_light = [kwargs_sersic_lens]
        lens_light_model_class = LightModel(light_model_list=lens_light_model_list)

        lensEquationSolver = LensEquationSolver(lens_model_class)
        x_image, y_image = lensEquationSolver.findBrightImage(source_x, source_y, kwargs_lens, numImages=4,
                                                              min_distance=deltaPix, search_window=numPix * deltaPix)
        mag = lens_model_class.magnification(x_image, y_image, kwargs=kwargs_lens)
        kwargs_ps = [{'ra_image': x_image, 'dec_image': y_image,
                                   'point_amp': np.abs(mag)*1000}]  # quasar point source position in the source plane and intrinsic brightness
        point_source_list = ['LENSED_POSITION']
        point_source_class = PointSource(point_source_type_list=point_source_list, fixed_magnification_list=[False])

        kwargs_numerics = {'supersampling_factor': 1}

        imageModel = ImageModel(image_data_class, psf_class, lens_model_class, source_model_class,
                                        lens_light_model_class, point_source_class, kwargs_numerics=kwargs_numerics)

        # generate image
        img = imageModel.image(kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_ps)
        #poisson = image_util.add_poisson(img, exp_time=exp_time)
        #bkg = image_util.add_background(img, sigma_bkd=sigma_bkg)
        img = img# + bkg + poisson

        image_data_class.update_data(img)
        kwargs_data['image_data'] = img

        kwargs_model = {'lens_model_list': lens_model_list,
                        'lens_light_model_list': lens_light_model_list,
                        'source_light_model_list': source_model_list,
                        'point_source_model_list': point_source_list
                         }

        # Save image file
        img_path = os.path.join(dest_dir, 'X_{0:03d}.npy'.format(i+1))
        np.save(img_path, img)
        ##### saveing parameters
        lens_dict = kwargs_lens[0]
        shear_dict = kwargs_shear
        metadata_df  = pd.DataFrame([lens_dict], columns=lens_dict.keys())
        metadata_df['gamma_ext'] = gamma_ext
        metadata_df['psi_ext'] = psi_ext
        #df_shear = pd.DataFrame([shear_dict], columns=shear_dict.keys())
        metadata_df['path'] = img_path
        metadata_df['source_R_sersic'] = source_R_sersic
        metadata_df['source_n_sersic'] = source_n_sersic
        metadata_df['sersic_source_e1'] = sersic_source_e1
        metadata_df['sersic_source_e2'] = sersic_source_e2
        metadata_df['source_x'] = source_x
        metadata_df['source_y'] = source_y
        metadata_df['lens_light_e1'] = lens_light_e1
        metadata_df['lens_light_e2'] = lens_light_e2
        metadata_df['lens_light_R_sersic'] = lens_light_R_sersic
        metadata_df['lens_light_n_sersic'] = lens_light_n_sersic

        # lens_light_R_sersic = np.random.normal(lens_light_R_sersic_mu, lens_light_R_sersic_sigma)
        # lens_light_n_sersic

        if i > 0:
            full_metadata_df = pd.concat([full_metadata_df, metadata_df], axis =0).reset_index(drop=True)
        else:
            full_metadata_df = metadata_df

        if show_img == True:
            cmap_string = 'gray'
            cmap = plt.get_cmap(cmap_string)
            cmap.set_bad(color='k', alpha=1.)
            cmap.set_under('k')

            v_min = -4
            v_max = 2

            f, axes = plt.subplots(1, 1, figsize=(6, 6), sharex=False, sharey=False)
            ax = axes
            im = ax.matshow(np.log10(image_sim), origin='lower', vmin=v_min, vmax=v_max, cmap=cmap, extent=[0, 1, 0, 1])
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            ax.autoscale(False)
            plt.show()
    #metadata_df_podcast = metadata_df_podcast[['name', 'theta_E', 'gamma', 'center_x', 'center_y', 'e1', 'e2', 'gamma_ext', 'psi_ext', 'source_x', 'source_y', 'source_n_sersic', 'source_R_sersic', 'sersic_source_e1', 'sersic_source_e2', 'lens_light_e1', 'lens_light_e2', 'lens_light_n_sersic', 'lens_light_R_sersic']]
    metadata_path = os.path.join(dest_dir, 'metadata.csv')
    full_metadata_df.to_csv(metadata_path, index=None)
    print(full_metadata_df.columns.values)
