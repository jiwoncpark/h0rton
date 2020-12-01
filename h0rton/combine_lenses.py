"""Script to combine the lenses for a joint H0 inference

Example
-------
To run this script, pass in the integer ID of the folder in which the individual D_dt samples are stored,
with the ID of the precision ceiling folder as the optional argument::
    
    $ python h0rton/combine_lenses.py 2 --prec_version_id 0

"""

import os
import numpy as np
import argparse
import pandas as pd
import lenstronomy
print(lenstronomy.__path__)
from h0rton.configs import TestConfig
import h0rton.h0_inference.h0_utils as h0_utils
from baobab.configs import BaobabConfig
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.PointSource.point_source import PointSource
import lenstronomy.Util.data_util as data_util
from baobab.sim_utils import Imager
from baobab.sim_utils import flux_utils, metadata_utils

def parse_args():
    """Parse command-line arguments
    
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('version_id', type=int, help='ID of the version folder in experiments')
    parser.add_argument('--prec_version_id', default=0, dest='prec_version_id', type=int,
                        help='ID of the version folder in experiments corresponding to the precision ceiling. Default: 0')
    parser.add_argument('--n_test', default=200, dest='n_test', type=int,
                        help='number of lenses to combine. Default: 200')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    version_id = args.version_id
    prec_version_id = args.prec_version_id
    is_prec_ceiling = bool(version_id == prec_version_id)
    n_test = args.n_test

    version_dir = '/home/jwp/stage/sl/h0rton/experiments/v{:d}'.format(version_id)
    if is_prec_ceiling:
        test_cfg_path = os.path.join(version_dir, 'simple_mc_default.json')
    else:
        test_cfg_path = os.path.join(version_dir, 'mcmc_default.json')
    test_cfg = TestConfig.from_file(test_cfg_path)
    baobab_cfg = BaobabConfig.from_file(test_cfg.data.test_baobab_cfg_path)
    #train_val_cfg = TrainValConfig.from_file(test_cfg.train_val_config_file_path)
    # Read in truth metadata
    metadata = pd.read_csv(os.path.join(baobab_cfg.out_dir, 'metadata.csv'), index_col=None, nrows=n_test)
    # Read in summary
    summary = pd.read_csv(os.path.join(version_dir, 'summary.csv'), index_col=None, nrows=n_test)
    summary['id'] = summary.index
    true_Om0 = 0.3

    # Drop irrelevant lenses
    summary.drop(summary[summary['id']>(n_test - 1)].index, inplace=True)
    outside_rung = summary[summary['inference_time'] == 0].index
    summary.drop(outside_rung, inplace=True)
    print("Number of lenses being combined: {:d}".format(summary.shape[0]))
    print("Lenses that were discarded: ", set(np.arange(n_test)) - set(summary['id'].values))

    # Assign doubles vs quads
    summary['is_quad'] = (summary['n_img'] == 4)
    print("Doubles: ", len(summary[~summary['is_quad']]))
    print("Quads: ", len(summary[summary['is_quad']]))
    min_doubles_quads = np.min([len(summary[~summary['is_quad']]), len(summary[summary['is_quad']])])
    doubles = summary[~summary['is_quad']].iloc[:min_doubles_quads]
    quads = summary[summary['is_quad']].iloc[:min_doubles_quads]

    ####################
    # Lens combination #
    ####################
    if False:
        # 1. Gaussian D_dt
        print("Gaussian D_dt parameterization:")
        # Combine all lenses
        print("Combining all lenses...")
        combined_path_all = os.path.join(version_dir, 'combined_H0_summary.npy')
        _ = h0_utils.combine_lenses('DdtGaussian', ddt_mean=summary['D_dt_mean'].values, ddt_sigma=summary['D_dt_std'].values, z_lens=summary['z_lens'].values, z_src=summary['z_src'].values, samples_save_path=combined_path_all, corner_save_path=None, n_run=100, n_burn=500, n_walkers=20, true_Om0=true_Om0)
        # Combine all doubles
        print("Combining the {:d} doubles...".format(min_doubles_quads))
        combined_path_doubles = os.path.join(version_dir, 'combined_H0_doubles.npy')
        _ = h0_utils.combine_lenses('DdtGaussian', ddt_mean=doubles['D_dt_mean'].values, ddt_sigma=doubles['D_dt_std'].values, z_lens=doubles['z_lens'].values, z_src=doubles['z_src'].values, samples_save_path=combined_path_doubles, corner_save_path=None, n_run=100, n_burn=500, n_walkers=20, true_Om0=true_Om0)
        # Combine all quads
        print("Combining the {:d} quads...".format(min_doubles_quads))
        combined_path_quads = os.path.join(version_dir, 'combined_H0_quads.npy')
        _ = h0_utils.combine_lenses('DdtGaussian', ddt_mean=quads['D_dt_mean'].values, ddt_sigma=quads['D_dt_std'].values, z_lens=quads['z_lens'].values, z_src=quads['z_src'].values, samples_save_path=combined_path_quads, corner_save_path=None, n_run=100, n_burn=500, n_walkers=20, true_Om0=true_Om0)
        # 2. Lognormal D_dt
        print("Lognormal D_dt parameterization:")
        # Combine all lenses
        print("Combining all lenses...")
        combined_path_all_lognormal = os.path.join(version_dir, 'combined_H0_summary_lognormal.npy')
        _ = h0_utils.combine_lenses('DdtLogNorm', ddt_mu=summary['D_dt_mu'].values, ddt_sigma=summary['D_dt_sigma'].values, z_lens=summary['z_lens'].values, z_src=summary['z_src'].values, samples_save_path=combined_path_all_lognormal, corner_save_path=None, n_run=100, n_burn=500, n_walkers=20, true_Om0=true_Om0)
        # Combine all doubles
        print("Combining the {:d} doubles...".format(min_doubles_quads))
        combined_path_doubles_lognormal = os.path.join(version_dir, 'combined_H0_doubles_lognormal.npy')
        _ = h0_utils.combine_lenses('DdtLogNorm', ddt_mu=doubles['D_dt_mu'].values, ddt_sigma=doubles['D_dt_sigma'].values, z_lens=doubles['z_lens'].values, z_src=doubles['z_src'].values, samples_save_path=combined_path_doubles_lognormal, corner_save_path=None, n_run=100, n_burn=500, n_walkers=20, true_Om0=true_Om0)
        # Combine all quads
        print("Combining the {:d} quads...".format(min_doubles_quads))
        combined_path_quads_lognormal = os.path.join(version_dir, 'combined_H0_quads_lognormal.npy')
        _ = h0_utils.combine_lenses('DdtLogNorm', ddt_mu=quads['D_dt_mu'].values, ddt_sigma=quads['D_dt_sigma'].values, z_lens=quads['z_lens'].values, z_src=quads['z_src'].values, samples_save_path=combined_path_quads_lognormal, corner_save_path=None, n_run=100, n_burn=500, n_walkers=20, true_Om0=true_Om0)
        # 2. KDE
        print("KDE D_dt:")
        # Combine all lenses
        print("Combining all lenses...")
        combined_path_all_kde = os.path.join(version_dir, 'combined_H0_summary_kde.npy')
        _ = h0_utils.combine_lenses('DdtHistKDE', lens_ids=summary['id'].values, samples_dir=os.path.join(version_dir, 'mcmc_default'), z_lens=summary['z_lens'].values, z_src=summary['z_src'].values, samples_save_path=combined_path_all_kde, corner_save_path=None, n_run=100, n_burn=200, n_walkers=20, true_Om0=true_Om0, binning_method='scott')
        # Combine all doubles
        print("Combining the {:d} doubles...".format(min_doubles_quads))
        combined_path_doubles_kde = os.path.join(version_dir, 'combined_H0_doubles_kde.npy')
        _ = h0_utils.combine_lenses('DdtHistKDE', lens_ids=doubles['id'].values, samples_dir=os.path.join(version_dir, 'mcmc_default'), z_lens=doubles['z_lens'].values, z_src=doubles['z_src'].values, samples_save_path=combined_path_doubles_kde, corner_save_path=None, n_run=100, n_burn=500, n_walkers=20, true_Om0=true_Om0, binning_method='scott')
        # Combine all quads
        print("Combining the {:d} quads...".format(min_doubles_quads))
        combined_path_quads_kde = os.path.join(version_dir, 'combined_H0_quads_kde.npy')
        _ = h0_utils.combine_lenses('DdtHistKDE', lens_ids=quads['id'].values, samples_dir=os.path.join(version_dir, 'mcmc_default'), z_lens=quads['z_lens'].values, z_src=quads['z_src'].values, samples_save_path=combined_path_quads_kde, corner_save_path=None, n_run=100, n_burn=500, n_walkers=20, true_Om0=true_Om0, binning_method='scott')

    # Combine for each Einstein brightness bin
    if is_prec_ceiling:
        # If this version is the precision ceiling, compute the Einstein ring brightness for the first time.
        print("Computing Einstein ring brightness...")
        summary['lensed_E_ring_flux'] = 0.0
        summary['lensed_E_ring_mag'] = 0.0
        lens_mass_model = LensModel(lens_model_list=['PEMD', 'SHEAR_GAMMA_PSI'])
        src_light_model = LightModel(light_model_list=['SERSIC_ELLIPSE'])
        lens_light_model = LightModel(light_model_list=['SERSIC_ELLIPSE'])
        ps_model = PointSource(point_source_type_list=['LENSED_POSITION'], fixed_magnification_list=[False])
        components = ['lens_mass', 'src_light', 'agn_light', 'lens_light'] 
        for lens_i in range(n_test):
            imager = Imager(components, lens_mass_model, src_light_model, lens_light_model=lens_light_model, ps_model=ps_model, kwargs_numerics={'supersampling_factor': 1}, min_magnification=0.0, for_cosmography=True)
            imager._set_sim_api(num_pix=64, kwargs_detector=baobab_cfg.survey_object_dict[baobab_cfg.survey_info.bandpass_list[0]].kwargs_single_band(), psf_kernel_size=99, which_psf_maps=[101]) # TODO: read from BaobabConfig rather than hardcode
            imager.kwargs_src_light = metadata_utils.get_kwargs_src_light(metadata.iloc[lens_i])
            imager.kwargs_src_light = flux_utils.mag_to_amp_extended(imager.kwargs_src_light, imager.src_light_model, imager.data_api)
            imager.kwargs_lens_mass = metadata_utils.get_kwargs_lens_mass(metadata.iloc[lens_i])
            sample_ps = metadata_utils.get_nested_ps(metadata.iloc[lens_i])
            imager.for_cosmography = False
            imager._load_agn_light_kwargs(sample_ps)
            lensed_total_flux, lensed_src_img = flux_utils.get_lensed_total_flux(imager.kwargs_lens_mass, imager.kwargs_src_light, None, imager.image_model, return_image=True)
            lensed_ring_total_flux = np.sum(lensed_src_img)
            summary.loc[lens_i, 'lensed_E_ring_flux'] = lensed_ring_total_flux
            summary.loc[lens_i, 'lensed_E_ring_mag'] = data_util.cps2magnitude(lensed_ring_total_flux, baobab_cfg.survey_object_dict[baobab_cfg.survey_info.bandpass_list[0]].kwargs_single_band()['magnitude_zero_point'])
        lensed_ring_bins = np.quantile(summary['lensed_E_ring_mag'].values, [0.25, 0.5, 0.75, 1])
        lensed_ring_bins[-1] += 0.1 # buffer 
        summary['lensed_ring_bin'] = np.digitize(summary['lensed_E_ring_mag'].values, lensed_ring_bins)
        summary.to_csv(os.path.join(version_dir, 'ering_summary.csv'), index=False)
    else:
        # Simply read in the preciously computed Einstein ring brightness
        prec_version_dir = '/home/jwp/stage/sl/h0rton/experiments/v{:d}'.format(prec_version_id)
        prec_summary = pd.read_csv(os.path.join(prec_version_dir, 'ering_summary.csv'), index_col=None, nrows=n_test)
        summary['lensed_E_ring_mag'] = prec_summary['lensed_E_ring_mag'].values
        lensed_ring_bins = np.quantile(summary['lensed_E_ring_mag'].values, [0.25, 0.5, 0.75, 1])
        lensed_ring_bins[-1] += 0.1 # buffer 
        summary['lensed_ring_bin'] = np.digitize(summary['lensed_E_ring_mag'].values, lensed_ring_bins)
    
    for bin_i in range(len(lensed_ring_bins)):
        take_bin_i = (summary['lensed_ring_bin'] == bin_i)
        print("Combining {:d} lenses in bin {:d}...".format(len(summary[take_bin_i]), bin_i))
        if False:
            if is_prec_ceiling:
                combined_path_ering_i = os.path.join(version_dir, 'combined_H0_ering_{:d}.npy'.format(bin_i))
                _ = h0_utils.combine_lenses(likelihood_type='DdtGaussian', 
                                            ddt_mean=summary['D_dt_mean'][take_bin_i].values, 
                                            ddt_sigma=summary['D_dt_std'][take_bin_i].values, 
                                            z_lens=summary['z_lens'][take_bin_i].values, 
                                            z_src=summary['z_src'][take_bin_i].values, 
                                            samples_save_path=combined_path_ering_i, 
                                            corner_save_path=None, 
                                            n_run=100, 
                                            n_burn=500, 
                                            n_walkers=20, 
                                            true_Om0=true_Om0)
            combined_path_ering_i_lognormal = os.path.join(version_dir, 'combined_H0_ering_{:d}_lognormal.npy'.format(bin_i))
            _ = h0_utils.combine_lenses(likelihood_type='DdtLogNorm', 
                                        ddt_mu=summary['D_dt_mu'][take_bin_i].values, 
                                        ddt_sigma=summary['D_dt_sigma'][take_bin_i].values, 
                                        z_lens=summary['z_lens'][take_bin_i].values, 
                                        z_src=summary['z_src'][take_bin_i].values, 
                                        samples_save_path=combined_path_ering_i_lognormal, 
                                        corner_save_path=None, 
                                        n_run=100, 
                                        n_burn=500, 
                                        n_walkers=20, 
                                        true_Om0=true_Om0)
        combined_path_ering_i_kde = os.path.join(version_dir, 'combined_H0_ering_{:d}_kde.npy'.format(bin_i))
        _ = h0_utils.combine_lenses(likelihood_type='DdtHistKDE', 
                                    lens_ids=summary['id'][take_bin_i].values, 
                                    samples_dir=os.path.join(version_dir, 'mcmc_default'),
                                    z_lens=summary['z_lens'][take_bin_i].values, 
                                    z_src=summary['z_src'][take_bin_i].values,  
                                    samples_save_path=combined_path_ering_i_kde, 
                                    corner_save_path=None, 
                                    n_run=100, 
                                    n_burn=500, 
                                    n_walkers=20, 
                                    true_Om0=true_Om0, 
                                    binning_method='scott')

if __name__ == '__main__':
    main()

    


