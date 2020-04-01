# -*- coding: utf-8 -*-
"""Generating a cosmological summary of the H0 or D_dt H0_samples

Example
-------
To run this script, pass in the version ID and the sampling method as the argument::
    
    $ python generate_summary.py 21 simple_mc_default

The summary will be saved to the same directory level as the sample directory.

"""
import os
import numpy as np
import pandas as pd
import argparse
import scipy.stats
from h0rton.configs import TestConfig
from h0rton.h0_inference import h0_utils

def parse_args():
    """Parse command-line arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('version_id', help='version ID', type=int)
    parser.add_argument('sampling_method', help='the sampling method (one of simple_mc_default, mcmc_default, hybrid', type=str)
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    # Folder where all the H0 samples live
    samples_dir = '/home/jwp/stage/sl/h0rton/experiments/v{:d}/{:s}'.format(args.version_id, args.sampling_method)
    # Read in test cfg for this version and sampling method
    test_cfg_path = os.path.join(samples_dir, '..', '{:s}.json'.format(args.sampling_method))
    test_cfg = TestConfig.from_file(test_cfg_path)
    if args.sampling_method == 'simple_mc_default':
        summarize_simple_mc_default(samples_dir, test_cfg)
    elif args.sampling_method == 'mcmc_default':
        summarize_mcmc(samples_dir, test_cfg, 'mcmc_default')
    elif args.sampling_method == 'hybrid':
        summarize_mcmc(samples_dir, test_cfg, 'hybrid')
    else:
        raise ValueError("This sampling method is not supported. Choose one of [simple_mc_default, mcmc_default, hybrid].")

def summarize_simple_mc_default(samples_dir, test_cfg):
    """Summarize the output of simple_mc_default, i.e. the uniform H0 samples with corresponding weights

    """
    H0_dicts = [f for f in os.listdir(samples_dir) if f.startswith('h0_dict')]
    H0_dicts.sort()
    # Read in the redshift columns of metadata
    metadata_path = os.path.join(test_cfg.data.test_dir, 'metadata.csv')
    redshifts = pd.read_csv(metadata_path, index_col=None, usecols=['z_lens', 'z_src'])

    summary_df = pd.DataFrame() # instantiate empty dataframe for storing summary
    for i, f_name in enumerate(H0_dicts):
        lens_i = int(os.path.splitext(f_name)[0].split('h0_dict_')[1])
        # Slice redshifts for this lensing system
        redshifts_i = redshifts.iloc[lens_i]
        z_lens = redshifts_i['z_lens']
        z_src = redshifts_i['z_src']
        # Read in H0 samples using lens identifier
        H0_dict = np.load(os.path.join(samples_dir, f_name), allow_pickle=True).item()
        H0_samples = H0_dict['h0_samples']
        weights = H0_dict['h0_weights']
        remove = np.isnan(weights)
        H0_samples = H0_samples[~remove]
        weights = weights[~remove]
        if np.sum(weights) == 0:
            H0_mean = -1
            H0_std = -1
            n_eff = 0
            D_dt_mu = -1
            D_dt_sigma = -1
            z_lens = -1
            z_src = -1
        else:
            H0_mean = np.average(H0_samples, weights=weights)
            H0_std = np.average((H0_samples - H0_mean)**2.0, weights=weights)**0.5
            n_eff = np.sum(weights)**2.0/(np.sum(weights**2.0))
            # Mean can be NaN even when there's no NaN in the weights
            if np.isnan(H0_mean):
                remove = np.logical_or(np.isnan(weights), weights == 0)
                H0_samples = H0_samples[~remove]
                weights = weights[~remove]
                H0_mean = np.average(H0_samples, weights=weights)
                H0_std = np.average((H0_samples - H0_mean)**2.0, weights=weights)**0.5
                n_eff = np.sum(weights)**2.0/(np.sum(weights**2.0))
            # Convert H0 H0_samples to D_dt
            cosmo_converter = h0_utils.CosmoConverter(z_lens, z_src)
            D_dt_samples = cosmo_converter.get_D_dt(H0_samples)
            D_dt_stats = h0_utils.get_lognormal_stats(D_dt_samples, weights)
            D_dt_mu = D_dt_stats['mu']
            D_dt_sigma = D_dt_stats['sigma']

        summary_i = dict(
                         id=lens_i,
                         H0_mean=H0_mean,
                         H0_std=H0_std,
                         D_dt_mu=D_dt_mu,
                         D_dt_sigma=D_dt_sigma,
                         n_eff=n_eff,
                         z_lens=z_lens,
                         z_src=z_src,
                         inference_time=H0_dict['inference_time'],
                         )
        summary_df = summary_df.append(summary_i, ignore_index=True)
    summary_df.to_csv(os.path.join(samples_dir, '..', 'summary.csv'))
    # Output list of problem lens IDs
    problem_id = summary_df.loc[(summary_df['n_eff'] < 10) | (summary_df['H0_std'] < 1.0)]['id'].astype(int)
    with open(os.path.join(samples_dir, '..', "mcmc_default_candidates.txt"), "w") as f:
        for pid in problem_id:
            f.write(str(pid) +"\n")

def summarize_mcmc(samples_dir, test_cfg, sampling_method):
    """Summarize the output of mcmc_default, i.e. MCMC samples from the D_dt posterior for each lens

    """
    D_dt_dicts = [f for f in os.listdir(samples_dir) if f.startswith('D_dt_dict')]
    D_dt_dicts.sort()
    oversampling_factor = 10
    n_sample_threshold = 5000
    # Read in summary generated from the simple MC run
    summary_df = pd.read_csv(os.path.join(samples_dir, '..', 'summary.csv'), index_col=None)
    # Initialize list for catastrophic lenses not solved by MCMC
    lenses_to_rerun = []
    for i, f_name in enumerate(D_dt_dicts):
        lens_i = int(os.path.splitext(f_name)[0].split('D_dt_dict_')[1])
        # Read in D_dt samples using lens identifier
        D_dt_dict = np.load(os.path.join(samples_dir, f_name), allow_pickle=True).item()
        # Rescale D_dt samples to correct for k_ext
        uncorrected_D_dt_samples = D_dt_dict['D_dt_samples'] # [old_n_samples,]
        uncorrected_D_dt_samples = h0_utils.remove_outliers_from_lognormal(uncorrected_D_dt_samples, 3).reshape(-1, 1) # [n_samples, 1] 
        k_ext_rv = getattr(scipy.stats, test_cfg.kappa_ext_prior.dist)(**test_cfg.kappa_ext_prior.kwargs)
        k_ext = k_ext_rv.rvs(size=[len(uncorrected_D_dt_samples), oversampling_factor]) # [n_samples, oversampling_factor]
        D_dt_samples = (uncorrected_D_dt_samples/(1.0 - k_ext)).squeeze() # [n_samples,]
        # Compute lognormal params for D_dt and update summary
        D_dt_stats = h0_utils.get_lognormal_stats(D_dt_samples)
        summary_df.loc[summary_df['id']==lens_i, 'D_dt_mu'] = D_dt_stats['mu']
        summary_df.loc[summary_df['id']==lens_i, 'D_dt_sigma'] = D_dt_stats['sigma']
        # Convert D_dt samples to H0
        D_dt_samples = scipy.stats.lognorm.rvs(scale=np.exp(D_dt_stats['mu']), s=D_dt_stats['sigma'], size=n_sample_threshold)
        redshifts = summary_df.loc[summary_df['id']==lens_i].squeeze()
        cosmo_converter = h0_utils.CosmoConverter(redshifts['z_lens'], redshifts['z_src'])
        H0_samples = cosmo_converter.get_H0(D_dt_samples)
        # Reject H0 samples outside H0 prior
        H0_samples = H0_samples[np.logical_and(H0_samples > 50.0, H0_samples < 90.0)]
        if len(H0_samples) < n_sample_threshold:
            summary_df.loc[summary_df['id']==lens_i, 'H0_mean'] = -1
            summary_df.loc[summary_df['id']==lens_i, 'H0_std'] = -1
            lenses_to_rerun.append(lens_i)
        else:
            # Compute normal params for H0 and update summary
            summary_df.loc[summary_df['id']==lens_i, 'H0_mean'] = np.mean(H0_samples)
            summary_df.loc[summary_df['id']==lens_i, 'H0_std'] = np.std(H0_samples)
        # Add extra inference time due to MCMC
        summary_df.loc[summary_df['id']==lens_i, 'inference_time'] += D_dt_dict['inference_time']
    # Replace existing summary
    summary_df.to_csv(os.path.join(samples_dir, '..', 'summary.csv'))
    # Output list of catastrophic/no-good lens IDs
    if sampling_method == 'mcmc_default':
        with open(os.path.join(samples_dir, '..', "hybrid_candidates.txt"), "w") as f:
            for lens_i in lenses_to_rerun:
                f.write(str(lens_i) +"\n")
    else: # hybrid case
        with open(os.path.join(samples_dir, '..', "no_good_candidates.txt"), "w") as f:
            for lens_i in lenses_to_rerun:
                f.write(str(lens_i) +"\n")

if __name__ == '__main__':
    main()