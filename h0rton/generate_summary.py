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
    if args.sampling_method == 'simple_mc_default':
        # Read in test cfg for this version and sampling method
        test_cfg = TestConfig.from_file(os.path.join(samples_dir, '..', '{:s}.json'.format(args.sampling_method)))
        redshifts = pd.read_csv(os.path.join(test_cfg.data.test_dir, 'metadata.csv'), index_col=None, usecols=['z_lens', 'z_src'])
        summarize_simple_mc_default(samples_dir, redshifts)
    elif args.sampling_method == 'mcmc_default':
        summarize_mcmc_default(samples_dir)
    elif args.sampling_ethod == 'hybrid':
        pass
    else:
        raise ValueError("This sampling method is not supported. Choose one of [simple_mc_default, mcmc_default, hybrid].")

def summarize_simple_mc_default(samples_dir, redshifts):
    """Summarize the output of simple_mc_default, i.e. the uniform H0 samples with corresponding weights

    """
    H0_dicts = [f for f in os.listdir(samples_dir) if f.startswith('h0_dict')]
    H0_dicts.sort()

    summary_df = pd.DataFrame() # instantiate empty dataframe for storing summary
    for i, f_name in enumerate(H0_dicts):
        lens_i = int(os.path.splitext(f_name)[0].split('h0_dict_')[1])
        # Slice redshifts for this lensing system
        redshifts_i = redshifts.iloc[lens_i]
        z_lens = redshifts_i['z_lens']
        z_src = redshifts_i['z_src']
        # Read in H0 H0_samples using lens identifier
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

def summarize_mcmc_default(samples_dir):
    """Summarize the output of mcmc_default, i.e. MCMC samples from the D_dt posterior for each lens

    """
    

if __name__ == '__main__':
    main()