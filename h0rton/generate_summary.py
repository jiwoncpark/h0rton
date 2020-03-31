# -*- coding: utf-8 -*-
"""Generating a cosmological summary of the H0 or D_dt H0_samples

Example
-------
To run this script, pass in the version ID and the sampling method as the argument::
    
    $ python generate_summary.py 21 simple_mc_default

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
    samples_dir = '/home/jwp/stage/sl/h0rton/experiments/v{:d}/{:s}'.format(args.version_id, args.sampling_method)
    # Read in test cfg for this version and sampling method
    test_cfg = TestConfig.from_file('/home/jwp/stage/sl/h0rton/experiments/v{:d}/{:s}.json'.format(args.version_id, args.sampling_method))
    redshifts = pd.read_csv(test_cfg.data.test_dir, index_col=None)

    H0_dicts = [f for f in os.listdir(samples_dir) if f.startswith('h0_dict')]
    H0_dicts.sort()

    summary_df = pd.DataFrame() # instantiate empty dataframe for storing summary
    for i, f_name in enumerate(H0_dicts):
        lens_i = int(os.path.splitext(f_name)[0].split('H0_dict_')[1])
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
                         inference_time=H0_dict['inference_time'],
                         )
        summary_df.append(summary_i, ignore_index=True)

    summary_df.to_csv(os.path.join(samples_dir, 'summary.csv'))

if __name__ == '__main__':
    main()