# -*- coding: utf-8 -*-
"""Generating a cosmological summary of the H0 or D_dt H0_samples

Example
-------
To run this script, pass in the version ID and the sampling method as the argument::
    
    $ python summarize.py 21 simple_mc_default

The summary will be saved to the same directory level as the sample directory.

"""
import os
import numpy as np
import pandas as pd
import argparse
import scipy.stats
from baobab.configs import BaobabConfig
from h0rton.configs import TestConfig
import h0rton.h0_inference.h0_utils as h0_utils
import h0rton.tdlmc_utils as tdlmc_utils

def parse_args():
    """Parse command-line arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('version_id', help='version ID', type=int)
    parser.add_argument('sampling_method', help='the sampling method (one of simple_mc_default, mcmc_default, hybrid', type=str)
    parser.add_argument('--rung_idx', help='the TDLMC rung index, if H0rton was run on TDLMC data', type=int, default=None)
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    # Folder where all the H0 samples live
    samples_dir = '/home/jwp/stage/sl/h0rton/experiments/v{:d}/{:s}'.format(args.version_id, args.sampling_method)
    # Read in test cfg for this version and sampling method
    test_cfg_path = os.path.join(samples_dir, '..', '{:s}.json'.format(args.sampling_method))
    test_cfg = TestConfig.from_file(test_cfg_path)
    if 'mcmc_default' in args.sampling_method:
        summarize_mcmc(samples_dir, test_cfg, 'mcmc_default', args.rung_idx)
    elif args.sampling_method == 'hybrid':
        summarize_mcmc(samples_dir, test_cfg, 'hybrid')
    elif args.sampling_method == 'simple_mc_default':
        summarize_simple_mc_default(samples_dir, test_cfg)
    else:
        raise ValueError("This sampling method is not supported. Choose one of [simple_mc_default, mcmc_default, hybrid].")

def summarize_simple_mc_default(samples_dir, test_cfg):
    """Summarize the output of simple_mc_default, i.e. the uniform H0 samples with corresponding weights

    """
    H0_dicts = [f for f in os.listdir(samples_dir) if f.startswith('h0_dict')]
    H0_dicts.sort()
    # Read in the redshift columns of metadata
    baobab_cfg = BaobabConfig.from_file(test_cfg.data.test_baobab_cfg_path)
    metadata_path = os.path.join(baobab_cfg.out_dir, 'metadata.csv')
    meta = pd.read_csv(metadata_path, index_col=None, usecols=['z_lens', 'z_src', 'n_img'])

    summary_df = pd.DataFrame() # instantiate empty dataframe for storing summary
    for i, f_name in enumerate(H0_dicts):
        lens_i = int(os.path.splitext(f_name)[0].split('h0_dict_')[1])
        # Slice meta for this lensing system
        meta_i = meta.iloc[lens_i]
        z_lens = meta_i['z_lens']
        z_src = meta_i['z_src']
        n_img = meta_i['n_img']
        # Read in H0 samples using lens identifier
        H0_dict = np.load(os.path.join(samples_dir, f_name), allow_pickle=True).item()
        H0_samples = H0_dict['h0_samples']
        weights = H0_dict['h0_weights']
        H0_normal_stats = h0_utils.get_normal_stats_naive(H0_samples, weights)
        n_eff = np.sum(weights)**2.0/(np.sum(weights**2.0))
        # Convert H0 H0_samples to D_dt
        cosmo_converter = h0_utils.CosmoConverter(z_lens, z_src)
        D_dt_samples = cosmo_converter.get_D_dt(H0_samples)
        D_dt_stats = h0_utils.get_lognormal_stats_naive(D_dt_samples, weights)
        D_dt_normal_stats = h0_utils.get_normal_stats_naive(D_dt_samples, weights)
        summary_i = dict(
                         id=lens_i,
                         measured_td_wrt0=list(H0_dict['measured_td_wrt0']),
                         H0_mean=H0_normal_stats['mean'],
                         H0_std=H0_normal_stats['std'],
                         D_dt_mu=D_dt_stats['mu'],
                         D_dt_sigma=D_dt_stats['sigma'],
                         D_dt_mean=D_dt_normal_stats['mean'],
                         D_dt_std=D_dt_normal_stats['std'],
                         n_eff=n_eff,
                         z_lens=z_lens,
                         z_src=z_src,
                         n_img=n_img,
                         inference_time=H0_dict['inference_time'],
                         )
        summary_df = summary_df.append(summary_i, ignore_index=True)
    summary_df.to_csv(os.path.join(samples_dir, '..', 'summary.csv'))
    # Output list of problem lens IDs
    problem_id = summary_df.loc[(summary_df['n_eff'] < 3) | (summary_df['H0_std'] < 1.0)]['id'].astype(int)
    with open(os.path.join(samples_dir, '..', "mcmc_default_candidates.txt"), "w") as f:
        for pid in problem_id:
            f.write(str(pid) +"\n")

def summarize_mcmc(samples_dir, test_cfg, sampling_method, rung_idx):
    """Summarize the output of mcmc_default, i.e. MCMC samples from the D_dt posterior for each lens

    """
    true_H0 = 70.0
    true_Om0 = 0.3
    if 'mcmc_default' in sampling_method:
        if rung_idx is None:
            # Read in the relevant columns of metadata, 
            baobab_cfg = BaobabConfig.from_file(test_cfg.data.test_baobab_cfg_path)
            metadata_path = os.path.join(baobab_cfg.out_dir, 'metadata.csv')
            summary_df = pd.read_csv(metadata_path, index_col=None, usecols=['z_lens', 'z_src', 'n_img'], nrows=500) # FIXME: capped test set size at 500, as the stored dataset may be much larger
        else:
            summary_df = tdlmc_utils.convert_to_dataframe(rung=rung_idx, save_csv_path=None)
            summary_df.sort_values('seed', axis=0, inplace=True)
            true_H0 = summary_df.iloc[0]['H0']
            true_Om0 = 0.27
        summary_df['id'] = summary_df.index
        summary_df['D_dt_mu'] = np.nan
        summary_df['D_dt_sigma'] = np.nan
        summary_df['H0_mean'] = np.nan
        summary_df['H0_std'] = np.nan
        summary_df['inference_time'] = 0.0
    else:
        summary_df = pd.read_csv(os.path.join(samples_dir, '..', 'summary.csv'), index_col=None) 

    D_dt_dicts = [f for f in os.listdir(samples_dir) if f.startswith('D_dt_dict')]
    D_dt_dicts.sort()
    oversampling = 20
    threshold = 1000
    # Initialize list for catastrophic lenses not solved by MCMC
    lenses_to_rerun = []
    lenses_run = []
    for i, f_name in enumerate(D_dt_dicts):
        lens_i = int(os.path.splitext(f_name)[0].split('D_dt_dict_')[1])
        lenses_run.append(lens_i)
        meta = summary_df.loc[summary_df['id']==lens_i, ['z_lens', 'z_src']].squeeze()
        # Read in D_dt samples using lens identifier
        D_dt_dict = np.load(os.path.join(samples_dir, f_name), allow_pickle=True).item()
        # Rescale D_dt samples to correct for k_ext
        uncorrected_D_dt_samples = D_dt_dict['D_dt_samples'] # [old_n_samples,]
        uncorrected_D_dt_samples = h0_utils.remove_outliers_from_lognormal(uncorrected_D_dt_samples, 3).reshape(-1, 1) # [n_samples, 1] 
        k_ext_rv = getattr(scipy.stats, test_cfg.kappa_ext_prior.dist)(**test_cfg.kappa_ext_prior.kwargs)
        k_ext = k_ext_rv.rvs(size=[len(uncorrected_D_dt_samples), oversampling]) # [n_samples, oversampling]
        if test_cfg.kappa_ext_prior.transformed:
            D_dt_samples = (uncorrected_D_dt_samples*k_ext).flatten()
        else:
            D_dt_samples = (uncorrected_D_dt_samples/(1.0 - k_ext)).flatten() # [n_samples,]
        # Compute lognormal params for D_dt and update summary
        try:
            D_dt_stats = h0_utils.get_lognormal_stats(D_dt_samples)
            D_dt_normal_stats = h0_utils.get_normal_stats(D_dt_samples)
        except:
            print("lens", lens_i)
            print("==========")
            lenses_to_rerun.append(lens_i)
            #continue
        summary_df.loc[summary_df['id']==lens_i, 'D_dt_mu'] = D_dt_stats['mu']
        summary_df.loc[summary_df['id']==lens_i, 'D_dt_sigma'] = D_dt_stats['sigma']
        summary_df.loc[summary_df['id']==lens_i, 'D_dt_mean'] = D_dt_normal_stats['mean']
        summary_df.loc[summary_df['id']==lens_i, 'D_dt_std'] = D_dt_normal_stats['std']
        # Convert D_dt samples to H0
        D_dt_samples = scipy.stats.lognorm.rvs(scale=np.exp(D_dt_stats['mu']), s=D_dt_stats['sigma'], size=oversampling*threshold)
        D_dt_samples = D_dt_samples[np.isfinite(D_dt_samples)]
        cosmo_converter = h0_utils.CosmoConverter(meta['z_lens'], meta['z_src'], H0=true_H0, Om0=true_Om0)
        H0_samples = cosmo_converter.get_H0(D_dt_samples)
        # Reject H0 samples outside H0 prior
        H0_samples = H0_samples[np.isfinite(H0_samples)]
        if len(H0_samples) > 0:
            H0_samples = H0_samples[np.logical_and(H0_samples > 50.0, H0_samples < 90.0)]
        if len(H0_samples) < threshold:
            lenses_to_rerun.append(lens_i)
        summary_df.loc[summary_df['id']==lens_i, 'H0_mean'] = np.mean(H0_samples)
        summary_df.loc[summary_df['id']==lens_i, 'H0_std'] = np.std(H0_samples)
        summary_df.loc[summary_df['id']==lens_i, 'inference_time'] += D_dt_dict['inference_time']
    # Replace existing summary
    summary_df.to_csv(os.path.join(samples_dir, '..', 'summary.csv'))
    # Output list of catastrophic/no-good lens IDs
    if sampling_method == 'mcmc_default':
        # List of lenses that skipped MCMC
        total_lenses = np.arange(test_cfg.data.n_test)
        lenses_not_run = set(list(total_lenses)) - set(list(lenses_run))
        lenses_for_hybrid = list(lenses_not_run.union(set(lenses_to_rerun)))
        with open(os.path.join(samples_dir, '..', "hybrid_candidates.txt"), "w") as f:
            for lens_i in lenses_for_hybrid:
                f.write(str(lens_i) +"\n")
    else: # hybrid case
        with open(os.path.join(samples_dir, '..', "no_good_candidates.txt"), "w") as f:
            for lens_i in lenses_to_rerun:
                f.write(str(lens_i) +"\n")

if __name__ == '__main__':
    main()