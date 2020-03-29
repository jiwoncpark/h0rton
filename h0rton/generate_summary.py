import os
import numpy as np
import argparse

def parse_args():
    """Parse command-line arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('version_id', help='version ID', type=int)
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    samples_dir = '/home/jwp/stage/sl/h0rton/experiments/v{:d}/simple_mc_default'.format(args.version_id)

    h0_dicts = [f for f in os.listdir(samples_dir) if f.startswith('h0_dict')]
    h0_dicts.sort()

    id_set = []
    mean_h0_set = []
    std_h0_set = []
    n_eff_set = []
    sampling_attempts = []
    inference_time = []

    for i, f_name in enumerate(h0_dicts):
        lens_i = int(os.path.splitext(f_name)[0].split('h0_dict_')[1])
        h0_dict = np.load(os.path.join(samples_dir, f_name), allow_pickle=True).item()
        samples = h0_dict['h0_samples']
        weights = h0_dict['h0_weights']
        #remove = np.logical_or(np.isnan(weights), weights == 0)
        remove = np.isnan(weights)
        samples = samples[~remove]
        weights = weights[~remove]
        if np.sum(weights) == 0:
            mean = -1
            std = -1
            n_eff = 0
        else:
            mean = np.average(samples, weights=weights)
            std = np.average((samples - mean)**2.0, weights=weights)**0.5
            n_eff = np.sum(weights)**2.0/(np.sum(weights**2.0))
            # Mean can be NaN even when there's no NaN in the weights
            if np.isnan(mean):
                remove = np.logical_or(np.isnan(weights), weights == 0)
                samples = samples[~remove]
                weights = weights[~remove]
                mean = np.average(samples, weights=weights)
                std = np.average((samples - mean)**2.0, weights=weights)**0.5
                n_eff = np.sum(weights)**2.0/(np.sum(weights**2.0))
        id_set.append(lens_i)
        mean_h0_set.append(mean)
        std_h0_set.append(std)
        n_eff_set.append(n_eff)
        sampling_attempts.append(h0_dict['n_sampling_attempts'])
        inference_time.append(h0_dict['inference_time'])

    summary = dict(
                   id=np.array(id_set).astype(int),
                   mean=np.array(mean_h0_set),
                   std=np.array(std_h0_set),
                   n_eff=np.array(n_eff_set),
                   inference_time=np.array(inference_time),
                   )
    h0_stats_save_path = os.path.join(samples_dir, 'summary')
    np.save(h0_stats_save_path, summary)

if __name__ == '__main__':
    main()