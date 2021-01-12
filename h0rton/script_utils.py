import os
import sys
import argparse
import random
from addict import Dict
import numpy as np
import torch

__all__ = ['parse_inference_args', 'seed_everything', 'HiddenPrints']
__all__ += ['get_batch_size', 'infer_bnn']


def infer_bnn(net, bnn_post, param_logL, test_loader,
              batch_size, n_dropout, n_samples_per_dropout,
              device, global_seed=123, mode='default'):
    """Infer with MC dropout

    """
    Y_dim = param_logL.Y_dim
    init_pos = np.empty([batch_size, n_dropout, n_samples_per_dropout, Y_dim])
    mcmc_pred = np.empty([batch_size, n_dropout, param_logL.out_dim])
    with torch.no_grad():
        net.train()
        # Send some empty forward passes through the test data without backprop
        # to adjust batchnorm weights
        # (This is often not necessary. Beware if using for just 1 lens.)
        for nograd_pass in range(5):
            for X_, Y_ in test_loader:
                X = X_.to(device)
                _ = net(X)
        # Obtain MC dropout samples
        for d in range(n_dropout):
            net.eval()
            for X_, Y_ in test_loader:
                X = X_.to(device)
                if mode == 'default_with_truth_mean':
                    Y = Y_.to(device)
                pred = net(X)
                break
            mcmc_pred_d = pred.cpu().numpy()
            # Replace BNN posterior's primary gaussian mean with truth values
            if mode == 'default_with_truth_mean':
                mcmc_pred_d[:, :Y_dim] = Y[:, :Y_dim].cpu().numpy()
            # Leave only the MCMC parameters in pred
            mcmc_pred_d = param_logL.remove_params_from_pred(mcmc_pred_d,
                                                             return_as_tensor=False)
            # Populate pred that will define the MCMC penalty function
            mcmc_pred[:, d, :] = mcmc_pred_d
            # Instantiate posterior to generate BNN samples, which will serve as
            # initial positions for walkers
            bnn_post.set_sliced_pred(mcmc_pred_d)
            init_pos[:, d, :, :] = bnn_post.sample(n_samples_per_dropout,
            sample_seed=global_seed+d)  # just the lens model params, no D_dt
    return init_pos, mcmc_pred


def get_batch_size(cfg_lens_indices, cfg_n_test, args_lens_indices_path):
    """Figure out how many consecutive lenses BNN will predict on

    Parameters
    ----------
    cfg_lens_indices : list
        lens indices specified in the config file
    cfg_n_test : int
        number of test lenses specified in the config file
    args_lens_indices_path : os.path instance or str
        path to the text file containing lens indices, from the command line

    """
    if cfg_lens_indices is None:
        if args_lens_indices_path is None:
            # Test on all n_test lenses in the test set
            n_test = cfg_n_test
            lens_range = range(n_test)
        else:
            # Test on the lens indices in a text file at the specified path
            lens_range = []
            with open(args_lens_indices_path, "r") as f:
                for line in f:
                    lens_range.append(int(line.strip()))
            n_test = len(lens_range)
            msg = ("Performing H0 inference on {n_test}"
                   " specified lenses...")
            print(msg)
    else:
        if args_lens_indices_path is None:
            # Test on the lens indices specified in the test config file
            lens_range = cfg_lens_indices
            n_test = len(lens_range)
            msg = ("Performing H0 inference on {n_test}"
                   " specified lenses...")
            print(msg)
        else:
            raise ValueError("Specific lens indices were specified in both the"
                             " test config file and the command-line argument.")
    batch_size = max(lens_range) + 1
    return batch_size, n_test, lens_range


def parse_inference_args():
    """Parse command-line arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('test_config_file_path',
                        help='path to the user-defined test config file')
    parser.add_argument('--lens_indices_path', default=None, dest='lens_indices_path', type=str,
                        help='path to a text file with specific lens indices to test on (Default: None)')
    args = parser.parse_args()
    # sys.argv rerouting for setuptools entry point
    if args is None:
        args = Dict()
        args.user_cfg_path = sys.argv[0]
        #args.n_data = sys.argv[1]
    return args


def seed_everything(global_seed):
    """Seed everything for reproducibility

    global_seed : int
        seed for `np.random`, `random`, and relevant `torch` backends

    """
    np.random.seed(global_seed)
    random.seed(global_seed)
    torch.manual_seed(global_seed)
    torch.cuda.manual_seed(global_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class HiddenPrints:
    """Hide standard output

    """
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout
