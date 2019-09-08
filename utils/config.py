import warnings
import numpy as np
import torch
from addict import Dict

cfg = Dict()

# Global configs
cfg.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cfg.global_seed = 1225

# Data
cfg.data = Dict(
                train_dir='data/tdlmc_train_DiagonalBNNPrior_seed1113',
                val_dir='data/tdlmc_val_DiagonalBNNPrior_seed1225',
                normalize=True,
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                X_dim=224,
                Y_cols=['lens_mass_gamma', 'lens_mass_theta_E', 'lens_mass_e1', 'lens_mass_e2',
                         'external_shear_gamma_ext', 'external_shear_psi_ext', 'src_light_magnitude',
                         'src_light_center_x', 'src_light_center_y', 'src_light_n_sersic',
                         'src_light_R_sersic', 'src_light_e1', 'src_light_e2',
                         'agn_light_magnitude'],
                plot_idx=np.arange(50),
                )
if cfg.data.train_dir == cfg.data.val_dir:
    warnings.warn("You're training and validating on the same dataset.", UserWarning, stacklevel=2)
cfg.data.Y_dim = len(cfg.data.Y_cols)

# Model
cfg.model = Dict(load_pretrained=True,
                 type='double',
                 )
if cfg.model.load_pretrained:
    # Pretrained model expects exactly this normalization
    cfg.data.mean = [0.485, 0.456, 0.406]
    cfg.data.std = [0.229, 0.224, 0.225]
if cfg.model.type == 'diagonal': # a single Gaussian w. diagonal cov
    # y_dim for the means + y_dim for the cov
    cfg.model.out_dim = cfg.data.Y_dim*2 
elif cfg.model.type == 'low_rank': # a single Gaussian w. rank-2 cov
# y_dim for the means + 3*y_dim for the cov
    cfg.model.out_dim = cfg.data.Y_dim*4
elif cfg.model.type == 'double': # two Gaussians each w. rank-2 cov
    # y_dim for the means + 2*y_dim for the cov for each Gaussian + 1 for amplitude weighting
    cfg.model.out_dim = cfg.data.Y_dim*8 + 1
else:
    raise NotImplementedError

# Optimization
cfg.optim = Dict(n_epochs=100,
                 learning_rate=1.e-4,
                 batch_size=50,
                 lr_scheduler=Dict(milestones=[15, 30, 60],
                                   gamma=0.7))

# Logging
cfg.log = Dict(checkpoint_dir='saved_models', # where to store saved models
               checkpoint_interval=5, # in epochs
               logging_interval=1, # in epochs
               )