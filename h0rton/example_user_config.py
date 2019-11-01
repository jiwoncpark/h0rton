import numpy as np
from h0rton.baobab_data import train_tdlmc_diagonal_config, val_tdlmc_diagonal_config
from addict import Dict

cfg = Dict()

# Global configs
cfg.device_type = 'cuda'
cfg.global_seed = 1225

# Data
cfg.data = Dict(train_dir=None,
                val_dir=None,
                train_baobab_cfg_path=train_tdlmc_diagonal_config.__file__,
                val_baobab_cfg_path=val_tdlmc_diagonal_config.__file__,
                normalize_image=True,
                mean_image=[0.485, 0.456, 0.406],
                std_image=[0.229, 0.224, 0.225],
                X_dim=224,
                whiten_features=True,
                log_parameterize=[True, True, False, False, True, True, True, False, False],
                Y_cols=['lens_mass_gamma', 'lens_mass_theta_E', 'lens_mass_e1', 'lens_mass_e2',
                         'external_shear_gamma_ext', 'external_shear_psi_ext', 'lens_light_R_sersic',
                         'src_light_center_x', 'src_light_center_y',],
                plot_idx=np.arange(100),
                )

# Model
cfg.model = Dict(load_pretrained=True,
                 likelihood_class='DoubleGaussianNLL',
                 )

# Optimization
cfg.optim = Dict(n_epochs=100,
                 learning_rate=1.e-4,
                 batch_size=50,
                 lr_scheduler=Dict(milestones=[50, 90],
                                   gamma=0.7))

# Logging
cfg.log = Dict(checkpoint_dir='saved_models', # where to store saved models
               checkpoint_interval=1, # in epochs
               logging_interval=1, # in epochs
               )