from addict import Dict

cfg = Dict()

# Global configs
cfg.device_type = 'cuda'
cfg.global_seed = 1225

# Data
cfg.data = Dict(train_dir='/home/jwp/stage/sl/h0rton/Rung1_train_prior=DiagonalCosmoBNNPrior_seed=1113',
                val_dir='/home/jwp/stage/sl/h0rton/Rung1_val_prior=DiagonalCosmoBNNPrior_seed=1225',
                Y_cols_to_log_parameterize=['lens_mass_gamma', 'lens_mass_theta_E',
                         'lens_light_R_sersic',],
                Y_cols_to_whiten=['lens_mass_gamma', 'lens_mass_theta_E', 'lens_mass_e1', 'lens_mass_e2', 'external_shear_gamma1', 'external_shear_gamma2', 'lens_light_R_sersic',
                         'src_light_center_x', 'src_light_center_y',],
                Y_cols=['lens_mass_gamma', 'lens_mass_theta_E', 'lens_mass_e1', 'lens_mass_e2', 'external_shear_gamma1', 'external_shear_gamma2', 'lens_light_R_sersic',
                         'src_light_center_x', 'src_light_center_y',],
                n_plotting=400,
                add_noise=True,
                noise_kwargs=dict(
                                  pixel_scale=0.08,
                                  exposure_time=9600.0,
                                  magnitude_zero_point=25.9463, 
                                  read_noise=12.0, 
                                  ccd_gain=2.5,
                                  sky_brightness=20.1,
                                  seeing=0.6, 
                                  num_exposures=1, 
                                  psf_type='GAUSSIAN', 
                                  kernel_point_source=None, 
                                  truncation=5,
                                  data_count_unit='e-', 
                                  background_noise=None
                                  )
                )

# Model
cfg.model = Dict(architecture='resnet18',
                 load_state=True,
                 state_path='/home/jwp/stage/sl/h0rton/saved_models/resnet18_epoch=659_12-23-2019_21:34.mdl',
                 likelihood_class='DoubleGaussianNLL',
                 )

# Optimization
cfg.optim = Dict(n_epochs=200000,
                 learning_rate=7e-5,
                 weight_decay=1.e-5,
                 batch_size=500,
                 lr_scheduler=Dict(milestones=[10, 30, 50], # with respect to resuming point
                                   gamma=0.7))

# Logging
cfg.log = Dict(checkpoint_dir='saved_models', # where to store saved models
               checkpoint_interval=20, # in epochs
               logging_interval=1, # in epochs
               monitor_sample_images=False,
               monitor_1d_marginal_mapping=True,
               monitor_weight_distributions=False,
               )
# H0 inference
cfg.h0_inference = Dict(
                        test_dir='/home/jwp/stage/sl/h0rton/Rung1_train_prior=DiagonalCosmoBNNPrior_seed=1113',
                        n_test=100,
                        state_path='/home/jwp/stage/sl/h0rton/saved_models/resnet18_epoch=159_12-12-2019_22:01.mdl',
                        
                        )