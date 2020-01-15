from addict import Dict

cfg = Dict()

# Global configs
cfg.device_type = 'cuda'
cfg.global_seed = 1234

# Data
cfg.data = Dict(train_dir='/home/jwp/stage/sl/h0rton/Rung1_train_prior=DiagonalCosmoBNNPrior_seed=1113',
                val_dir='/home/jwp/stage/sl/h0rton/Rung1_val_prior=DiagonalCosmoBNNPrior_seed=1225',
                Y_cols_to_log_parameterize=[],
                Y_cols_to_whiten=['lens_mass_gamma', 'lens_mass_theta_E', 'lens_mass_e1', 'lens_mass_e2', 'external_shear_gamma1', 'external_shear_gamma2', 'lens_light_R_sersic',
                'src_light_center_x', 'src_light_center_y',],
                Y_cols=['lens_mass_gamma', 'lens_mass_theta_E', 'lens_mass_e1', 'lens_mass_e2', 'external_shear_gamma1', 'external_shear_gamma2', 'lens_light_R_sersic',
                'src_light_center_x', 'src_light_center_y',],
                Y_cols_latex_names=[r"$\gamma_\mathrm{lens}$", r"$\theta_E (^{\prime \prime})$", r"$e_1$", r"$e_2$", r"$\gamma_1$", r"$\gamma_2$", r"$\theta_E (^{\prime \prime})$", r"$x_\mathrm{src} (^{\prime \prime})$", r"$y_\mathrm{src} (^{\prime \prime})$",],
                Y_cols_range=[0.05, 0.05, 0.05, 0.05, 0.005, 0.005, 0.05, 0.005, 0.005],
                n_plotting=100,
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
cfg.model = Dict(architecture='resnet34',
                 load_state=False,
                 state_path='/home/jwp/stage/sl/h0rton/saved_models/resnet18_epoch=354_01-14-2020_18:39.mdl',
                 likelihood_class='DoubleGaussianNLL',
                 )

# Optimization
cfg.optim = Dict(n_epochs=1000,
                 learning_rate=3e-4,
                 weight_decay=1.e-5,
                 batch_size=500,
                 lr_scheduler=Dict(milestones=[400, 450, 500, 550],# with respect to resuming point
                                   gamma=0.5))

# Logging
cfg.log = Dict(checkpoint_dir='overnight_models', # where to store saved models
               checkpoint_interval=5, # in epochs
               logging_interval=5, # in epochs
               monitor_sample_images=False,
               monitor_1d_marginal_mapping=True,
               monitor_weight_distributions=False,
               )