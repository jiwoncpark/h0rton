import numpy as np
from addict import Dict

cfg = Dict()

cfg.name = 'v7'
cfg.destination_dir = '/home/jwp/stage/sl/h0rton'
cfg.seed = 1225 # random seed
cfg.bnn_prior_class = 'DiagonalCosmoBNNPrior'
cfg.n_data = 512 # number of images to generate
cfg.train_vs_val = 'val'
cfg.components = ['lens_mass', 'external_shear', 'src_light', 'lens_light', 'agn_light']

cfg.selection = dict(
                 magnification=dict(
                                    min=2.0
                                    ),
                 initial=["lambda x: x['lens_mass']['theta_E'] > 0.5",],
                 )

cfg.survey_info = dict(
                       survey_name="HST",
                       bandpass_list=["TDLMC_F160W"],
                       override_obs_kwargs=dict(
                                                exposure_time=5400.0
                                                )
                       )

cfg.psf = dict(
           type='PIXEL', # string, type of PSF ('GAUSSIAN' and 'PIXEL' supported)
           kernel_size=91, # dimension of provided PSF kernel, only valid when profile='PIXEL'
           which_psf_maps=[101], # None if rotate among all available PSF maps, else seed number of the map to generate all images with that map
           )

cfg.numerics = dict(
                supersampling_factor=1)

cfg.image = dict(
             num_pix=64, # cutout pixel size
             inverse=False, # if True, coord sys is ra to the left, if False, to the right 
             squeeze_bandpass_dimension=False
             )

cfg.bnn_omega = dict(
                 # Inference hyperparameters defining the cosmology
                 cosmology = dict(
                                  H0=70.0, # Hubble constant at z = 0, in [km/sec/Mpc]
                                  Om0=0.3, # Omega matter: density of non-relativistic matter in units of the critical density at z=0.
                                  ),
                 # Hyperparameters of lens and source redshifts
                 redshift = dict(
                                model='independent_dist',
                                # Grid of redshift to sample from
                                z_lens=dict(
                                            dist='normal',
                                            mu=0.5,
                                            sigma=0.2,
                                            lower=0.0
                                            ),
                                z_src=dict(
                                           dist='normal',
                                           mu=2.0,
                                           sigma=0.4,
                                           lower=0.0
                                           ),
                                min_diff=0.0,
                                ),
                 # Hyperparameters of line-of-sight structure
                 LOS = dict(
                            kappa_ext = dict(
                                            dist='transformed_kappa_normal',
                                            mu=1.0,
                                            sigma=0.025,
                                             ),
                            ),
                 magnification = dict(
                                      frac_error_sigma=0.1
                                      ),
                 # Hyperparameters and numerics for inferring the velocity dispersion for a given lens model
                 kinematics = dict(
                                   calculate_vel_disp=False,
                                   vel_disp_frac_err_sigma=0.05,
                                   anisotropy_model='analytic',
                                   kwargs_anisotropy=dict(
                                                          aniso_param=1.0
                                                          ),
                                   kwargs_aperture=dict(
                                                        aperture_type='slit',
                                                        center_ra=0.0,
                                                        center_dec=0.0,
                                                        width=1.0, # arcsec
                                                        length=1.0, # arcsec
                                                        angle=0.0,
                                                        ),
                                   kwargs_psf=dict(
                                                  psf_type='GAUSSIAN',
                                                  fwhm=0.6
                                                  ),
                                   kwargs_numerics=dict(
                                                       sampling_number=1000,
                                                       interpol_grid_num=1000,
                                                       log_integration=True,
                                                       max_integrate=100,
                                                       min_integrate=0.001
                                                       ),
                                   ),
                 time_delays = dict(
                                    calculate_time_delays=True,
                                    error_sigma=0.25,
                                    #frac_error_sigma=0.1,
                                    ),
                 lens_mass = dict(
                                 profile='PEMD', # only available type now
                                 # Normal(mu, sigma^2)
                                 center_x = dict(
                                          dist='normal', # one of ['normal', 'beta']
                                          mu=0.0,
                                          sigma=0.07,
                                          ),
                                 center_y = dict(
                                          dist='normal',
                                          mu=0.0,
                                          sigma=0.07,
                                          ),
                                 # Lognormal(mu, sigma^2)
                                 gamma = dict(
                                              dist='normal',
                                              mu=2.0,
                                              sigma=0.1,
                                              ),
                                 theta_E = dict(
                                                dist='normal',
                                                mu=1.1,
                                                sigma=0.1,
                                                ),
                                 # Beta(a, b)
                                 q = dict(
                                           dist='normal',
                                           mu=0.7,
                                           sigma=0.15,
                                           upper=1.0,
                                           lower=0.3,
                                           ),
                                 phi = dict(
                                           dist='uniform',
                                           lower=-np.pi*0.5,
                                           upper=np.pi*0.5
                                           ),
                                 ),

                 external_shear = dict(
                                       profile='SHEAR_GAMMA_PSI',
                                       gamma_ext = dict(
                                                        dist='uniform',
                                                        lower=0.0,
                                                        upper=0.05,
                                                        ),
                                       psi_ext = dict(
                                           dist='uniform',
                                           lower=-np.pi*0.5,
                                           upper=np.pi*0.5,
                                           ),
                                       ),

                 lens_light = dict(
                                  profile='SERSIC_ELLIPSE', # only available type now
                                  # Centered at lens mass
                                  # Lognormal(mu, sigma^2)
                                  magnitude = dict(
                                             dist='uniform',
                                           lower=17.0,
                                           upper=19.0,
                                           ),
                                  n_sersic = dict(
                                                  dist='normal',
                                                  mu=3.0,
                                                  sigma=0.5,
                                                  lower=2.0,
                                                  ),
                                  R_sersic = dict(
                                                  dist='normal',
                                                  mu=0.8,
                                                  sigma=0.15,
                                                  lower=0.6,
                                                  ),
                                  # Beta(a, b)
                                  q = dict(
                                           dist='normal',
                                           mu= 0.85,
                                           sigma=0.15,
                                           upper=1.0,
                                           ),
                                  phi = dict(
                                           dist='uniform',
                                           lower=-np.pi*0.5,
                                           upper=np.pi*0.5,
                                           ),
                                  ),

                 src_light = dict(
                                profile='SERSIC_ELLIPSE', # only available type now
                                # Lognormal(mu, sigma^2)
                                magnitude = dict(
                                             dist='uniform',
                                             lower=20.0,
                                             upper=25
                                             ),
                                n_sersic = dict(
                                                dist='normal',
                                                mu=3.0,
                                                sigma=0.5,
                                                lower=0.0,
                                                ),
                                R_sersic = dict(
                                                dist='normal',
                                                mu=0.35,
                                                sigma=0.05,
                                                lower=0.25,
                                                upper=0.45,
                                                ),
                                # Normal(mu, sigma^2)
                                center_x = dict(
                                                dist='uniform',
                                                lower=-0.2,
                                                upper=0.2
                                                ),
                                center_y = dict(
                                                dist='uniform',
                                                lower=-0.2,
                                                upper=0.2
                                                ),
                                q = dict(
                                         dist='normal',
                                         mu= 0.6,
                                         sigma=0.1,
                                         upper=1.0,
                                         lower=0.3,
                                         ),
                                phi = dict(
                                           dist='uniform',
                                           lower=-np.pi*0.5,
                                           upper=np.pi*0.5,
                                           ),
                                ),

                 agn_light = dict(
                                 profile='LENSED_POSITION', # contains one of 'LENSED_POSITION' or 'SOURCE_POSITION'
                                 # Centered at host
                                 # Pre-magnification, image-plane amplitudes if 'LENSED_POSITION'
                                 # Lognormal(mu, sigma^2)
                                 magnitude = dict(
                                             dist='uniform',
                                             lower=20.0,
                                             upper=22.5
                                             ),
                                 ),
                 )