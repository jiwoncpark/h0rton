from types import SimpleNamespace as SNS
import torch

cfg = SNS()

# Global configs
cfg.DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cfg.GLOBAL_SEED = 1113

# Data
cfg.DATA = SNS()
cfg.DATA.TRAIN = 'data/tdlmc_val_DiagonalBNNPrior_seed1225'
cfg.DATA.VAL = 'data/tdlmc_val_DiagonalBNNPrior_seed1225'
cfg.DATA.NORMALIZE = True
cfg.DATA.MEAN = [0.485, 0.456, 0.406]
cfg.DATA.STD = [0.229, 0.224, 0.225]
cfg.DATA.X_DIM = 224
cfg.DATA.Y_COLS = ['lens_mass_center_x', 'lens_mass_center_y', 'lens_mass_gamma',
 'lens_mass_theta_E', 'lens_mass_e1', 'lens_mass_e2',
 'external_shear_gamma_ext', 'external_shear_psi_ext', 'src_light_magnitude',
 'src_light_center_x', 'src_light_center_y', 'src_light_n_sersic',
 'src_light_R_sersic', 'src_light_e1', 'src_light_e2', 'lens_light_magnitude',
 'lens_light_center_x', 'lens_light_center_y', 'lens_light_n_sersic',
 'lens_light_R_sersic', 'lens_light_e1', 'lens_light_e2',
 'agn_light_magnitude']
cfg.DATA.Y_DIM = len(cfg.DATA.Y_COLS)

# Model
cfg.MODEL = SNS()
cfg.MODEL.LOAD_PRETRAINED = True
if cfg.MODEL.LOAD_PRETRAINED:
    # Pretrained model expects exactly this normalization
    cfg.DATA.MEAN = [0.485, 0.456, 0.406]
    cfg.DATA.STD = [0.229, 0.224, 0.225]
cfg.MODEL.TYPE = 'mixture'
if cfg.MODEL.TYPE == 'diagonal': # a single Gaussian w. diagonal cov
    # y_dim for the means + y_dim for the cov
    cfg.MODEL.OUT_DIM = cfg.DATA.Y_DIM*2 
elif cfg.MODEL.TYPE == 'low_rank': # a single Gaussian w. rank-2 cov
# y_dim for the means + 3*y_dim for the cov
    cfg.MODEL.OUT_DIM = cfg.DATA.Y_DIM*4
elif cfg.MODEL.TYPE == 'mixture': # two Gaussians each w. rank-2 cov
    # y_dim for the means + 2*y_dim for the cov for each Gaussian + 1 for amplitude weighting
    cfg.MODEL.OUT_DIM = cfg.DATA.Y_DIM*8 + 1
else:
    raise NotImplementedError

# Optimization
cfg.OPTIM = SNS()
cfg.OPTIM.N_EPOCHS = 10
cfg.OPTIM.BATCH_SIZE = 20
cfg.OPTIM.LEARNING_RATE = 1.e-4

# Logging
cfg.LOG = SNS()
cfg.LOG.CHECKPOINT_DIR = 'saved_models' # where to store saved models
cfg.LOG.CHECKPOINT_INTERVAL = 1 # in epochs