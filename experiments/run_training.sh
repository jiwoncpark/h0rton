# 2 HST orbits
python h0rton/train.py experiments/v2/train_val_cfg.json # dropout = 0.001
python h0rton/train.py experiments/v2/train_val_cfg_drop=0.005.json 
python h0rton/train.py experiments/v2/train_val_cfg_no_dropout.json
# 1 HST orbit
python h0rton/train.py experiments/v3/train_val_cfg.json # dropout = 0.001
python h0rton/train.py experiments/v3/train_val_cfg_drop=0.005.json 
python h0rton/train.py experiments/v3/train_val_cfg_no_dropout.json
# 0.5 HST orbit
python h0rton/train.py experiments/v4/train_val_cfg.json # dropout = 0.001
python h0rton/train.py experiments/v4/train_val_cfg_drop=0.005.json 
python h0rton/train.py experiments/v4/train_val_cfg_no_dropout.json