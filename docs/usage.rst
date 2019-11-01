How to train
============

1. Generate the training and validation data, e.g.

::

$python -m baobab.generate h0rton/trainval_data/train_tdlmc_diagonal_config.py

2. Edit the configuration parameters `h0rton/example_user_config.py`. Make sure the `cfg.data` field agrees with the training data you generated.

3. Run

::

$python -m h0rton.train h0rton/example_user_config.py

You can visualize the training results by running

::

$tensorboard --logdir runs