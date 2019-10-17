How to train
============

1. Edit the configuration parameters `h0rton/config.py`. Make sure the `cfg.DATA` field agrees with the training data you generated.

2. Run

::

$python -m h0rton.train

You can visualize the training results by running

::

$tensorboard --logdir runs