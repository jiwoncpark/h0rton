# h0rton

.. image:: https://travis-ci.com/jiwoncpark/h0rton.svg?branch=master
    :target: https://travis-ci.org/jiwoncpark/h0rton

.. image:: https://readthedocs.org/projects/pybaobab/badge/?version=latest
        :target: https://pybaobab.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status

Hierarchical Bayesian inference of the Hubble constant

### How to install

1. Create a conda virtual environment and activate it.
```shell
conda create -n h0rton python=3.6 -y
conda activate h0rton
```

2. Install PyTorch stable and torchvision following the [official instructions](https://pytorch.org/), e.g.,
```shell
conda install pytorch torchvision cudatoolkit=9.2 -c pytorch
```

3. Install fastell4py
```shell
git clone https://github.com/sibirrer/fastell4py <DESIRED DESTINATION>
cd <DESIRED DESTINATION>/fastell4py
python setup.py install --user
```

4. Install `h0rton`
```shell
pip install -e . -r requirements.txt
```

### How to train

1. Edit the configuration parameters `utils/config.py`. Make sure the `cfg.DATA` field agrees with the training data you generated.

2. Run
```shell
python train.py
```

You can visualize the training results by running
```
tensorboard --logdir runs
```

Email @joshualin24 and @jiwoncpark for any questions.

There is an ongoing [document](https://www.overleaf.com/read/pswdqwttjbjr) that details our Bayesian inference method, written and maintained by Ji Won.

Challenge webpage: https://tdlmc.github.io/
Experimental design paper: https://arxiv.org/abs/1801.01506