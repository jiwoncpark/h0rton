======
h0rton
======

.. image:: https://travis-ci.com/jiwoncpark/h0rton.svg?branch=master
    :target: https://travis-ci.org/jiwoncpark/h0rton

.. image:: https://readthedocs.org/projects/h0rton/badge/?version=latest
        :target: https://h0rton.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status

Hierarchical Bayesian inference of the Hubble constant

Installation
============

0. You'll need a Fortran compiler and Fortran-compiled `fastell4py`, which you can get on a debian system by running

::

$sudo apt-get install gfortran
$git clone https://github.com/sibirrer/fastell4py.git <desired location>
$cd <desired location>/fastell4py
$python setup.py install --user

1. Virtual environments are strongly recommended, to prevent dependencies with conflicting versions. Create a conda virtual environment and activate it:

::

$conda create -n h0rton python=3.6 -y
$conda activate h0rton

2. Now do one of the following. 

**Option 2(a):** clone the repo (please do this if you'd like to contribute to the development).

::

$git clone https://github.com/jiwoncpark/h0rton.git
$cd h0rton
$pip install -e . -r requirements.txt

**Option 2(b):** pip install the release version (only recommended if you're a user).

::

$pip install h0rton


3. (Optional) To run the notebooks, add the Jupyter kernel.

::

$python -m ipykernel install --user --name h0rton --display-name "Python (h0rton)"

4. (Optional) To experiment with H0 inference using the TDLMC data, download all their data first using the provided convenience script:

::

$source h0rton/tdlmc_data/download_tdlmc_data.sh 

How to train
============

1. Edit the configuration parameters `h0rton/config.py`. Make sure the `cfg.DATA` field agrees with the training data you generated.

2. Run

::

$python -m h0rton.train

You can visualize the training results by running

::

$tensorboard --logdir runs

Feedback
========

Suggestions are always welcome! If you encounter issues or areas for improvement, please message @jiwoncpark or `make an issue
<https://github.com/jiwoncpark/h0rton/issues>`_.

There is an ongoing `document <https://www.overleaf.com/read/pswdqwttjbjr>`_ that details our Bayesian inference method, written and maintained by Ji Won.

Challenge webpage: https://tdlmc.github.io/
Experimental design paper: https://arxiv.org/abs/1801.01506