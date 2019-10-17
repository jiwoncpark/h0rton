============
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