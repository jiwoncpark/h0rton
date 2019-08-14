# time_delay_lens_modeling_challenge

A private repo for a Rung 3 blind submission for the Time Delay Lens Modeling Challenge (TDLMC).

### Installation instructions

1. Create a conda virtual environment and activate it.
```shell
conda create -n tdlmc python=3.6 -y
conda activate tdlmc
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

4. Install all other dependencies
```shell
pip install -r requirements.txt
```

Email @joshualin24 and @jiwoncpark for any questions.

Challenge webpage: https://tdlmc.github.io/
Experimental design paper: https://arxiv.org/abs/1801.01506
