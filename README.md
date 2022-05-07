# GProTorch: 
Gaussian Process Library for Molecules, Proteins and General Chemistry in PyTorch

## Install

We recommend using a conda virtual environment.

```
conda create -n gprotorch python=3.8
conda activate gprotorch
pip install gpytorch botorch
pip install scikit-learn pandas pytest tqdm
conda install -c conda-forge rdkit
```

Optional for running tests.

```
pip install gpflow grakel jupyter
```

