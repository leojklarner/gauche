[![Project Status: Active – The project has reached a stable, usable state and is being actively developed.](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Docs](https://assets.readthedocs.org/static/projects/badges/passing-flat.svg)](https://leojklarner.github.io/gauche/)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/leojklarner/gauche/HEAD)
[![DOI:10.48550/arXiv.2212.04450](https://zenodo.org/badge/DOI/10.48550/arXiv.2212.04450.svg)](https://doi.org/10.48550/arXiv.2212.04450)
[![fair-software.eu](https://img.shields.io/badge/fair--software.eu-%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8B%20%20%E2%97%8B%20%20%E2%97%8B-orange)](https://fair-software.eu)
[![CodeFactor](https://www.codefactor.io/repository/github/leojklarner/gauche/badge)](https://www.codefactor.io/repository/github/leojklarner/gauche)
<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>

<p align="left">
  <a href="https://leojklarner.github.io/gauche/">
    <img src="https://raw.githubusercontent.com/leojklarner/gauche/main/imgs/gauche_logo.png" width="45%" />
    <img src="https://github.com/leojklarner/gauche/blob/main/imgs/gauche.gif?raw=true" width="22%" hspace="30"/>
  </a>
</p>

[Documentation](https://leojklarner.github.io/gauche/) | [Paper](https://arxiv.org/abs/2212.04450)

A Gaussian Process Library for Molecules, Proteins and Reactions.

## What's New?
|   |   |
|---|---|
| [BNN Regression on Molecules](https://leojklarner.github.io/gauche/notebooks/gp_regression_on_molecules.html)  |  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/leojklarner/gauche/blob/main/notebooks/BNN%20Regression%20on%20Molecules.ipynb)   |
| [Bayesian Optimisation Over Molecules](https://leojklarner.github.io/gauche/notebooks/bayesian_optimisation_over_molecules.html)  |  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/leojklarner/gauche/blob/main/notebooks/Bayesian%20Optimisation%20Over%20Molecules.ipynb)   |


## Install

We recommend using a conda virtual environment:.
```
conda env create -f conda_env.yml

pip install --no-deps rxnfp
pip install --no-deps drfp
pip install transformers
```

Optional for running tests.
```
pip install gpflow grakel
```

## Example usage

### BNN Regression on Molecules

|   |   |  
|---|---|
[Tutorial (BNN Regression on Molecules)](https://leojklarner.github.io/gauche/notebooks/gp_regression_on_molecules.html)  | [Docs](https://leojklarner.github.io/gauche/modules/dataloader.html)
| [![Open In Colab(https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/assets/colab-badge.svg)]([https://colab.research.google.com/github/leojklarner/gauche/blob/main/notebooks/Bayesian%20Optimisation%20Over%20Molecules.ipynb](https://colab.research.google.com/github/leojklarner/gauche/blob/main/notebooks/BNN%20Regression%20on%20Molecules.ipynb)) | |

```python
from gauche.dataloader import DataLoaderMP
from gauche.dataloader.data_utils import transform_data
from sklearn.model_selection import train_test_split

loader = DataLoaderMP()
loader.load_benchmark(dataset, dataset_paths[dataset])
loader.featurize(feature)
X = loader.features
y = loader.labels

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_set_size, random_state=i)
#  We standardise the outputs but leave the inputs unchanged
_, y_train, _, y_test, y_scaler = transform_data(X_train, y_train, X_test, y_test)
```

### Bayesian Optimisation Over Molecules

|   |   |  
|---|---|
[Tutorial (Bayesian Optimisation Over Molecules)](https://leojklarner.github.io/gauche/notebooks/bayesian_optimisation_over_molecules.html)  | [Docs](https://leojklarner.github.io/gauche/modules/kernels.html)
| [![Open In Colab(https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/leojklarner/gauche/blob/main/notebooks/Bayesian%20Optimisation%20Over%20Molecules.ipynb) | |

```python
from botorch.models.gp_regression import SingleTaskGP
from gprotorch.kernels.fingerprint_kernels.tanimoto_kernel import TanimotoKernel

# We define our custom GP surrogate model using the Tanimoto kernel

class TanimotoGP(SingleTaskGP):

    def __init__(self, train_X, train_Y):
        super().__init__(train_X, train_Y, GaussianLikelihood())
        self.mean_module = ConstantMean()
        self.covar_module = ScaleKernel(base_kernel=TanimotoKernel())
        self.to(train_X)  # make sure we're on the right device/dtype

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)
```

## Citing

If GAUCHE is useful for your work please consider citing the following paper:

```bibtex
@misc{griffiths2022gauche,
      title={GAUCHE: A Library for Gaussian Processes in Chemistry}, 
      author={Ryan-Rhys Griffiths and Leo Klarner and Henry B. Moss and Aditya Ravuri and Sang Truong and Bojana Rankovic and Yuanqi Du and Arian Jamasb and Julius Schwartz and Austin Tripp and Gregory Kell and Anthony Bourached and Alex Chan and Jacob Moss and Chengzhi Guo and Alpha A. Lee and Philippe Schwaller and Jian Tang},
      year={2022},
      eprint={2212.04450},
      archivePrefix={arXiv},
      primaryClass={physics.chem-ph}
}

```
