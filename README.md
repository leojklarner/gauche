[![Project Status: Active – The project has reached a stable, usable state and is being actively developed.](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![fair-software.eu](https://img.shields.io/badge/fair--software.eu-%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8B%20%20%E2%97%8B%20%20%E2%97%8B-orange)](https://fair-software.eu)
<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>

<p align="left">
  <a href="https://leojklarner.github.io/gauche/">
    <img src="https://raw.githubusercontent.com/leojklarner/gauche/main/imgs/gauche_logo.png" width="45%" />
    <img src="https://github.com/leojklarner/gauche/blob/main/imgs/gauche.gif?raw=true" width="22%" hspace="30"/>
  </a>
</p>

A Gaussian Process Library for Molecules, Proteins and Reactions.

## Install

We recommend using a conda virtual environment:.
```
conda env create -f conda_env.yml

pip install --no-deps rxnfp
pip install --no-deps drfp
pip install transformers
pip install mordred
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

## Representations

The representations considered are summarised graphically in the figure with the tabulated references included below. For molecular graph representations, all featurisations currently included in PyTorch Geometric [2] are supported.

<p align="left">
  <a href="https://leojklarner.github.io/gauche/">
    <img src="https://raw.githubusercontent.com/leojklarner/gauche/main/imgs/overview_figure2.png" width="100%" />
  </a>
</p>

<table>
<thead>
  <tr>
    <th>Application</th>
    <th>Representation</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td rowspan="4">Molecules</td>
    <td>ECFP Fingerprints [1]</td>
  </tr>
  <tr>
    <td>Graphs [2] </td>
  </tr>
  <tr>
    <td>SMILES [3, 4]</td>
  </tr>
  <tr>
    <td>SELFIES [5] </td>
  </tr>
  <tr>
    <td rowspan="4">Chemical Reactions</td>
    <td>One-Hot Encoding</td>
  </tr>
  <tr>
    <td>Data-Driven Reaction Fingerprints [6]</td>
  </tr>
  <tr>
    <td>Differential Reaction Fingerprints [7]</td>
  </tr>
  <tr>
    <td>Reaction SMARTS</td>
  </tr>
  <tr>
    <td rowspan="2">Proteins</td>
    <td>Sequences</td>
  </tr>
  <tr>
    <td>Graphs [8]</td>
  </tr>
</tbody>
</table>

## References

[1] Rogers, D. and Hahn, M., 2010. [Extended-connectivity fingerprints](https://pubs.acs.org/doi/abs/10.1021/ci100050t). Journal of Chemical Information and Modeling, 50(5), pp.742-754.

[2] Fey, M., & Lenssen, J. E. (2019). [Fast graph representation learning with PyTorch Geometric](https://arxiv.org/abs/1903.02428). arXiv preprint arXiv:1903.02428.

[3] Weininger, D., 1988. [SMILES, a chemical language and information system. 1. Introduction to methodology and encoding rules.](https://pubs.acs.org/doi/pdf/10.1021/ci00057a005) Journal of Chemical Information and Computer Sciences, 28(1), pp.31-36.

[4] Weininger, D., Weininger, A. and Weininger, J.L., 1989. [SMILES. 2. Algorithm for generation of unique SMILES notation](https://pubs.acs.org/doi/pdf/10.1021/ci00062a008). Journal of Chemical Information and Computer Sciences, 29(2), pp.97-101.

[5] Krenn, M., Häse, F., Nigam, A., Friederich, P. and Aspuru-Guzik, A., 2020. [Self-referencing embedded strings (SELFIES): A 100% robust molecular string representation](https://iopscience.iop.org/article/10.1088/2632-2153/aba947/meta). Machine Learning: Science and Technology, 1(4), p.045024.

[6] Probst, D., Schwaller, P. and Reymond, J.L., 2022. [Reaction classification and yield prediction using the differential reaction fingerprint DRFP](https://pubs.rsc.org/en/content/articlehtml/2022/dd/d1dd00006c). Digital Discovery, 1(2), pp.91-97.

[7] Schwaller, P., Probst, D., Vaucher, A.C., Nair, V.H., Kreutter, D., Laino, T. and Reymond, J.L., 2021. [Mapping the space of chemical reactions using attention-based neural networks](https://www.nature.com/articles/s42256-020-00284-w). Nature Machine Intelligence, 3(2), pp.144-152.

[8] Jamasb, A., Viñas Torné, R., Ma, E., Du, Y., Harris, C., Huang, K., Hall, D., Lió, P. and Blundell, T., 2022. [Graphein-a Python library for geometric deep learning and network analysis on biomolecular structures and interaction networks](https://proceedings.neurips.cc/paper_files/paper/2022/hash/ade039c1db0391106a3375bd2feb310a-Abstract-Conference.html). Advances in Neural Information Processing Systems, 35, pp.27153-27167.

## Citing

If GAUCHE is useful for your work please consider citing the following paper:
