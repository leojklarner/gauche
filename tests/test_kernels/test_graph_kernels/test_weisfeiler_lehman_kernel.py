"""
Unit tests for the Weisfeiler Lehman graph kernel.
"""

import gpytorch
import graphein.molecule as gm
import numpy as np
import pytest
import torch
from sklearn.model_selection import train_test_split

from gauche import SIGP, NonTensorialInputs
from gauche.dataloader import MolPropLoader
from gauche.dataloader.data_utils import transform_data
from gauche.kernels.graph_kernels import WeisfeilerLehmanKernel

graphein_config = gm.MoleculeGraphConfig(
    node_metadata_functions=[gm.total_degree],
    edge_metadata_functions=[gm.add_bond_type],
)


@pytest.mark.parametrize(
    "node_label",
    ["element", "total_degree", "XYZ", None, 1],
)
def test_weisfeiler_lehman_kernel_node_label(node_label):
    """
    Test if Weisfeiler Lehman kernel works as intended
    when using node labels.
    """

    class GraphGP(SIGP):
        def __init__(self, train_x, train_y, likelihood):
            super().__init__(train_x, train_y, likelihood)
            self.mean = gpytorch.means.ConstantMean()
            self.covariance = WeisfeilerLehmanKernel(node_label=node_label)

        def forward(self, x):
            mean = self.mean(torch.zeros(len(x), 1)).float()
            covariance = self.covariance(x)

            # for numerical stability
            jitter = max(covariance.diag().mean().detach().item() * 1e-4, 1e-4)
            covariance += torch.eye(len(x)) * jitter
            return gpytorch.distributions.MultivariateNormal(mean, covariance)

    loader = MolPropLoader()
    loader.load_benchmark("Photoswitch")
    loader.featurize("molecular_graphs", graphein_config=graphein_config)

    X, y = loader.features, loader.labels

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    _, y_train, _, y_test, _ = transform_data(
        np.zeros_like(y_train), y_train, np.zeros_like(y_test), y_test
    )

    X_train, X_test = NonTensorialInputs(X_train), NonTensorialInputs(X_test)
    y_train = torch.tensor(y_train).flatten().float()
    y_test = torch.tensor(y_test).flatten().float()

    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = GraphGP(X_train, y_train, likelihood)

    if node_label in ["element", "total_degree"]:
        model.train()
        likelihood.train()
        output = model(X_train)

        model.eval()
        likelihood.eval()
        output = model(X_test)
    else:
        with pytest.raises(Exception):
            output = model(X_test)


@pytest.mark.parametrize(
    "edge_label",
    ["bond_type", "bond_type_and_total_degree", None, 1],
)
def test_weisfeiler_lehman_kernel_edge_label(edge_label):
    """
    Test if Weisfeiler Lehman kernel also works when
    using edge labels.
    """

    class GraphGP(SIGP):
        def __init__(self, train_x, train_y, likelihood):
            super().__init__(train_x, train_y, likelihood)
            self.mean = gpytorch.means.ConstantMean()
            self.covariance = WeisfeilerLehmanKernel(
                node_label="element", edge_label=edge_label
            )

        def forward(self, x):
            mean = self.mean(torch.zeros(len(x), 1)).float()
            covariance = self.covariance(x)

            # for numerical stability
            jitter = max(covariance.diag().mean().detach().item() * 1e-4, 1e-4)
            covariance += torch.eye(len(x)) * jitter
            return gpytorch.distributions.MultivariateNormal(mean, covariance)

    loader = MolPropLoader()
    loader.load_benchmark("Photoswitch")
    loader.featurize("molecular_graphs", graphein_config=graphein_config)

    X, y = loader.features, loader.labels

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    _, y_train, _, y_test, _ = transform_data(
        np.zeros_like(y_train), y_train, np.zeros_like(y_test), y_test
    )

    X_train, X_test = NonTensorialInputs(X_train), NonTensorialInputs(X_test)
    y_train = torch.tensor(y_train).flatten().float()
    y_test = torch.tensor(y_test).flatten().float()

    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = GraphGP(X_train, y_train, likelihood)

    if edge_label == "bond_type" or edge_label == None:
        model.train()
        likelihood.train()
        output = model(X_train)

        model.eval()
        likelihood.eval()
        output = model(X_test)
    else:
        with pytest.raises(Exception):
            output = model(X_train)
