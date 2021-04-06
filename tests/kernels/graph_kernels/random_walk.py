"""
Verifies the PyTorch implementation of random walk graph kernels
against existing libraries.
"""

import pytest
import numpy.testing as npt
import grakel
import os
import torch
from rdkit.Chem import MolFromSmiles
from rdkit.Chem.rdmolops import GetAdjacencyMatrix
from gprotorch.dataloader.mol_prop import DataLoaderMP

benchmark_path = os.path.abspath(
    os.path.join(
        os.getcwd(), '..', '..', '..', 'data', 'property_prediction', 'ESOL.csv'
    )
)


@pytest.mark.parametrize(
    'weight, series_type, p',
    [
        (0.1, 'geometric', None),
        (0.1, 'exponential', None),
        (0.1, 'geometric', 3),
        (0.1, 'exponential', 3),
    ]
)
def test_random_walk_unlabelled(weight, series_type, p):
    """
    Tests the implementation of the random walk kernel for unlabelled graphs
    against a respective kernel from the GraKel package.
    Args:
        weight: the scaling parameter of the power series
        series_type: whether to use a geometric or exponential power series
        p: whether to use the closed for expression or a fixed number of iterations

    Returns: None

    """

    from gprotorch.kernels.graph_kernels.random_walk import RandomWalkUnlabelled

    # load the ESOL dataset
    loader = DataLoaderMP()
    loader.load_benchmark('ESOL', benchmark_path)

    # convert the SMILES representations to graphs/adjacency matrices
    adj_mats = [GetAdjacencyMatrix(MolFromSmiles(smiles)) for smiles in loader.features[:50]]
    tensor_adj_mats = [torch.Tensor(adj_mat) for adj_mat in adj_mats]
    grakel_graphs = [grakel.Graph(adj_mat) for adj_mat in adj_mats]

    # calculate the GProTorch kernel results
    with torch.no_grad():
        random_walk_gprotorch = RandomWalkUnlabelled(
            p=p, series_type=series_type, uniform_probabilities=False, normalise=True)
        random_walk_gprotorch.weight = weight
        gprotorch_results = random_walk_gprotorch.forward(tensor_adj_mats, tensor_adj_mats)

    # calculate the grakel covariance matrix
    random_walk_grakel = grakel.kernels.RandomWalk(
        lamda=weight, kernel_type=series_type, p=p, normalize=True)
    grakel_results = random_walk_grakel.fit_transform(grakel_graphs)

    npt.assert_almost_equal(
        grakel_results, gprotorch_results.numpy(),
        decimal=2
    )

if __name__ == '__main__':
    print()