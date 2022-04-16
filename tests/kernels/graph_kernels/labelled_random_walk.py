"""
Verifies the PyTorch implementation of random walk graph kernels
against existing libraries.
"""

import numpy.testing as npt
import grakel
import scipy.sparse as sp
import os
import numpy as np
import torch
from rdkit.Chem import MolFromSmiles
from rdkit.Chem.rdmolops import GetAdjacencyMatrix
from gprotorch.dataloader.mol_prop import DataLoaderMP
from gprotorch.kernels.graph_kernels.random_walk_labelled import RandomWalk
from gprotorch.kernels.graph_kernels.graph_kernel_utils import (
    get_label_adj_mats,
    normalise_covariance,
    adj_mat_preprocessing,
)


def labelled_random_walk_kernel():

    weight = 1/16

    train_mols = [
        r"CC1(C(N2C(S1)C(C2=O)NC(=O)CC3=CC=CC=C3)C(=O)[O])C",
        r"C1CC(=O)NC(=O)C1N2CC3=C(C2=O)C=CC=C3N",
        r"CC1=CN=C(C(=C1OC)C)CS(=O)C2=NC3=C(N2)C=C(C=C3)OC",
        r"COC1=CC=C(C=C1)N2C3=C(CCN(C3=O)C4=CC=C(C=C4)N5CCCCC5=O)C(=N2)C(=O)N",
        r"CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)NC4=NC=CC(=N4)C5=CN=CC=C5.CS(=O)(=O)O",
    ]

    train_mols = [MolFromSmiles(smiles) for smiles in train_mols]
    torch_adj_mats = [
        get_label_adj_mats(x, adj_mat_format="torch_sparse")
        for x in train_mols
    ]
    torch_adj_mats, largest_mol, label_dim = adj_mat_preprocessing(
        torch_adj_mats
    )
    torch_adj_mats = torch.stack(
        [
            mat.to_dense().view(largest_mol * largest_mol * label_dim)
            for mat in torch_adj_mats
        ]
    )

    our_kernel = RandomWalk(label_dimension=label_dim)
    our_kernel.weight = weight
    with torch.no_grad():
        our_covar = our_kernel.forward(torch_adj_mats, torch_adj_mats)

    sp_adj_mats = [
        get_label_adj_mats(x, adj_mat_format="numpy_dense") for x in train_mols
    ]

    # initialise Grakel random walk kernel

    grakel_kernel = grakel.kernels.RandomWalkLabeled(lamda=weight)
    grakel_covar = np.zeros([len(train_mols), len(train_mols)])

    for i in range(len(sp_adj_mats)):
        for j in range(len(sp_adj_mats)):
            grakel_covar[i, j] = grakel_kernel.pairwise_operation(
                sp_adj_mats[i], sp_adj_mats[j]
            )

    grakel_covar = normalise_covariance(torch.from_numpy(grakel_covar))
    print(grakel_covar)
    print(our_covar)
    npt.assert_almost_equal(
        grakel_covar.numpy(), our_covar.numpy(),
    )

    print()


if __name__ == "__main__":

    labelled_random_walk_kernel()

    print()
