"""
Contains re-usable utility functions for calculating graph kernels.
"""

import torch
import numpy as np
import scipy.sparse as sp
from collections import defaultdict


def kronecker_inner_product(mat1, mat2):
    r"""
    This function calculates the Kronecker inner product.
    Given two matrices X\in\mathbb{R}^{N \times M \times d}
    and Y\in\mathbb{R}^{P \times Q \times d}, indexed by
    a,b,c and d,e,f respectively, this function calculates
    the Kronecker product between the indices a, d, b and e,
    and fills the resulting Kronecker matrix with the inner
    product of the corresponding vectors in dimensions c and f, i.e.
    z_{Pa+d,Qb+e} = \langle \mathbf{x}_{a,b,:}, \mathbf{y}_{d,e,:}
    \rangle = \sum_{c,f=0}^{d-1} x_{a,b,c}, y_{d,e,f}
    assuming that the matrices are 0-indexed.

    Args:
        mat1: left-hand matrix
        mat2: right-hand matrix

    Returns: Kronecker inner product matrix

    """

    kron_size = (mat1.shape[0] * mat2.shape[0], mat1.shape[1] * mat2.shape[1])
    return torch.einsum("abc,dec->adbe", mat1, mat2).reshape(kron_size)


def normalise_covariance(covariance_matrix):
    """
    Scales the covariance matrix to [0,1] by applying k(G1,G2) = k(G1,G2)/sqrt(k(G1,G1)*k(G2,G2))
    this is necessary when using different-size graphs as larger graphs will have more possible random walks
    and a smaller graph might be more similar to a larger graph than to itself.
    Args:
        covariance_matrix: the covariance matrix to scale

    Returns: the normalised covariance matrix

    """

    normalisation_factor = torch.unsqueeze(
        torch.sqrt(torch.diagonal(covariance_matrix)), -1
    )
    normalisation_factor = normalisation_factor * torch.transpose(
        normalisation_factor, -2, -1
    )
    return torch.div(covariance_matrix, normalisation_factor)


def get_label_adj_mats(mol, adj_mat_format):
    """

    Args:
        mol:
        adj_mat_format:

    Returns:

    """

    assert adj_mat_format in [
        "torch_sparse",
        "scipy_sparse",
        "torch_dense",
        "numpy_dense",
    ], f"Invalid adjacency matrix format: {adj_mat_format}"

    adj_mats = defaultdict(lambda: [[], []])
    num_atoms = mol.GetNumAtoms()

    for bond in mol.GetBonds():

        start_atom = bond.GetBeginAtom().GetSymbol()
        start_idx = bond.GetBeginAtomIdx()
        end_atom = bond.GetEndAtom().GetSymbol()
        end_idx = bond.GetEndAtomIdx()

        label = hash(
            frozenset([start_atom, end_atom, bond.GetBondTypeAsDouble()])
        )

        adj_mats[label][0].extend([start_idx, end_idx])
        adj_mats[label][1].extend([end_idx, start_idx])

    if adj_mat_format == "torch_sparse":

        def mat_transform(indices):
            mat = torch.sparse_coo_tensor(
                torch.LongTensor(indices),
                torch.ones(len(indices[0])),
                (num_atoms, num_atoms),
            ).coalesce()
            return mat

    elif adj_mat_format == "scipy_sparse":

        def mat_transform(indices):
            mat = sp.coo_matrix(
                (np.ones(len(indices[0])), np.array(indices)),
                (num_atoms, num_atoms),
            ).tocsr()
            return mat

    elif adj_mat_format == "torch_dense":

        def mat_transform(indices):
            mat = torch.zeros((num_atoms, num_atoms))
            mat[indices] = 1
            return mat

    elif adj_mat_format == "numpy_dense":

        def mat_transform(indices):
            mat = np.zeros((num_atoms, num_atoms))
            mat[tuple(indices)] = 1
            return mat

    else:
        raise NotImplementedError(
            f"Invalid adjacency matrix format: {adj_mat_format}"
        )

    adj_mats = {k: mat_transform(v) for k, v in adj_mats.items()}

    return adj_mats, num_atoms


def adj_mat_preprocessing(adj_mats, dense=False):

    assert all(adj_mats), 'Found empty adjacency matrices.'

    largest_mol = max([num_atoms for _, num_atoms in adj_mats])
    train_labels = {label for mol, _ in adj_mats for label in mol}
    train_labels = {label: i for i, label in enumerate(train_labels)}

    processed_adj_mats = []

    for adj_mat, _ in adj_mats:
        indices = []
        labels = []
        for label, mat in adj_mat.items():
            indices.append(mat.indices())
            labels.append(torch.full([mat.indices().shape[1]], train_labels[label]))
        indices = torch.concat(indices, 1)
        labels = torch.concat(labels)
        processed_adj_mats.append(torch.sparse_coo_tensor(
            indices=torch.vstack([indices, labels]),
            values=torch.ones_like(labels).float(),
            size=(largest_mol, largest_mol, len(train_labels))
        ))

    if dense:
        processed_adj_mats = [adj_mat.to_dense() for adj_mat in processed_adj_mats]

    return processed_adj_mats, largest_mol, len(train_labels)
