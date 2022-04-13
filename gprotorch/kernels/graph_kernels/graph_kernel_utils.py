"""
Contains re-usable utility functions for calculating graph kernels.
"""

import torch
import pandas as pd
import scipy.sparse as sp
import numpy as np


def normalise_covariance(covariance_matrix):
    """
    Scales the covariance matrix to [0,1] by applying k(G1,G2) = k(G1,G2)/sqrt(k(G1,G1)*k(G2,G2))
    this is necessary when using different-size graphs as larger graphs will have more possible random walks
    and a smaller graph might be more similar to a larger graph than to itself.
    Args:
        covariance_matrix: the covariance matrix to scale

    Returns: the normalised covariance matrix

    """

    normalisation_factor = torch.unsqueeze(torch.sqrt(torch.diagonal(covariance_matrix)), -1)
    normalisation_factor = normalisation_factor * torch.transpose(normalisation_factor, -2, -1)
    return torch.div(covariance_matrix, normalisation_factor)


def get_molecular_edge_labels(mol):
    """
    Extracts all unique atom - bond type - atom combinations from the specified
    rdkit molecule for downstream use as edge labels.

    Args:
        mol: rdkit molecule class

    Returns: list of [atom symbol, atom symbol, bond type] lists

    """

    edge_src = []
    edge_dst = []
    edge_label = []

    for idx, bond in enumerate(mol.GetBonds()):
        bond_edge_label = hash(frozenset([
            bond.GetBeginAtom().GetSymbol(),
            bond.GetEndAtom().GetSymbol(),
            bond.GetBondTypeAsDouble()
        ]))

        edge_src.append(bond.GetBeginAtomIdx())
        edge_dst.append(bond.GetEndAtomIdx())
        edge_label.append(bond_edge_label)

    return edge_src, edge_dst, edge_label


def get_sparse_adj_mat(index_tuples, shape_tuple):
    """
    Constructs an undirected, unweighted sparse adjacency matrix from a list of index tuples.

    Args:
        index_tuples: list of index tuples
        shape_tuple: specifies the shape of the resulting sparse matrix

    Returns: a symmetric sparse adjacency matrix

    """

    index_tuples = list(zip(*index_tuples))
    adj_mat = torch.sparse_coo_tensor(
        indices=torch.LongTensor([index_tuples[0] + index_tuples[1], index_tuples[1] + index_tuples[0]]),
        values=torch.ones(len(index_tuples[0]) + len(index_tuples[1])),
        size=shape_tuple
    )

    return adj_mat
