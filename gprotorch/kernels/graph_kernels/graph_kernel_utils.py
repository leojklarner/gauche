"""
Contains re-usable utility functions for calculating graph kernels.
"""

import torch


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
