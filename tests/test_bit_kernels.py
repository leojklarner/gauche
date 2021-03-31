"""
Test suite for bit vector kernels.
Author: Ryan-Rhys Griffiths 2021
"""

import pytest
import torch

from gprotorch.kernels.bit_vector_kernel import BitDistance


@pytest.mark.parametrize("x1, x2", [
    (torch.ones((2, 2)), torch.ones((2, 2))),
    (2*torch.ones((2, 2)), 2*torch.ones((2, 2))),
    (10*torch.ones((2, 2)), 10*torch.ones((2, 2))),
    (10000*torch.ones((2, 2)), 10000*torch.ones((2, 2))),
    (0.1*torch.ones((2, 2)), 0.1*torch.ones((2, 2)))
])
def test_tanimoto_similarity_with_equal_inputs(x1, x2):
    """
    Test the Tanimoto similarity metric between two equal input vectors.
    """
    dist_object = BitDistance()
    tan_similarity = dist_object._tan_similarity(x1, x2, postprocess=False, x1_eq_x2=True)

    assert (tan_similarity == torch.ones((2, 2))).all() == True


def test_tanimoto_similarity_with_unequal_inputs():
    """
    Test the Tanimoto similarity metric between two unequal input vectors.
    """
    x1 = torch.ones((2, 2))
    x2 = 2*torch.ones((2, 2))
    dist_object = BitDistance()
    tan_similarity = dist_object._tan_similarity(x1, x2, postprocess=False)

    assert torch.allclose(tan_similarity, torch.tensor(0.6666666666666))


# GPflow Tanimoto kernel implementation below

# class Tanimoto(gpflow.kernels.Kernel):
#     def __init__(self):
#         super().__init__()
#         self.variance = gpflow.Parameter(1.0, transform=positive())
#
#     def K(self, X, X2=None):
#         """
#         Compute the Tanimoto kernel matrix σ² * ((<x, y>) / (||x||^2 + ||y||^2 - <x, y>))
#         :param X: N x D array
#         :param X2: M x D array. If None, compute the N x N kernel matrix for X.
#         :return: The kernel matrix of dimension N x M
#         """
#         if X2 is None:
#             X2 = X
#
#         Xs = tf.reduce_sum(tf.square(X), axis=-1)  # Squared L2-norm of X
#         X2s = tf.reduce_sum(tf.square(X2), axis=-1)  # Squared L2-norm of X2
#         cross_product = tf.tensordot(X, X2, [[-1], [-1]])  # outer product of the matrices X and X2
#
#         # Analogue of denominator in Tanimoto formula
#
#         denominator = -cross_product + broadcasting_elementwise(tf.add, Xs, X2s)
#
#         return self.variance * cross_product / denominator
#
#     def K_diag(self, X):
#         """
#         Compute the diagonal of the N x N kernel matrix of X
#         :param X: N x D array
#         :return: N x 1 array
#         """
#         return tf.fill(tf.shape(X)[:-1], tf.squeeze(self.variance))