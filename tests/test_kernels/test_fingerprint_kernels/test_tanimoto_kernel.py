"""
Test suite for Tanimoto kernel.
"""

import gpflow
import pytest
import tensorflow as tf
import torch
from gpflow.utilities import positive
from gpflow.utilities.ops import broadcasting_elementwise
from gauche.kernels.fingerprint_kernels.tanimoto_kernel import (
    TanimotoKernel, batch_tanimoto_sim
)
from gpytorch.kernels import ScaleKernel


@pytest.mark.parametrize(
    "x1, x2",
    [
        (torch.ones((2, 2)), torch.ones((2, 2))),
        (2 * torch.ones((2, 2)), 2 * torch.ones((2, 2))),
        (10 * torch.ones((2, 2)), 10 * torch.ones((2, 2))),
        (10000 * torch.ones((2, 2)), 10000 * torch.ones((2, 2))),
        (0.1 * torch.ones((2, 2)), 0.1 * torch.ones((2, 2))),
    ],
)
def test_tanimoto_similarity_with_equal_inputs(x1, x2):
    """Test the Tanimoto similarity metric between two equal input tensors.
    """
    tan_similarity = batch_tanimoto_sim(x1, x2)
    assert torch.isclose(tan_similarity, torch.ones((2, 2))).all()


def test_tanimoto_similarity_with_unequal_inputs():
    """
    Test the Tanimoto similarity metric between two unequal input tensors.
    [1, 1]   and   [2, 2] are x1 and x2 respectively yielding output [2/3, 2/3]
    [1, 1]         [2, 2]                                            [2/3, 2/3]
    """
    x1 = torch.ones((2, 2))
    x2 = 2 * torch.ones((2, 2))
    tan_similarity = batch_tanimoto_sim(x1, x2)

    assert torch.allclose(tan_similarity, torch.tensor(0.6666666666666))


def test_tanimoto_similarity_with_very_unequal_inputs():
    """
    Test the Tanimoto similarity metric between two tensors that are unequal in every dimension.
    """

    x1 = torch.tensor([[1, 3], [2, 4]], dtype=torch.float64)
    x2 = torch.tensor([[4, 2], [3, 1]], dtype=torch.float64)
    tan_similarity = batch_tanimoto_sim(x1, x2)

    assert torch.allclose(
        tan_similarity,
        torch.tensor([[0.5, 6 / 14], [2 / 3, 0.5]], dtype=torch.float64),
    )


def test_tanimoto_similarity_with_batch_dimension():
    """
    Test the Tanimoto similarity metric between two tensors that are unequal and have a batch dimension.
    """

    x1 = torch.tensor([[1, 3], [2, 4]], dtype=torch.float64)
    x2 = torch.tensor([[4, 2], [3, 1]], dtype=torch.float64)

    # Add a batch dimension
    x1 = x1[None, :]
    x2 = x2[None, :]

    tan_similarity = batch_tanimoto_sim(x1, x2)

    assert torch.allclose(
        tan_similarity,
        torch.tensor([[0.5, 6 / 14], [2 / 3, 0.5]], dtype=torch.float64),
    )


def test_tanimoto_kernel():
    """
    Test the Tanimoto kernel when integrated with GP.
    """

    x = torch.randint(0, 2, (10, 5))
    # Non-batch: Simple option
    covar_module = ScaleKernel(TanimotoKernel())
    covar = covar_module(x)  # Output: LazyTensor of size (10 x 10)
    assert covar.size() == torch.Size([10, 10])
    batch_x = torch.randint(0, 2, (2, 10, 5))
    # Batch: Simple option
    covar_module = ScaleKernel(TanimotoKernel())
    covar = covar_module(batch_x)  # Output: LazyTensor of size (2 x 10 x 10)
    assert covar.size() == torch.Size([2, 10, 10])


def test_against_tf_implementation():
    """
    Tests the GPyTorch Tanimoto kernel against the TensorFlow implementation
    """
    torch.set_printoptions(precision=8)
    torch.manual_seed(0)
    x = torch.randint(0, 2, (100, 1000), dtype=torch.float64)

    covar_module = TanimotoKernel()
    covar = covar_module(x)
    torch_res = covar.evaluate()

    tf_x = x.numpy()
    tf_x = tf.convert_to_tensor(tf_x, dtype=tf.float64)

    tf_covar_module = Tanimoto()
    tf_covar = tf_covar_module.K(tf_x, tf_x)
    print(tf_covar)

    assert torch.allclose(torch.tensor(tf_covar.numpy()), torch_res)


# GPflow Tanimoto kernel implementation below used for testing.


class Tanimoto(gpflow.kernels.Kernel):
    def __init__(self):
        super().__init__()
        self.variance = gpflow.Parameter(1.0, transform=positive())

    def K(self, X, X2=None):
        """
        Compute the Tanimoto kernel matrix σ² * ((<x, y>) / (||x||^2 + ||y||^2 - <x, y>))
        :param X: N x D array
        :param X2: M x D array. If None, compute the N x N kernel matrix for X.
        :return: The kernel matrix of dimension N x M
        """
        if X2 is None:
            X2 = X

        Xs = tf.reduce_sum(tf.square(X), axis=-1)  # Squared L2-norm of X
        X2s = tf.reduce_sum(tf.square(X2), axis=-1)  # Squared L2-norm of X2
        cross_product = tf.tensordot(
            X, X2, [[-1], [-1]]
        )  # outer product of the matrices X and X2

        # Analogue of denominator in Tanimoto formula

        denominator = -cross_product + broadcasting_elementwise(
            tf.add, Xs, X2s
        )

        return self.variance * cross_product / denominator

    def K_diag(self, X):
        """
        Compute the diagonal of the N x N kernel matrix of X
        :param X: N x D array
        :return: N x 1 array
        """
        return tf.fill(tf.shape(X)[:-1], tf.squeeze(self.variance))
