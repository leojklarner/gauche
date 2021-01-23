"""
Simple tests to check for the correct implementation
of the kernels
"""

import numpy.testing as npt
import pytest
import torch


@pytest.mark.parametrize(
    "x1, x2, result",
    [
        (
            torch.tensor(
                [
                    [0, 1, 0, 1, 0, 1],
                    [1, 1, 1, 1, 1, 1],
                    [1, 0, 0, 0, 1, 1],
                    [1, 0, 1, 0, 0, 1],
                ]
            ),
            torch.tensor(
                [
                    [0, 1, 0, 1, 0, 1],
                    [1, 1, 1, 1, 1, 1],
                    [1, 0, 0, 0, 1, 1],
                    [1, 0, 1, 0, 0, 1],
                ]
            ),
            torch.tensor(
                [
                    [2.0000, 1.0000, 0.4000, 0.4000],
                    [1.0000, 2.0000, 1.0000, 1.0000],
                    [0.4000, 1.0000, 2.0000, 1.0000],
                    [0.4000, 1.0000, 1.0000, 2.0000],
                ]
            ),
        ),
        (
            torch.tensor(
                [
                    [0, 1, 0, 1, 0, 1],
                    [1, 1, 1, 1, 1, 1],
                    [1, 0, 0, 0, 1, 1],
                    [1, 0, 1, 0, 0, 1],
                ]
            ),
            torch.tensor(
                [[0, 1, 0, 1, 0, 1], [1, 1, 1, 1, 1, 1], [1, 0, 0, 0, 1, 1]]
            ),
            torch.tensor(
                [
                    [2.0000, 1.0000, 0.4000],
                    [1.0000, 2.0000, 1.0000],
                    [0.4000, 1.0000, 2.0000],
                    [0.4000, 1.0000, 1.0000],
                ]
            ),
        ),
        (
            torch.tensor(
                [[1, 1, 1, 1, 1, 1], [1, 0, 0, 0, 1, 1], [1, 0, 1, 0, 0, 1]]
            ),
            torch.tensor(
                [
                    [0, 1, 0, 1, 0, 1],
                    [1, 1, 1, 1, 1, 1],
                    [1, 0, 0, 0, 1, 1],
                    [1, 0, 1, 0, 0, 1],
                ]
            ),
            torch.tensor(
                [
                    [1.0000, 2.0000, 1.0000, 1.0000],
                    [0.4000, 1.0000, 2.0000, 1.0000],
                    [0.4000, 1.0000, 1.0000, 2.0000],
                ]
            ),
        ),
    ],
)
def test_tanimoto_kernel_symmetric(x1, x2, result):
    """
    Tests whether the torch implementation of the Tanimoto kernel
    gives the correct results for simple test cases
    """
    from gprotorch.kernels.tanimoto import Tanimoto

    test = Tanimoto()
    test.variance = 2

    output = test.forward(x1, x2)

    npt.assert_almost_equal(
        output.detach().numpy(), result.detach().numpy(), decimal=4
    )
