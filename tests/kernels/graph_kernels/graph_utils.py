import torch
from gprotorch.kernels.graph_kernels.graph_kernel_utils import kronecker_inner_product


def test_kronecker_inner_product():
    """
    This functions checks that the Kronecker inner product
    gives the correct result for adjacency matrices
    with discrete labels.

    Returns: None

    """

    a = torch.Tensor(
        [
            [[0, 0], [0, 1], [0, 0]],
            [[0, 1], [0, 0], [1, 0]],
            [[0, 0], [1, 0], [0, 0]],
        ]
    )

    b = torch.Tensor(
        [
            [[0, 0], [1, 0]],
            [[1, 0], [0, 0]],
        ]
    )

    c = torch.Tensor(
        [
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0, 0],
        ]
    )

    assert torch.equal(kronecker_inner_product(a, b), c)


if __name__ == '__main__':
    pass
