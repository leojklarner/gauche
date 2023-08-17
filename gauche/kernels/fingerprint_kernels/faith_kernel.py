"""
Faith Kernel. Operates on representations including bit vectors e.g. Morgan/ECFP6 fingerprints count vectors e.g.
RDKit fragment features.
"""

import torch
from gpytorch.kernels import Kernel

tkwargs = {"dtype": torch.double}


def batch_faith_sim(
        x1: torch.Tensor, x2: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    """
    Faith similarity between two batched tensors, across last 2 dimensions.
    eps argument ensures numerical stability if all zero tensors are added.

    (2 * <x1, x2>) + d / 2n

    Where <.> is the inner product, d is the number of common zeros and n is the dimension of the input vectors

    Args:
        x1: `[b x n x d]` Tensor where b is the batch dimension
        x2: `[b x m x d]` Tensor
        eps: Float for numerical stability. Default value is 1e-6
    Returns:
        Tensor denoting the Faith similarity.
    """

    if x1.ndim < 2 or x2.ndim < 2:
        raise ValueError("Tensors must have a batch dimension")

    n = x1.shape[-1]
    d = torch.sum((x1[-1] == 0) & (x2[-1] == 0), dim=-1, keepdims=True)
    dot_prod = torch.matmul(x1, torch.transpose(x2, -1, -2))

    similarity = (2 * dot_prod + d + eps) / (2 * n + eps)

    return similarity.clamp_min_(0)  # zero out negative values for numerical stability


class FaithKernel(Kernel):
    r"""
     Computes a covariance matrix based on the Faith kernel
     between inputs :math:`\mathbf{x_1}` and :math:`\mathbf{x_2}`:

    .. note::

     This kernel does not have an `outputscale` parameter. To add a scaling parameter,
     decorate this kernel with a :class:`gpytorch.test_kernels.ScaleKernel`.

     Example:
         >>> x = torch.randint(0, 2, (10, 5))
         >>> # Non-batch: Simple option
         >>> covar_module = gpytorch.kernels.ScaleKernel(FaithKernel())
         >>> covar = covar_module(x)  # Output: LazyTensor of size (10 x 10)
         >>>
         >>> batch_x = torch.randint(0, 2, (2, 10, 5))
         >>> # Batch: Simple option
         >>> covar_module = gpytorch.kernels.ScaleKernel(FaithKernel())
         >>> covar = covar_module(batch_x)  # Output: LazyTensor of size (2 x 10 x 10)
    """

    is_stationary = False
    has_lengthscale = False

    def __init__(self, **kwargs):
        super(FaithKernel, self).__init__(**kwargs)

    def forward(self, x1, x2, diag=False, **params):
        if diag:
            assert x1.size() == x2.size() and torch.equal(x1, x2)
            return torch.ones(
                *x1.shape[:-2], x1.shape[-2], dtype=x1.dtype, device=x1.device
            )
        else:
            return self.covar_dist(x1, x2, **params)

    def covar_dist(
            self,
            x1,
            x2,
            last_dim_is_batch=False,
            **params,
    ):
        r"""This is a helper method for computing the bit vector similarity between
        all pairs of points in x1 and x2.

        Args:
            :attr:`x1` (Tensor `n x d` or `b1 x ... x bk x n x d`):
                First set of data.
            :attr:`x2` (Tensor `m x d` or `b1 x ... x bk x m x d`):
                Second set of data.
            :attr:`last_dim_is_batch` (tuple, optional):
                Is the last dimension of the data a batch dimension or not?

        Returns:
            (:class:`Tensor`, :class:`Tensor) corresponding to the distance matrix between `x1` and `x2`.
            The shape depends on the kernel's mode
            * `diag=False`
            * `diag=False` and `last_dim_is_batch=True`: (`b x d x n x n`)
            * `diag=True`
            * `diag=True` and `last_dim_is_batch=True`: (`b x d x n`)
        """
        if last_dim_is_batch:
            x1 = x1.transpose(-1, -2).unsqueeze(-1)
            x2 = x2.transpose(-1, -2).unsqueeze(-1)

        return batch_faith_sim(x1, x2)
