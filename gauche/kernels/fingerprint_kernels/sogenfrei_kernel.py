"""
Sogenfrei Kernel. Operates on representations including bit vectors e.g. Morgan/ECFP6 fingerprints count vectors e.g.
RDKit fragment features.
"""

import torch
from gpytorch.kernels import Kernel

tkwargs = {"dtype": torch.double}


def batch_sogenfrei_sim(
    x1: torch.Tensor, x2: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    """
    Sogenfrei similarity between two batched tensors, across last 2 dimensions.
    eps argument ensures numerical stability if all zero tensors are added.

    <x1, x2>**2 / (|x1| + |x2|)

    Where <.> is the inner product and || is the L1 norm

    Args:
        x1: `[b x n x d]` Tensor where b is the batch dimension
        x2: `[b x m x d]` Tensor
        eps: Float for numerical stability. Default value is 1e-6
    Returns:
        Tensor denoting the Sogenfrei similarity.
    """

    if x1.ndim < 2 or x2.ndim < 2:
        raise ValueError("Tensors must have a batch dimension")

    # Compute L1 norm
    x1_norm = torch.sum(x1, dim=-1, keepdims=True)
    x2_norm = torch.sum(x2, dim=-1, keepdims=True)
    dot_prod = torch.matmul(x1, torch.transpose(x2, -1, -2))

    similarity = (dot_prod + eps) ** 2 / (
        x1_norm + torch.transpose(x2_norm, -1, -2) + eps
    )

    return similarity.to(**tkwargs).clamp_min_(
        0
    )  # zero out negative values for numerical stability


class SogenfreiKernel(Kernel):
    r"""
     Computes a covariance matrix based on the Sogenfrei kernel
     between inputs :math:`\mathbf{x_1}` and :math:`\mathbf{x_2}`:

    .. note::

     This kernel does not have an `outputscale` parameter. To add a scaling parameter,
     decorate this kernel with a :class:`gpytorch.test_kernels.ScaleKernel`.

     Example:
         >>> x = torch.randint(0, 2, (10, 5))
         >>> # Non-batch: Simple option
         >>> covar_module = gpytorch.kernels.ScaleKernel(SogenfreiKernel())
         >>> covar = covar_module(x)  # Output: LazyTensor of size (10 x 10)
         >>>
         >>> batch_x = torch.randint(0, 2, (2, 10, 5))
         >>> # Batch: Simple option
         >>> covar_module = gpytorch.kernels.ScaleKernel(SogenfreiKernel())
         >>> covar = covar_module(batch_x)  # Output: LazyTensor of size (2 x 10 x 10)
    """

    is_stationary = False
    has_lengthscale = False

    def __init__(self, **kwargs):
        super(SogenfreiKernel, self).__init__(**kwargs)

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

        return batch_sogenfrei_sim(x1, x2)
