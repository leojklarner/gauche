"""
Tanimoto Kernel. Operates on discrete or real-valued representations including bit vectors
(e.g. Morgan/ECFP6 fingerprints) and count vectors (e.g. RDKit fragment features).
"""

import gpytorch
import torch
from gauche.kernels.fingerprint_kernels.base_fingerprint_kernel import (
    BitKernel,
)


class TanimotoKernel(BitKernel):
    r"""
     Computes a covariance matrix based on the Tanimoto kernel
     between inputs :math:`\mathbf{x_1}` and :math:`\mathbf{x_2}`:

     .. math::

    \begin{equation*}
     k_{\text{Tanimoto}}(\mathbf{x}, \mathbf{x'}) = \frac{\langle\mathbf{x},
     \mathbf{x'}\rangle}{\left\lVert\mathbf{x}\right\rVert^2 + \left\lVert\mathbf{x'}\right\rVert^2 -
     \langle\mathbf{x}, \mathbf{x'}\rangle}
    \end{equation*}

    This kernel is positive-definite for all real-valued inputs
    (see https://arxiv.org/abs/2306.14809).

    .. note::

     This kernel does not have an `outputscale` parameter. To add a scaling parameter,
     decorate this kernel with a :class:`gpytorch.test_kernels.ScaleKernel`.

     Example:
         >>> x = torch.randint(0, 2, (10, 5))
         >>> # Non-batch: Simple option
         >>> covar_module = gpytorch.kernels.ScaleKernel(TanimotoKernel())
         >>> covar = covar_module(x)  # Output: LazyTensor of size (10 x 10)
         >>>
         >>> batch_x = torch.randint(0, 2, (2, 10, 5))
         >>> # Batch: Simple option
         >>> covar_module = gpytorch.kernels.ScaleKernel(TanimotoKernel())
         >>> covar = covar_module(batch_x)  # Output: LazyTensor of size (2 x 10 x 10)
    """

    is_stationary = False
    has_lengthscale = False

    def __init__(self, **kwargs):
        super(TanimotoKernel, self).__init__(**kwargs)
        self.metric = "tanimoto"

    def forward(self, x1, x2, diag=False, **params):
        if diag:
            assert x1.size() == x2.size() and torch.equal(x1, x2)
            return torch.ones(
                *x1.shape[:-2], x1.shape[-2], dtype=x1.dtype, device=x1.device
            )
        else:
            return self.covar_dist(x1, x2, **params)


class MinMaxKernel(TanimotoKernel):
    r"""
    A continuous extension to the Tanimoto kernel on binary fingerprints.

     .. math::

    \begin{equation*}
     k_{\text{MinMax}}(\mathbf{x}, \mathbf{x'}) = \frac{\sum_i \min(x_i, x'_i)}{\sum_i \max(x_i, x'_i)}
    \end{equation*}

    ..
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.metric = "minmax"
