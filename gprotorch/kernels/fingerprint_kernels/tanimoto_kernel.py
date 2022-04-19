"""
Tanimoto Kernel. Operates on representations including bit vectors e.g. Morgan/ECFP6 fingerprints count vectors e.g.
RDKit fragment features.
"""

import gpytorch

from gprotorch.kernels.fingerprint_kernels.base_fingerprint_kernel import BitKernel


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

   .. note::

    This kernel does not have an `outputscale` parameter. To add a scaling parameter,
    decorate this kernel with a :class:`gpytorch.test_kernels.ScaleKernel`.

    Example:
        >>> x = torch.randint(0, 2, (10, 5))
        >>> # Non-batch: Simple option
        >>> covar_module = gpytorch.test_kernels.ScaleKernel(TanimotoKernel())
        >>> covar = covar_module(x)  # Output: LazyTensor of size (10 x 10)
        >>>
        >>> batch_x = torch.randint(0, 2, (2, 10, 5))
        >>> # Batch: Simple option
        >>> covar_module = gpytorch.test_kernels.ScaleKernel(TanimotoKernel())
        >>> covar = covar_module(batch_x)  # Output: LazyTensor of size (2 x 10 x 10)
    """
    def __init__(self, **kwargs):
        super(TanimotoKernel, self).__init__(**kwargs)
        self.metric = 'tanimoto'

    def forward(self, x1, x2, **params):
        return self.covar_dist(x1, x2, **params)