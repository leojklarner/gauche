# Author: Leo Klarner
"""
Molecule kernels for Gaussian Process Regression implemented in PyTorch.
"""

import gpytorch
import torch
from gpytorch.constraints import Positive
from gpytorch.kernels import Kernel
from gpytorch.lazy import MatmulLazyTensor, RootLazyTensor


class Tanimoto(Kernel):
    """
    An implementation of the Tanimoto kernel (aka. Jaccard similarity)
    as a gpytorch kernel
    """

    def __init__(self, variance_constraint=None, **kwargs):
        super().__init__(**kwargs)
        # put positivity constraint on variance as default
        if variance_constraint is None:
            variance_constraint = Positive()
        # initialise variance parameter, potentially different for each batch
        self.register_parameter(
            name="raw_variance",
            parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1)),
        )
        # apply variance constraint after parameter is initialised
        self.register_constraint("raw_variance", variance_constraint)

    # methods for directly applying the variance constraint
    # when setting the parameter

    @property
    def variance(self):
        return self.raw_variance_constraint.transform(self.raw_variance)

    @variance.setter
    def variance(self, value):
        self._set_variance(value)

    def _set_variance(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_variance)
        self.initialize(
            raw_variance=self.raw_variance_constraint.inverse_transform(value)
        )

    def forward(self, x1, x2, diag=False, **params):
        """
        Compute the Tanimoto kernel matrix σ² * ((<x, y>) / ((||x||2)^2 + (||y||2)^2 - <x, y>))

        :param x1: N x D (or B x N x D) (batched) data tensor
        :param x2: M x D (or B x M x D) (batched) data tensor
        :param diag: whether to return only the diagonal of the covariance matrix
        :return: The kernel/covariance matrix of dimension N x M
        """

        # calculate the squared L2-norm over the feature dimension of the x1 data tensor
        x1_norm = torch.unsqueeze(torch.sum(torch.square(x1), dim=-1), dim=-1)

        # check if both data tensors are identical (i.e. whether called for training or testing)
        if x1.size() == x2.size() and torch.equal(x1, x2):
            x2_norm = x1_norm

            # calculate the symmetric matrix product of identical data tensors
            mat_prod = RootLazyTensor(x1)

        else:
            # if both data tensors are not identical
            # calculate the squared L2-norm over the feature dimension of the x2 data tensor
            x2_norm = torch.unsqueeze(
                torch.sum(torch.square(x2), dim=-1), dim=-1
            )

            # calculate the asymmetric matrix product of the different data tensors
            mat_prod = MatmulLazyTensor(x1, torch.transpose(x2, -2, -1))

        # calculate the matrix product without using lazy tensors
        # mat_prod = torch.matmul(x1, torch.transpose(x2, -2, -1))

        # convert the lazy tensors back to normal tensors
        # as subtraction and division are not implemented in the standard
        # LazyTensor class; could be added if performance critical
        mat_prod = gpytorch.lazy.LazyTensor.evaluate(mat_prod)

        # calculate the Tanimoto similarity
        denominator = torch.sub(
            torch.add(x1_norm, torch.transpose(x2_norm, -2, -1)), mat_prod
        )
        result = self.variance * torch.div(mat_prod, denominator)

        if diag:
            return torch.diag(result)
        else:
            return result
