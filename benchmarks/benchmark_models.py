"""
GP Model definitions to be used in the benchmarks
"""

from gauche.kernels.fingerprint_kernels.tanimoto_kernel import (
    TanimotoKernel,
)
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import LinearKernel, ScaleKernel
from gpytorch.means import ConstantMean
from gpytorch.models import ExactGP


class TanimotoGP(ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(TanimotoGP, self).__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean()
        # We use the Tanimoto kernel to work with molecular fingerprint representations
        self.covar_module = ScaleKernel(TanimotoKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)


class ScalarProductGP(ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ScalarProductGP, self).__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean()
        # We use the scalar product kernel on molecular fingerprint representations
        self.covar_module = ScaleKernel(LinearKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)
