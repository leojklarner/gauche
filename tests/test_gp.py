
import sys
sys.path.append('../')

import unittest
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from gprotorch.dataloader import DataLoaderMP
from gprotorch.dataloader.data_utils import transform_data

import torch
import gpytorch
from gpytorch.models import ExactGP
from botorch import fit_gpytorch_model

from gprotorch import SIGP, Inputs
from gprotorch.kernels.fingerprint_kernels.tanimoto_kernel import TanimotoKernel

class SIGPTestClass(SIGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(TanimotoKernel())

    def forward(self, x):
        x = torch.cat(x.data, axis=0)
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class GPyTorchBenchmark(ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(TanimotoKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class TestReproducibility(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42); np.random.seed(42)
        loader = DataLoaderMP()
        loader.load_benchmark("Photoswitch", "../data/property_prediction/photoswitches.csv")

        loader.featurize('fragprints')
        X = loader.features
        y = loader.labels

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        _, y_train, _, y_test, y_scaler = transform_data(X_train, y_train, X_test, y_test)

        self.y_scaler = y_scaler
        self.X_train = torch.tensor(X_train.astype(np.float64))
        self.X_test = torch.tensor(X_test.astype(np.float64))
        self.y_train = torch.tensor(y_train).flatten()
        self.y_test = torch.tensor(y_test).flatten()

    def _run_model(self, model, likelihood, X_test):
        model.train()
        likelihood.train()

        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
        fit_gpytorch_model(mll)    

        model.eval()
        likelihood.eval()

        y_pred = self.y_scaler.inverse_transform(model(X_test).mean.detach().unsqueeze(dim=1))
        y_test = self.y_scaler.inverse_transform(self.y_test.unsqueeze(dim=1))

        return np.sqrt(mean_squared_error(y_test, y_pred))

    def test_sigp(self):

        # benchmark
        torch.manual_seed(42); np.random.seed(42)

        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = GPyTorchBenchmark(self.X_train, self.y_train, likelihood)
        benchmark_rmse = self._run_model(model, likelihood, self.X_test)

        # our run
        torch.manual_seed(42); np.random.seed(42)

        X_train = Inputs([self.X_train[[i], ] for i in range(len(self.X_train))])
        X_test = Inputs([self.X_test[[i], ] for i in range(len(self.X_test))])

        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = SIGPTestClass(X_train, self.y_train, likelihood)
        sigp_rmse = self._run_model(model, likelihood, X_test)

        assert sigp_rmse == benchmark_rmse

if __name__ == '__main__':
    unittest.main()
