
import sys
sys.path.append('../')

import unittest
import numpy as np
from functools import lru_cache
from rdkit.Chem import MolFromSmiles
from grakel import Graph as GraKelGraph
from grakel.kernels import WeisfeilerLehman
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from gauche.dataloader import DataLoaderMP
from gauche.dataloader.data_utils import transform_data

import torch
import gpytorch
from gpytorch.models import ExactGP
from botorch import fit_gpytorch_model

from gauche.gp import SIGP, GraphKernel, Inputs
from gauche.kernels.fingerprint_kernels.tanimoto_kernel import TanimotoKernel

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

# ---------------------------------------------------------------------------

class WLKernel(GraphKernel):
    def __init__(self):
        super().__init__(graph_kernel=WeisfeilerLehman())

class GraphGP(SIGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean = gpytorch.means.ConstantMean()
        self.covariance = WLKernel()

    def forward(self, x):
        mean = self.mean(torch.zeros(len(x.data), 1)).float()
        covariance = self.covariance(x) + torch.eye(len(x.data))*1e-3
        return gpytorch.distributions.MultivariateNormal(mean, covariance)

class TestGraphKernel(unittest.TestCase):
    def setUp(self):
        loader = DataLoaderMP()
        loader.load_benchmark("Photoswitch", "../data/property_prediction/photoswitches.csv")
        bond_types = {1.0: 'S', 1.5: 'A', 2.0: 'D', 3.0: 'O'}

        def to_graph(mol):
            ''' from leo's branch, modified '''
            node_labels = {i: mol.GetAtomWithIdx(i).GetSymbol() for i in range(mol.GetNumAtoms())}
            edges = {}
            for bond in mol.GetBonds():
                start_idx = bond.GetBeginAtomIdx()
                end_idx = bond.GetEndAtomIdx()
                bond_type = bond.GetBondTypeAsDouble()

                edges[(start_idx, end_idx)] = bond_types[bond_type]
                edges[(end_idx, start_idx)] = bond_types[bond_type]
            edge_list = list(edges.keys())
            assert len(edge_list) == len(set(edge_list))

            graph = GraKelGraph(edge_list,
                node_labels=node_labels,
                edge_labels=edges,
                graph_format='adjacency')
            return graph

        X = [to_graph(MolFromSmiles(mol)) for mol in loader.features]
        y = loader.labels

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        _, y_train, _, y_test, y_scaler = transform_data(
            np.zeros_like(y_train), y_train, np.zeros_like(y_test), y_test)

        self.X_train = Inputs(X_train)
        self.X_test = Inputs(X_test)
        self.y_train = torch.tensor(y_train).flatten().float()
        self.y_test = torch.tensor(y_test).flatten().float()
        self.y_scaler = y_scaler

    def test_graph_gp(self):
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = GraphGP(self.X_train, self.y_train, likelihood)

        model.train(); likelihood.train()
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

        for i in range(100):
            optimizer.zero_grad()
            output = model(self.X_train)
            loss = -mll(output, self.y_train)
            loss.backward()
            optimizer.step()

        # fit_gpytorch_model(mll)  # <- this method doesn't work as autodiscovery seems to fail
        model.eval(); likelihood.eval()

        y_pred = self.y_scaler.inverse_transform(model(self.X_test).mean.detach().unsqueeze(dim=1))
        y_test = self.y_scaler.inverse_transform(self.y_test.unsqueeze(dim=1))

        assert np.sqrt(mean_squared_error(y_test, y_pred)) < 21

if __name__ == '__main__':
    unittest.main()
