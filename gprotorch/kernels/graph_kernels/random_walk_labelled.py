"""
Contains GPyTorch random walk kernels on undirected graphs.
Notes: requires pytorch 1.8.0+ for torch.kron()
"""

import torch
import gpytorch
import numpy as np
from tqdm import tqdm
from gpytorch.kernels import Kernel
from gpytorch.constraints import Positive
from gprotorch.kernels.graph_kernels.graph_kernel_utils import normalise_covariance, get_label_adj_mats
from rdkit.Chem import MolFromSmiles
from line_profiler_pycharm import profile

from gpytorch import settings
settings.lazily_evaluate_kernels(False)


class RandomWalk(Kernel):
    """
    Calculates the random walk kernel introduced by Gärtner et al LTKM, 129-143, 2003
    on labelled graphs using the fixed-point iteration method introduced in section 4.3 of
    Vishwanathan et al JLMR, 11:1201–1242, 2010.
    """

    def __init__(self,
                 normalise=True,
                 uniform_probabilities=True,
                 num_fixed_point_iters=10,
                 weight_constraint=None,
                 weight_prior=None,
                 **kwargs):
        """
        Initialises the kernel class.

        Args:
            normalise: whether to normalise the graph kernels
            uniform_probabilities: whether the starting and stopping probabilities are uniform across nodes
            (as in Vishwanathan et al) or whether they are all set to 1 (as in Gärtner et al)
            (both are equivalent if normalisation is performed)
            num_fixed_point_iters: how many fixed-point iterations to perform
            weight_constraint: the constraint to put on the weight parameter,
            Positive() by default, but in some circumstances Interval might be better()
            weight_prior: prior of the weight parameter
            **kwargs: base kernel class arguments
        """

        super(RandomWalk, self).__init__(**kwargs)

        self.normalise = normalise
        self.uniform_probabilities = uniform_probabilities
        self.num_fixed_point_iters = num_fixed_point_iters

        self.register_parameter(
            name='raw_weight', parameter=torch.nn.Parameter(torch.zeros(1))
        )

        if weight_constraint is None:
            weight_constraint = Positive()

        self.register_constraint('raw_weight', weight_constraint)

        if weight_prior is not None:
            self.register_prior(
                'weight_prior',
                weight_prior,
                lambda m: m.weight,
                lambda m, v: m._set_weight(v)
            )

    @property
    def weight(self) -> torch.Tensor:
        return self.raw_weight_constraint.transform(self.raw_weight)

    @weight.setter
    def weight(self,
               value: torch.Tensor
               ) -> None:
        self._set_weight(value)

    def _set_weight(self,
                    value: torch.Tensor
                    ) -> None:
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_weight)
        self.initialize(raw_weight=self.raw_weight_constraint.inverse_transform(value))

    @profile
    def forward(self,
                x1: torch.Tensor,
                x2: torch.Tensor,
                **params) -> torch.Tensor:
        """
        Calculates the covariance matrix between x1 and x2.

        Args:
            x1: list of rdkit molecules
            x2: list of rdkit molecules
            **params:

        Returns: covariance matrix

        """

        # check if the matrices are identical (order-dependent)
        x1_equals_x2 = torch.equal(x1, x2)

        x1_size = int(np.sqrt(x1.shape[1]))
        if x1_equals_x2:
            x2_size = x1_size
        else:
            x2_size = int(np.sqrt(x1.shape[2]))

        rw_probs = torch.ones(x1_size * x2_size)

        # reshape adjacency matrices
        x1 = x1.view((x1.shape[0], x1_size, x1_size))

        if x1_equals_x2:
            x2 = x1
        else:
            x2 = x2.view((x2.shape[0], x2_size, x2_size))

        # derive nonzero elements
        nnz1 = [(torch.nonzero(x, as_tuple=False), x[torch.nonzero(x, as_tuple=True)]) for x in x1]
        unique1 = [torch.unique(z) for _, z in nnz1]
        if x1_equals_x2:
            nnz2 = nnz1
            unique2 = unique1
        else:
            nnz2 = [(torch.nonzero(x, as_tuple=False), x[torch.nonzero(x, as_tuple=True)]) for x in x2]
            unique2 = [torch.unique(z) for _, z in nnz2]

        covar_mat = torch.zeros([len(x1), len(x2)])

        for idx1 in tqdm(range(len(x1))):
            for idx2 in range(len(x2)):

                # if applicable, fill lower triangular part
                # with already calculated upper triangular part
                if x1_equals_x2 and idx1 == idx2:
                    covar_mat[idx1, idx2] = 1

                elif x1_equals_x2 and idx2 < idx1:
                    covar_mat[idx1, idx2] = covar_mat[idx2, idx1]

                else:

                    kron_i = []
                    kron_j = []

                    nnz_id1, nnz_val1 = nnz1[idx1]
                    nnz_id2, nnz_val2 = nnz2[idx1]

                    common_labels = np.intersect1d(unique1[idx1], unique2[idx2])

                    if common_labels.size == 0:
                        covar_mat[idx1, idx2] = 0
                        continue

                    for label in common_labels:
                        label_ids1 = nnz_id1[torch.where(nnz_val1 == label)]
                        label_ids2 = nnz_id2[torch.where(nnz_val2 == label)]
                        kron_i.append(
                            torch.cartesian_prod(label_ids1[:, 0] * x2_size, label_ids2[:, 0]).sum(1))
                        kron_j.append(torch.cartesian_prod(label_ids1[:, 1] * x2_size, label_ids2[:, 1]).sum(1))

                    kron_i = torch.cat(kron_i)
                    kron_j = torch.cat(kron_j)

                    # creating the Kronecker product matrix as a sparse
                    # tensor and converting it to dense is faster than
                    # directly instantiating or overwriting a dense one
                    kron_mat = torch.sparse_coo_tensor(
                        torch.vstack([kron_i, kron_j]),
                        torch.ones(len(kron_i)),
                        size=(kron_i.max()+1, kron_j.max()+1)
                    ).to_dense()

                    rw_probs_reduced = rw_probs[:kron_i.max()+1]
                    kron_mat = torch.diag(rw_probs_reduced) - self.weight * kron_mat

                    correct_solve = torch.linalg.solve(kron_mat, rw_probs_reduced)
                    covar_mat[idx1, idx2] = rw_probs_reduced.T @ correct_solve

        if self.normalise:
            covar_mat = normalise_covariance(covar_mat)

        return covar_mat


if __name__ == '__main__':
    from gprotorch.dataloader.mol_prop import DataLoaderMP

    dataloader = DataLoaderMP()
    dataloader.load_benchmark("FreeSolv", "../../../data/property_prediction/FreeSolv.csv")
    train_mols = [MolFromSmiles(mol) for mol in dataloader.features]
    train_y = torch.from_numpy(dataloader.labels)

    train_adj_mats = [get_label_adj_mats(x) for x in train_mols][:300]
    largest_mol = max(mol.GetNumAtoms() for mol in train_mols)
    train_labels = {label for mol in train_adj_mats for label in mol}
    train_labels = {label: i for i, label in enumerate(train_labels)}

    new_adj_mats = []

    for train_adj_mat in train_adj_mats:
        new_adj_mats_labels = []
        for label, adj_mat in train_adj_mat.items():
            new_adj_mat = torch.sparse_coo_tensor(
                indices=adj_mat.indices(),
                values=torch.full_like(adj_mat.values(), train_labels[label]),
                size=(largest_mol, largest_mol)
            )
            new_adj_mats_labels.append(new_adj_mat)

        if len(new_adj_mats_labels) == 1:
            new_adj_mats.append(new_adj_mats_labels[0].coalesce().to_dense())
        elif len(new_adj_mats_labels) > 1:
            new_adj_mat = new_adj_mats_labels[0]
            for adj_mat in new_adj_mats_labels[1:]:
                new_adj_mat += adj_mat
            new_adj_mats.append(new_adj_mat.coalesce().to_dense())
        else:
            pass

    new_adj_mats = torch.stack([adj_mat.flatten() for adj_mat in new_adj_mats])

    class ExactGPModel(gpytorch.models.ExactGP):
        def __init__(self, train_x, train_y, likelihood):
            super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
            self.mean_module = gpytorch.means.ConstantMean()
            self.covar_module = RandomWalk()

        def forward(self, x):
            mean_x = self.mean_module(x)
            settings.lazily_evaluate_kernels._set_state(False)
            covar_x = self.covar_module(x)
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = ExactGPModel(new_adj_mats, train_y, likelihood)

    model.train()
    likelihood.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    output = model(new_adj_mats)

