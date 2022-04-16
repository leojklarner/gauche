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
from gprotorch.kernels.graph_kernels.graph_kernel_utils import normalise_covariance, get_label_adj_mats, kronecker_inner_product
from rdkit.Chem import MolFromSmiles
from gpytorch import settings


class RandomWalk(Kernel):
    """
    Calculates the random walk kernel introduced by Gärtner et al LTKM, 129-143, 2003
    on labelled graphs using the fixed-point iteration method introduced in section 4.3 of
    Vishwanathan et al JLMR, 11:1201–1242, 2010.
    """

    def __init__(self,
                 label_dimension,
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

        self.label_dimension = label_dimension

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

    def _set_weight(self, value: torch.Tensor) -> None:
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_weight)
        self.initialize(raw_weight=self.raw_weight_constraint.inverse_transform(value))

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

        # check if the matrices are identical
        x1_equals_x2 = torch.equal(x1, x2)

        # Step 1: pre-process flattened adjacency matrices

        # get the dimensions of the padded adjacency matrices
        # and check that they are identical, this is irrelevant
        # for this kernel but may lead to issues with GPyTorch
        x1_padding_dim = int(np.sqrt(x1.shape[1] / self.label_dimension))
        if not x1_equals_x2:
            x2_padding_dim = int(np.sqrt(x2.shape[1] / self.label_dimension))
            assert x1_padding_dim == x2_padding_dim, 'Dimension mismatch of padded adjacency matrices.'

        # unflatten the flattened adjacency matrices and
        # remove padding added in preprocessing
        x1 = [x.view(x1_padding_dim, x1_padding_dim, self.label_dimension) for x in x1]
        if x1_equals_x2:
            x2 = x1
        else:
            x2 = [x.view(x2_padding_dim, x2_padding_dim, self.label_dimension) for x in x2]

        x1_node_num = [x.nonzero().max(0).values[0] for x in x1]
        x1 = [x[:node_num, :node_num, :] for x, node_num in zip(x1, x1_node_num)]

        if x1_equals_x2:
            x2_node_num = x1_node_num
            x2 = x1
        else:
            x2_node_num = [x.nonzero().max(0).values[0] for x in x2]
            x2 = [x[:node_num, :node_num, :] for x, node_num in zip(x2, x2_node_num)]

        # initialise random walk start and stop probabilities,
        # to avoid re-initialisation in loop
        rw_probs = torch.ones(max(x1_node_num) * max(x2_node_num), device=x1[0].device)

        # Step 2: fill kernel matrix for all (x1, x2) pairs

        covar_mat = torch.zeros([len(x1), len(x2)])

        for idx1 in tqdm(range(len(x1))):
            for idx2 in range(len(x2)):

                if x1_equals_x2 and idx2 < idx1:
                    covar_mat[idx1, idx2] = covar_mat[idx2, idx1]

                else:

                    # calculate the Kronecker inner product
                    # formulating this as an einsum function
                    # is much faster than alternatives, such as
                    # pre-splitting the matrices by labels
                    kron_mat = kronecker_inner_product(x1[idx1], x2[idx2])

                    # adjust the shape of the start and stop probability
                    # vector and calculate the geometric sum expression
                    rw_probs_reduced = rw_probs[:x1_node_num[idx1]*x2_node_num[idx2]]
                    kron_mat = torch.diag(rw_probs_reduced) - self.weight * kron_mat

                    # calculate the random walk kernel
                    correct_solve = torch.linalg.solve(kron_mat, rw_probs_reduced)
                    covar_mat[idx1, idx2] = correct_solve.sum()

        # normalise the covariance matrix
        covar_mat = normalise_covariance(covar_mat)

        return covar_mat


if __name__ == '__main__':

    from gprotorch.dataloader.mol_prop import DataLoaderMP

    dataloader = DataLoaderMP()
    dataloader.load_benchmark("FreeSolv", "../../../data/property_prediction/FreeSolv.csv")
    train_mols = [MolFromSmiles(mol) for mol in dataloader.features]
    train_y = torch.from_numpy(dataloader.labels).float()

    train_adj_mats = [get_label_adj_mats(x, 'torch_sparse') for x in train_mols]

    new_adj_mats = []
    y_inds = []

    for i, adj_mat in enumerate(train_adj_mats):
        if adj_mat:
            indices = []
            labels = []
            for label, mat in adj_mat.items():
                indices.append(mat.indices())
                labels.append(torch.full([mat.indices().shape[1]], train_labels[label]))
            indices = torch.concat(indices, 1)
            labels = torch.concat(labels)
            new_adj_mats.append(torch.sparse_coo_tensor(
                indices=torch.vstack([indices, labels]),
                values=torch.ones_like(labels).float(),
                size=(largest_mol, largest_mol, len(train_labels))
            ))
            y_inds.append(i)

    new_adj_mats = torch.stack([mat.to_dense().view(largest_mol*largest_mol*len(train_labels)) for mat in new_adj_mats])
    train_y = train_y[y_inds]

    class ExactGPModel(gpytorch.models.ExactGP):
        def __init__(self, train_x, train_y, likelihood):
            super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
            self.covar_module = RandomWalk(
                label_dimension=len(train_labels),
                weight_constraint=gpytorch.constraints.Interval(0, 1/16)
            )

        def forward(self, x):
            mean_x = torch.zeros(x.shape[0], dtype=x.dtype, device=x.device)
            #settings.lazily_evaluate_kernels._set_state(False)
            covar_x = self.covar_module(x)
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = ExactGPModel(new_adj_mats, train_y, likelihood)

    model.train()
    likelihood.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for i in range(100):
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = model(new_adj_mats)
        # Calc loss and backprop gradients
        loss = -mll(output, train_y)
        loss.backward()
        print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
            i + 1, 10, loss.item(),
            model.covar_module.weight.item(),
            model.likelihood.noise.item()
        ))
        optimizer.step()

