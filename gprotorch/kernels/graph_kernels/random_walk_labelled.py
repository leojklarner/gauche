"""
Contains GPyTorch random walk kernels on undirected graphs.
Notes: requires pytorch 1.8.0+ for torch.kron()
"""

import torch
import itertools
from copy import copy
from tqdm import tqdm
from gpytorch.kernels import Kernel
from gpytorch.kernels.kernel import default_postprocess_script
from math import factorial
from gpytorch.constraints import Positive
from gprotorch.kernels.graph_kernels.graph_kernel_utils import normalise_covariance, get_molecular_edge_labels, get_sparse_adj_mat
from rdkit.Chem import MolFromSmiles, rdmolops


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

    def forward(self,
                x1: list,
                x2: list,
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
        x1_equals_x2 = (x1 == x2)

        # Step 1: pre-compute all unique atom-bond-atom combinations for all molecules

        x1_edge_labels = [get_molecular_edge_labels(mol) for mol in x1]

        if x1_equals_x2:
            x2_edge_labels = x1_edge_labels
        else:
            x2_edge_labels = [get_molecular_edge_labels(mol) for mol in x2]

        # Step 2: populate the covariance matrix

        covar_mat = torch.zeros([len(x1), len(x2)])

        for idx1 in tqdm(range(len(x1))):
            for idx2 in range(len(x2)):

                # if applicable, fill lower triangular part
                # with already calculated upper triangular part
                if x1_equals_x2 and idx2 < idx1:
                    covar_mat[idx1, idx2] = covar_mat[idx2, idx1]

                else:

                    # calculate label-filtered adjacency matrices

                    num_atoms1, num_atoms2 = x1[idx1].GetNumAtoms(), x2[idx2].GetNumAtoms()
                    labels1, labels2 = x1_edge_labels[idx1], x2_edge_labels[idx2]
                    common_edge_labels = set(labels1.to_list()) & set(labels2.to_list())
                    product_graph_size = num_atoms1*num_atoms2

                    adj_mats = []

                    for label in common_edge_labels:

                        adj_mat1 = get_sparse_adj_mat(
                            index_tuples=labels1[labels1 == label].index.to_list(),
                            shape_tuple=(num_atoms1, num_atoms1)
                        )

                        adj_mat2 = get_sparse_adj_mat(
                            index_tuples=labels2[labels2 == label].index.to_list(),
                            shape_tuple=(num_atoms2, num_atoms2)
                        )

                        adj_mats.append((adj_mat1, adj_mat2))

                    start_stop_probs = torch.ones(product_graph_size)
                    if self.uniform_probabilities:
                        start_stop_probs.div_(product_graph_size)

                    x = copy(start_stop_probs)

                    for _ in range(self.num_fixed_point_iters):

                        mat_vec_prod = torch.zeros(product_graph_size)
                        for adj1, adj2 in adj_mats:
                            x = x.reshape((num_atoms2, num_atoms1))
                            x = adj2.mm((adj1.t().mm(x.t())).t())  # PyTorch only allows sparse@dense matmuls, use .t()
                            x = x.reshape(product_graph_size)
                            mat_vec_prod += x

                        x = start_stop_probs + self.weight * mat_vec_prod

                    covar_mat[idx1, idx2] = start_stop_probs.t().matmul(x)

        return covar_mat


if __name__ == '__main__':
    from gprotorch.dataloader.mol_prop import DataLoaderMP

    dataloader = DataLoaderMP()
    dataloader.load_benchmark("FreeSolv", "../../../data/property_prediction/FreeSolv.csv")
    mols = [MolFromSmiles(mol) for mol in dataloader.features]

    rw_kernel = RandomWalk()
    rw_kernel.forward(x1=mols[:500], x2=mols[:500])

    print()
