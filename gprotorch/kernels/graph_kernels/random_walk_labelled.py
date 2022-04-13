"""
Contains GPyTorch random walk kernels on undirected graphs.
Notes: requires pytorch 1.8.0+ for torch.kron()
"""

import torch
import cProfile
import pstats
from pstats import SortKey
import itertools
import pandas as pd
import numpy as np
from copy import copy
from tqdm import tqdm
from gpytorch.kernels import Kernel
from gpytorch.kernels.kernel import default_postprocess_script
from math import factorial
from gpytorch.utils import linear_cg
from gpytorch.lazy import kronecker_product_lazy_tensor
from gpytorch.constraints import Positive
from gprotorch.kernels.graph_kernels.graph_kernel_utils import normalise_covariance, get_label_adj_mats, LinearCG, sp_adj_mats
from rdkit.Chem import MolFromSmiles, rdmolops, GetAdjacencyMatrix

from scipy.sparse.linalg import LinearOperator
from scipy.sparse.linalg import cg
from numpy.linalg import multi_dot


linear_cg_solver = LinearCG.apply


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

        # Step 1a: pre-compute all unique atom-bond-atom combinations for all molecules

        x1_adj_mats = [get_label_adj_mats(x) for x in x1]

        if x1_equals_x2:
            x2_adj_mats = x1_adj_mats
        else:
            x2_adj_mats = [get_label_adj_mats(x) for x in x2]

        # Step 2: populate the covariance matrix

        covar_mat = torch.zeros([len(x1), len(x2)])

        for idx1 in tqdm(range(len(x1))):
            for idx2 in range(len(x2)):

                # if applicable, fill lower triangular part
                # with already calculated upper triangular part
                if x1_equals_x2 and idx2 < idx1:
                    covar_mat[idx1, idx2] = covar_mat[idx2, idx1]

                else:

                    adj_mat1, adj_mat2 = x1_adj_mats[idx1], x2_adj_mats[idx2]

                    common_labels = adj_mat1.keys() & adj_mat2.keys()

                    if not common_labels:
                        covar_mat[idx1, idx2] = torch.zeros(1)

                    else:
                        num_atoms1, num_atoms2 = x1[idx1].GetNumAtoms(), x2[idx2].GetNumAtoms()
                        kron_size = num_atoms1 * num_atoms2
                        start_stop_probs = torch.ones(kron_size).unsqueeze(1)
                        if self.uniform_probabilities:
                            start_stop_probs /= kron_size


                        #inv_vec = linear_cg_solver(
                        #    adj_mat1, num_atoms1, adj_mat2, num_atoms2,
                        #    common_labels, start_stop_probs, self.weight,
                        #)

                        if False:

                            def lsf(x):
                                y = 0
                                xm = x.reshape((num_atoms1, num_atoms2), order='F')
                                for label in common_labels:
                                    y += np.reshape(adj_mat1[label] @ xm @ adj_mat2[label], (kron_size,),
                                                    order='F')
                                return x - self.weight.detach().numpy() * y

                            A = LinearOperator((kron_size, kron_size), matvec=lsf)
                            b = np.ones(num_atoms1 * num_atoms2)
                            inv_vec, _ = cg(A, b, tol=1.0e-6, maxiter=20, atol='legacy')


                            max_diff = (inv_vec - correct_solve).abs().max().item()
                            #if not max_diff < 1e-5:
                            #    print(idx1, idx2)
                            #    for label in common_labels:
                            #        print(adj_mat1[label])
                            #        print(adj_mat2[label])
                            #    print()

                        kron_mat = torch.zeros((num_atoms1 * num_atoms2, num_atoms1 * num_atoms2))
                        for label in common_labels:
                            mat1 = adj_mat1[label].to_dense()
                            mat2 = adj_mat2[label].to_dense()
                            kron_mat += torch.kron(mat1, mat2)

                        kron_mat = torch.diag(torch.ones((num_atoms1 * num_atoms2))) - self.weight * kron_mat
                        correct_solve = torch.linalg.solve(kron_mat, start_stop_probs)

                        covar_mat[idx1, idx2] = start_stop_probs.T @ correct_solve

                        #self.weight.backward()

        if self.normalise:
            covar_mat = normalise_covariance(covar_mat)

        return covar_mat


if __name__ == '__main__':
    from gprotorch.dataloader.mol_prop import DataLoaderMP

    dataloader = DataLoaderMP()
    dataloader.load_benchmark("FreeSolv", "../../../data/property_prediction/FreeSolv.csv")
    mols = [MolFromSmiles(mol) for mol in dataloader.features]

    rw_kernel = RandomWalk()
    rw_kernel.weight = 1/16
    cov_mat = rw_kernel.forward(x1=mols[:300], x2=mols[:300])
    print(cov_mat)
