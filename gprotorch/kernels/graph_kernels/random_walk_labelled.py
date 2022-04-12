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
from gpytorch.constraints import Positive
from gprotorch.kernels.graph_kernels.graph_kernel_utils import normalise_covariance, get_molecular_edge_labels, get_sparse_adj_mat
from rdkit.Chem import MolFromSmiles, rdmolops, GetAdjacencyMatrix


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

        #x1_edge_labels = [get_molecular_edge_labels(mol) for mol in x1]

        #if x1_equals_x2:
        #    x2_edge_labels = x1_edge_labels
        #else:
        #    x2_edge_labels = [get_molecular_edge_labels(mol) for mol in x2]

        # Step 1b: pre-compute all adjacency matrices
        edge_lists1 = []
        for mol in x1:
            adj_mat = GetAdjacencyMatrix(mol)
            adj_mat = adj_mat @ np.diag(1 / adj_mat.sum(1))
            edge_list = pd.DataFrame(
                data=adj_mat[adj_mat.astype(bool)],
                index=pd.MultiIndex.from_arrays(adj_mat.nonzero(), names=['src', 'dst']),
                columns=['weight']
            )
            edge_src, edge_dst, edge_label = get_molecular_edge_labels(mol)
            edge_list.loc[zip(edge_src, edge_dst), 'label'] = edge_label
            edge_list.loc[zip(edge_dst, edge_src), 'label'] = edge_label
            edge_list = edge_list.reset_index()
            edge_lists1.append({k:np.hsplit(v.values[:, :-1], 3) for k,v in edge_list.groupby('label')})

        if x1_equals_x2:
            edge_lists2 = edge_lists1
        else:
            edge_lists2 = []
            for mol in x2:
                adj_mat = GetAdjacencyMatrix(mol)
                adj_mat = adj_mat @ np.diag(1 / adj_mat.sum(1))
                edge_list = pd.DataFrame(
                    data=adj_mat[adj_mat.astype(bool)],
                    index=pd.MultiIndex.from_arrays(adj_mat.nonzero(), names=['src', 'dst']),
                    columns=['weight']
                )
                edge_src, edge_dst, edge_label = get_molecular_edge_labels(mol)
                edge_list.loc[zip(edge_src, edge_dst), 'label'] = edge_label
                edge_list.loc[zip(edge_dst, edge_src), 'label'] = edge_label
                edge_list = edge_list.reset_index()
                edge_lists2.append({k: np.hsplit(v.values[:, :-1], 3) for k, v in edge_list.groupby('label')})

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

                    edge_list1 = edge_lists1[idx1]
                    edge_list2 = edge_lists2[idx2]
                    common_labels = edge_list1.keys() & edge_list2.keys()
                    num_atoms1, num_atoms2 = x1[idx1].GetNumAtoms(), x2[idx2].GetNumAtoms()
                    kron_size = num_atoms1*num_atoms2

                    # derive the cartesian product of the two edge lists for all
                    # entries with identical augmented edge labels and
                    # derive the indices of the corresponding Kronecker product
                    #common_labels = edge_list1.merge(edge_list2, on='label', suffixes=('_1', '_2'))

                    if not common_labels:
                        covar_mat[idx1, idx2] = torch.zeros(1)
                        continue

                    kron_row = []
                    kron_col = []
                    kron_data = []

                    for label in common_labels:
                        i1, j1, k1 = edge_list1[label]
                        i2, j2, k2 = edge_list2[label]

                        kron_row.append((num_atoms2 * i1 + i2.T).flatten())
                        kron_col.append((num_atoms2 * j1 + j2.T).flatten())
                        kron_data.append((k1 * k2.T).flatten())

                    kron_row = np.concatenate(kron_row)
                    kron_col = np.concatenate(kron_col)
                    kron_data = torch.from_numpy(np.concatenate(kron_data)).float()

                    if False:

                        kron_mat = torch.sparse_coo_tensor(
                            indices=torch.LongTensor([kron_row, kron_col]), values=kron_data,
                            size=(kron_size, kron_size)
                        ).coalesce()

                        # construct a fitting diagonal Tensor and calculate (I - lambda * W)
                        diag = torch.sparse_coo_tensor(
                            indices=torch.LongTensor([range(kron_size), range(kron_size)]),
                            values=torch.ones(kron_size), size=(kron_size, kron_size),
                        )

                    kron_mat = torch.zeros([kron_size, kron_size])
                    kron_mat[kron_row, kron_col] = kron_data

                    kron_mat = torch.diag(torch.ones(kron_size)) - self. weight * kron_mat

                    start_stop_probs = torch.ones(kron_size).unsqueeze(1)
                    if self.uniform_probabilities:
                        start_stop_probs.div_(kron_size)

                    #result = linear_cg(kron_mat.matmul, start_stop_probs)
                    result = torch.linalg.solve(kron_mat, start_stop_probs)
                    result = start_stop_probs.T @ result

                    covar_mat[idx1, idx2] = result

        if self.normalise:
            covar_mat = normalise_covariance(covar_mat)

        return covar_mat


if __name__ == '__main__':
    from gprotorch.dataloader.mol_prop import DataLoaderMP

    dataloader = DataLoaderMP()
    dataloader.load_benchmark("FreeSolv", "../../../data/property_prediction/FreeSolv.csv")
    mols = [MolFromSmiles(mol) for mol in dataloader.features]

    rw_kernel = RandomWalk()
    rw_kernel.weight = 0.2
    cov_mat = rw_kernel.forward(x1=mols[:100], x2=mols[:100])
    print(cov_mat)
