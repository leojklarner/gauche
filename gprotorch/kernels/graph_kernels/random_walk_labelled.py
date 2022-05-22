"""
Contains GPyTorch random walk kernels on undirected graphs.
Notes: requires pytorch 1.8.0+ for torch.kron()
"""

import torch
from tqdm import trange
from gprotorch import Kernel, Inputs
from gprotorch.kernels.graph_kernels.graph_kernel_utils import normalise_covariance, kronecker_inner_product
from rdkit.Chem import MolFromSmiles
from gpytorch import settings
from functools import lru_cache

class RandomWalk(Kernel):
    @lru_cache(maxsize=5)
    def kern(self, x1):
        if type(x1) == Inputs:
            x1 = x1.data
        x1_node_num = [x.shape[0] for x in x1]

        x2_node_num = x1_node_num
        x2 = x1
        rw_probs = torch.ones(max(x1_node_num) * max(x2_node_num), device=x1[0].device)

        covar_mat = torch.zeros([len(x1), len(x2)])

        for idx1 in trange(len(x1)):
            for idx2 in range(len(x2)):

                if idx2 < idx1:
                    covar_mat[idx1, idx2] = covar_mat[idx2, idx1]
                else:
                    kron_mat = kronecker_inner_product(x1[idx1], x2[idx2])
                    rw_probs_reduced = rw_probs[:x1_node_num[idx1]*x2_node_num[idx2]]
                    kron_mat = torch.diag(rw_probs_reduced) - self.weight * kron_mat

                    correct_solve = torch.linalg.solve(kron_mat, rw_probs_reduced)
                    covar_mat[idx1, idx2] = correct_solve.sum()

        # covar_mat = normalise_covariance(covar_mat)
        return covar_mat
