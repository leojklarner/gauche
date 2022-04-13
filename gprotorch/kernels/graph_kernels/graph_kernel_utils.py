"""
Contains re-usable utility functions for calculating graph kernels.
"""
import numpy
import torch
from typing import Any
from torch.autograd import Function
import pandas as pd
import scipy.sparse as sp
import numpy as np
from collections import defaultdict
from gpytorch.utils import linear_cg
from numpy.linalg import inv
from numpy.linalg import eig
from numpy.linalg import multi_dot
from scipy.linalg import expm
from scipy.sparse.linalg import cg
from scipy.sparse.linalg import LinearOperator
import scipy.sparse as sp

def normalise_covariance(covariance_matrix):
    """
    Scales the covariance matrix to [0,1] by applying k(G1,G2) = k(G1,G2)/sqrt(k(G1,G1)*k(G2,G2))
    this is necessary when using different-size graphs as larger graphs will have more possible random walks
    and a smaller graph might be more similar to a larger graph than to itself.
    Args:
        covariance_matrix: the covariance matrix to scale

    Returns: the normalised covariance matrix

    """

    normalisation_factor = torch.unsqueeze(torch.sqrt(torch.diagonal(covariance_matrix)), -1)
    normalisation_factor = normalisation_factor * torch.transpose(normalisation_factor, -2, -1)
    return torch.div(covariance_matrix, normalisation_factor)


def sp_adj_mats(mol):

    adj_mats = defaultdict(lambda: [[], []])
    num_atoms = mol.GetNumAtoms()

    for bond in mol.GetBonds():

        start_atom = bond.GetBeginAtom().GetSymbol()
        start_idx = bond.GetBeginAtomIdx()
        end_atom = bond.GetEndAtom().GetSymbol()
        end_idx = bond.GetEndAtomIdx()

        label = hash(frozenset([
            start_atom, end_atom,
            bond.GetBondTypeAsDouble()
        ]))

        adj_mats[label][0].extend([start_idx, end_idx])
        adj_mats[label][1].extend([end_idx, start_idx])

    adj_mats = {k: sp.coo_matrix(
        (numpy.ones(len(v[0])), (v[0], v[1])), (num_atoms, num_atoms)).tocsr() for k, v in adj_mats.items()}

    return adj_mats


def get_label_adj_mats(mol):
    """

    Args:
        mol:

    Returns:

    """

    adj_mats = defaultdict(lambda: [[], []])
    num_atoms = mol.GetNumAtoms()

    for bond in mol.GetBonds():

        start_atom = bond.GetBeginAtom().GetSymbol()
        start_idx = bond.GetBeginAtomIdx()
        end_atom = bond.GetEndAtom().GetSymbol()
        end_idx = bond.GetEndAtomIdx()

        label = hash(frozenset([
            start_atom, end_atom,
            bond.GetBondTypeAsDouble()
        ]))

        adj_mats[label][0].extend([start_idx, end_idx])
        adj_mats[label][1].extend([end_idx, start_idx])

    adj_mats = {k: torch.sparse_coo_tensor(
        torch.LongTensor(v),
        torch.ones(len(v[0])),
        (num_atoms, num_atoms)).coalesce() for k, v in adj_mats.items()}

    return adj_mats


def kronecker_matvec(adj_mat1, num_atoms1, adj_mat2, num_atoms2, labels, vec):
    #vec = vec.reshape((num_atoms2, num_atoms1))
    vec = torch.hstack(torch.tensor_split(vec, num_atoms1))
    res = [torch.sparse.mm(adj_mat1[label], torch.mm(adj_mat2[label], vec).t()).t() for label in labels]
    if len(res) == 1:
        sum_res = res[0]
    elif len(res) > 1:
        sum_res = res[0]
        for add_res in res[1:]:
            sum_res += add_res
    else:
        raise NotImplementedError('')

    #return sum_res.reshape((num_atoms1 * num_atoms2, 1))
    return torch.vstack(torch.hsplit(sum_res, num_atoms1))


class LinearCG(Function):
    """
    Differentiable linear conjugate gradient solver for sparse Kronecker products.
    """

    @staticmethod
    def forward(ctx: Any, *args: Any, **kwargs: Any) -> Any:
        adj_mat1, num_atoms1, adj_mat2, num_atoms2, common_labels, start_stop_probs, weight = args
        inv_vec = linear_cg(
            matmul_closure=lambda x: x - weight * kronecker_matvec(
                adj_mat1, num_atoms1, adj_mat2, num_atoms2, common_labels, x),
            rhs=start_stop_probs
        )

        #grad = kronecker_matvec(adj_mat1, num_atoms1, adj_mat2, num_atoms2, common_labels, inv_vec)
        #weight_grad = linear_cg(
        #    matmul_closure=lambda x: x - weight * kronecker_matvec(
        #        adj_mat1, num_atoms1, adj_mat2, num_atoms2, common_labels, x),
        #    rhs=grad
        #)
        #weight_grad=torch.Tensor([1])
        #ctx.save_for_backward(weight_grad)

        return inv_vec

    @staticmethod
    def backward(ctx: Any, grad_output: Any) -> Any:
        weight_grad = ctx.saved_tensors
        grads = (None, None, None, None, None, None, None, grad_output.mm(weight_grad))
        return grads


class ScipyCG(Function):

    def forward(ctx: Any, *args: Any, **kwargs: Any) -> Any:

        adj_mat1, num_atoms1, adj_mat2, num_atoms2, common_labels, start_stop_probs, weight = args

        kron_shape = num_atoms1*num_atoms2, num_atoms1*num_atoms2

        def lsf(x):
            y = 0
            xm = x.reshape((num_atoms1, num_atoms2), order='F')
            for label in common_labels:
                y += np.reshape(multi_dot((adj_mat1[label], xm, adj_mat2[label])), (kron_shape,), order='F')
            return x - weight * y

        A = LinearOperator((kron_shape, kron_shape), matvec=lambda x: lsf(x))
        b = np.ones(num_atoms1*num_atoms2)
        x_sol, _ = cg(A, b, tol=1.0e-6, maxiter=20, atol='legacy')
