"""
Test suite for Tanimoto kernel.
"""

import pytest
import torch
import numpy as np
from gauche.kernels.fingerprint_kernels.tanimoto_kernel import (
    TanimotoKernel,
    batch_tanimoto_sim,
)
from gpytorch.kernels import ScaleKernel
from rdkit import DataStructs
from rdkit.Chem import AllChem, MolFromSmiles


@pytest.mark.parametrize(
    "x1, x2",
    [
        (torch.ones((2, 2)), torch.ones((2, 2))),
        (2 * torch.ones((2, 2)), 2 * torch.ones((2, 2))),
        (10 * torch.ones((2, 2)), 10 * torch.ones((2, 2))),
        (10000 * torch.ones((2, 2)), 10000 * torch.ones((2, 2))),
        (0.1 * torch.ones((2, 2)), 0.1 * torch.ones((2, 2))),
    ],
)
def test_tanimoto_similarity_with_equal_inputs(x1, x2):
    """Test the Tanimoto similarity metric between two equal input tensors."""
    tan_similarity = batch_tanimoto_sim(x1, x2)
    assert torch.isclose(tan_similarity, torch.ones((2, 2))).all()


def test_tanimoto_similarity_with_unequal_inputs():
    """
    Test the Tanimoto similarity metric between two unequal input tensors.
    [1, 1]   and   [2, 2] are x1 and x2 respectively yielding output [2/3, 2/3]
    [1, 1]         [2, 2]                                            [2/3, 2/3]
    """
    x1 = torch.ones((2, 2))
    x2 = 2 * torch.ones((2, 2))
    tan_similarity = batch_tanimoto_sim(x1, x2)

    assert torch.allclose(tan_similarity, torch.tensor(0.6666666666666))


def test_tanimoto_similarity_with_very_unequal_inputs():
    """
    Test the Tanimoto similarity metric between two tensors that are unequal in every dimension.
    """

    x1 = torch.tensor([[1, 3], [2, 4]], dtype=torch.float64)
    x2 = torch.tensor([[4, 2], [3, 1]], dtype=torch.float64)
    tan_similarity = batch_tanimoto_sim(x1, x2)

    assert torch.allclose(
        tan_similarity,
        torch.tensor([[0.5, 6 / 14], [2 / 3, 0.5]], dtype=torch.float64),
    )


def test_tanimoto_similarity_with_batch_dimension():
    """
    Test the Tanimoto similarity metric between two tensors that are unequal and have a batch dimension.
    """

    x1 = torch.tensor([[1, 3], [2, 4]], dtype=torch.float64)
    x2 = torch.tensor([[4, 2], [3, 1]], dtype=torch.float64)

    # Add a batch dimension
    x1 = x1[None, :]
    x2 = x2[None, :]

    tan_similarity = batch_tanimoto_sim(x1, x2)

    assert torch.allclose(
        tan_similarity,
        torch.tensor([[0.5, 6 / 14], [2 / 3, 0.5]], dtype=torch.float64),
    )


def test_tanimoto_kernel():
    """
    Test the Tanimoto kernel when integrated with GP.
    """

    x = torch.randint(0, 2, (10, 5))
    # Non-batch: Simple option
    covar_module = ScaleKernel(TanimotoKernel())
    covar = covar_module(x)  # Output: LazyTensor of size (10 x 10)
    assert covar.size() == torch.Size([10, 10])
    batch_x = torch.randint(0, 2, (2, 10, 5))
    # Batch: Simple option
    covar_module = ScaleKernel(TanimotoKernel())
    covar = covar_module(batch_x)  # Output: LazyTensor of size (2 x 10 x 10)
    assert covar.size() == torch.Size([2, 10, 10])


def test_against_rdkit():
    """
    Test Tanoimoto kernel against the RDKit implementation.
    """

    mols = [MolFromSmiles("CCOC"), MolFromSmiles("CCO"), MolFromSmiles("COC")]
    fpgen = AllChem.GetMorganGenerator()
    fps = [fpgen.GetFingerprint(mol) for mol in mols]

    rdkit_sims = np.array(
        [
            DataStructs.TanimotoSimilarity(fps[i], fps[j])
            for i in range(len(fps))
            for j in range(len(fps))
        ]
    )

    fps = torch.Tensor(fps)
    gauche_sims = TanimotoKernel()(fps, fps).numpy().flatten().tolist()

    assert np.allclose(
        rdkit_sims,
        gauche_sims,
    )
