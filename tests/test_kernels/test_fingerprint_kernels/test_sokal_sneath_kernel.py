"""
Test suite for Sokal-Sneath kernel.
"""

import pytest
import torch
from gpytorch.kernels import ScaleKernel

from gauche.kernels.fingerprint_kernels.sokal_sneath_kernel import (
    SokalSneathKernel,
    batch_sokal_sneath_sim,
)

tkwargs = {"dtype": torch.double}


@pytest.mark.parametrize(
    "x1, x2",
    [
        (torch.ones((2, 2), **tkwargs), torch.ones((2, 2), **tkwargs)),
    ],
)
def test_sokal_sneath_similarity_with_equal_inputs(x1, x2):
    """Test the Sokal-Sneath similarity metric between two equal input tensors."""
    similarity = batch_sokal_sneath_sim(x1, x2)
    assert torch.isclose(similarity, torch.ones((2, 2), **tkwargs)).all()


def test_sokal_sneath_similarity_with_unequal_inputs():
    """Test the Sokal-Sneath similarity metric between two unequal input tensors."""
    x1 = torch.tensor([1, 0, 1, 1], **tkwargs)
    x2 = torch.tensor([1, 1, 0, 0], **tkwargs)
    # Add a batch dimension
    x1 = x1[None, :]
    x2 = x2[None, :]
    similarity = batch_sokal_sneath_sim(x1, x2)

    assert torch.allclose(similarity, torch.tensor(1.0 / 7.0, **tkwargs))


def test_sokal_sneath_kernel():
    """Test the Sokal-Sneath kernel when integrated with GP."""

    x = torch.randint(0, 2, (10, 5))
    # Non-batch: Simple option
    covar_module = ScaleKernel(SokalSneathKernel())
    covar = covar_module(x)  # Output: LazyTensor of size (10 x 10)
    assert covar.size() == torch.Size([10, 10])
    batch_x = torch.randint(0, 2, (2, 10, 5))
    # Batch: Simple option
    covar_module = ScaleKernel(SokalSneathKernel())
    covar = covar_module(batch_x)  # Output: LazyTensor of size (2 x 10 x 10)
    assert covar.size() == torch.Size([2, 10, 10])
