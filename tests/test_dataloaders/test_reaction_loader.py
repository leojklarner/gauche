"""
Pytest-based unit tests for the reaction yield 
prediction data loader.
"""

import pytest

import os
import itertools
import numpy as np
from gauche.dataloader import ReactionLoader

benckmark_cols = {
    "DreherDoyle": {
        "features": ["ligand", "additive", "base", "aryl halide"],
        "labels": "yield",
    },
    "DreherDoyleRXN": {"features": "rxn", "labels": "yield"},
    "SuzukiMiyaura": {
        "features": [
            "reactant_1_smiles",
            "reactant_2_smiles",
            "catalyst_smiles",
            "ligand_smiles",
            "reagent_1_smiles",
            "solvent_1_smiles",
        ],
        "labels": "yield",
    },
    "SuzukiMiyauraRXN": {"features": "rxn", "labels": "yield"},
}


@pytest.mark.parametrize(
    "dataset, representation, kwargs",
    [
        (d, f, kw)
        for d, (f, kw) in itertools.product(
            ["DreherDoyle", "DreherDoyleRXN", "SuzukiMiyaura", "SuzukiMiyauraRXN"],
            [
                ("ohe", {}),
                ("rxnfp", {}),
                ("drfp", {"nBits": 1024}),
                ("bag_of_smiles", {"max_ngram": 3}),
            ],
        )
    ],
)
def test_benchmark_loader(dataset, representation, kwargs):
    """
    Test if all benchmarks can be loaded with all representation.
    """

    if (
        "RXN" not in dataset
        and representation in ["rxnfp", "drfp", "bag_of_smiles"]
        or "RXN" in dataset
        and representation == "ohe"
    ):
        # rxnfp, drfp, and bag_of_smiles only work for reaction SMARTS
        # and vice versa ohe only works for multiple reaction SMILES
        with pytest.raises(Exception):
            dataloader = ReactionLoader()
            ReactionLoader().load_benchmark(
                benchmark=dataset,
            )
            dataloader.featurize(representation, **kwargs)
    else:
        # load through benchmark loading method

        dataloader = ReactionLoader()
        dataloader.load_benchmark(
            benchmark=dataset,
        )
        dataloader.featurize(representation, **kwargs)


def test_invalid_data():
    """
    Test data loader for invalid input csv.
    """

    dataloader = ReactionLoader()
    dataloader.read_csv(
        path=os.path.join(os.path.abspath(__file__), "invalid_reaction_data.csv"),
        reactant_column=["ligand", "additive", "base", "aryl halide"],
        label_column="yield",
    )
    assert len(dataloader.features) == 1 and len(dataloader.labels) == 1


@pytest.mark.parametrize("representation", ["XYZ", 2, True, None, ("ohe", "rxnfp")])
def test_invalid_representation(representation):
    """
    Test behaviour of data loader for invalid representation choice.
    """

    dataloader = ReactionLoader()
    dataloader.load_benchmark("DreherDoyle")
    with pytest.raises(Exception):
        dataloader.featurize(representation)


@pytest.mark.parametrize(
    "benchmark", ["XYZ", 2, True, None, ("DreherDoyle", "SuzukiMiyaura")]
)
def test_invalid_benchmark(benchmark):
    """
    Test behaviour of data loader for invalid benchmark choice.
    """

    dataloader = ReactionLoader()
    with pytest.raises(Exception):
        dataloader.load_benchmark(benchmark)


def test_custom_featurizer_smiles_length():
    """
    Test behaviour of data loader for custom featurizer that
    returns the length of the SMILES string.
    """

    def custom_featurizer(smiles):
        return np.array([len(s) for s in smiles]).reshape(-1, 1)

    dataloader = ReactionLoader()
    dataloader.load_benchmark("DreherDoyleRXN")
    dataloader.featurize(custom_featurizer)
    assert dataloader.features.shape[1] == 1
