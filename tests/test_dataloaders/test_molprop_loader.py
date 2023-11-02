"""
Pytest-based unit tests for the molecular property 
prediction data loader.
"""

import pytest

import os
import itertools
from gauche.dataloader import DataLoaderMP


@pytest.mark.parametrize(
    "dataset, representation",
    [
        (d, f)
        for d, f in itertools.product(
            ["Photoswitch", "ESOL", "FreeSolv", "Lipophilicity"],
            [
                "ecfp_fingerprints",
                "fragments",
                "ecfp_fragprints",
                "molecular_graphs",
                "bag_of_smiles",
                "bag_of_selfies",
                "mqn",
            ],
        )
    ],
)
def test_benchmark_loading(dataset, representation):
    """
    Test if all benchmarks can be loaded with all representation.
    """

    dataset_root = os.path.abspath(
        os.path.join("..", "..", "data", "property_prediction")
    )

    dataloader = DataLoaderMP()
    dataloader.load_benchmark(
        dataset, path=os.path.join(dataset_root, dataset + ".csv")
    )
    dataloader.featurize(representation)
