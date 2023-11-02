"""
Pytest-based unit tests for the molecular property 
prediction data loader.
"""

import pytest

import os
import itertools
import numpy as np
from gauche.dataloader import MolPropLoader


benckmark_cols = {
    "Photoswitch": {
        "features": "SMILES",
        "labels": "E isomer pi-pi* wavelength in nm",
    },
    "ESOL": {
        "features": "smiles",
        "labels": "measured log solubility in mols per litre",
    },
    "FreeSolv": {"features": "smiles", "labels": "expt"},
    "Lipophilicity": {"features": "smiles", "labels": "exp"},
}


@pytest.mark.parametrize(
    "dataset, representation, kwargs",
    [
        (d, f, kw)
        for d, (f, kw) in itertools.product(
            ["Photoswitch", "ESOL", "FreeSolv", "Lipophilicity"],
            [
                ("ecfp_fingerprints", {"bond_radius": 2, "nBits": 1024}),
                ("fragments", {}),
                ("ecfp_fragprints", {"bond_radius": 2, "nBits": 1024}),
                ("molecular_graphs", {"graphein_config": None}),
                ("bag_of_smiles", {"max_ngram": 3}),
                ("bag_of_selfies", {"max_ngram": 3}),
                ("mqn", {}),
            ],
        )
    ],
)
def test_benchmark_loader(dataset, representation, kwargs):
    """
    Test if all benchmarks can be loaded with all representation.
    """

    dataset_root = os.path.abspath(
        os.path.join("..", "..", "data", "property_prediction")
    )

    # load through benchmark loading method

    dataloader = MolPropLoader()
    dataloader.load_benchmark(
        benchmark=dataset,
    )
    dataloader.featurize(representation, **kwargs)

    # load through csv loading method

    dataloader_csv = MolPropLoader()
    dataloader_csv.read_csv(
        path=os.path.join(dataset_root, dataset + ".csv"),
        smiles_column=benckmark_cols[dataset]["features"],
        labels_column=benckmark_cols[dataset]["labels"],
    )
    dataloader_csv.featurize(representation, **kwargs)

    # check if both dataloaders have the same features and labels
    # to ensure that there is no stochasticity in the featurization
    # skip for molecular_graphs representation, since that might
    # require an isomorphism test
    if representation != "molecular_graphs":
        assert np.array_equal(dataloader.features, dataloader_csv.features)
    assert np.array_equal(dataloader.labels, dataloader_csv.labels)


def test_invalid_data():
    """
    Test data loader for invalid input csv.
    """

    dataloader = MolPropLoader()
    dataloader.read_csv(
        path=os.path.join(os.getcwd(), "invalid_molprop_data.csv"),
        smiles_column="SMILES",
        labels_column="labels",
    )
    assert len(dataloader.features) == 2 and len(dataloader.labels) == 2


@pytest.mark.parametrize(
    "representation", ["XYZ", 2, True, None, ("ecfp_fingerprints", "fragments")]
)
def test_invalid_representation(representation):
    """
    Test behaviour of data loader for invalid representation choice.
    """

    dataloader = MolPropLoader()
    dataloader.load_benchmark("ESOL")
    with pytest.raises(Exception):
        dataloader.featurize(representation)


@pytest.mark.parametrize("benchmark", ["XYZ", 2, True, None, ("ESOL", "Photoswitch")])
def test_invalid_benchmark(benchmark):
    """
    Test behaviour of data loader for invalid benchmark choice.
    """

    dataloader = MolPropLoader()
    with pytest.raises(Exception):
        dataloader.load_benchmark(benchmark)
