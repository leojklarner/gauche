import numpy as np
import pandas as pd
from drfp import DrfpEncoder
from rxnfp.transformer_fingerprints import (
    get_default_model_and_tokenizer,
    RXNBERTFingerprintGenerator,
)

from gprotorch.data_featuriser import one_hot, rxnfp, drfp
from gprotorch.dataloader import DataLoader


class ReactionLoader(DataLoader):
    def __init__(self):
        super(ReactionLoader, self).__init__()
        self.task = "reaction_yield_prediction"
        self._features = None
        self._labels = None

    @property
    def features(self):
        return self._features

    @features.setter
    def features(self, value):
        self._features = value

    @property
    def labels(self):
        return self._labels

    @labels.setter
    def labels(self, value):
        self._labels = value

    def validate(self, drop=True):
        """Checks if the features are valid SMILES strings and (potentially)
        drops the entries that are not.

        Args:
            drop: whether to drop invalid entries

        """

        invalid_idx = []
        # todo how to validate reactions

    ## TODO should add *kwargs here?
    def featurize(self, representation):
        """Transforms reaction components into the specified reaction representation.

        Args:
            representation: the desired reaction representation, one of [one-hot, rxnfp, drfp]
        """

        valid_representations = [
            "ohe",
            "rxnfp",
            "drfp",
        ]

        if representation == "ohe":
            self.features = one_hot(self.features)

        elif representation == "rxnfp":
            self.features = rxnfp(self.features.to_list())

        elif representation == "drfp":
            self.features = drfp(self.features.to_list())

        else:
            raise Exception(
                f"The specified representation choice {representation} is not a valid option."
                f"Choose between {valid_representations}."
            )

    def load_benchmark(self, benchmark, path):
        """Loads features and labels from one of the included benchmark datasets
        and feeds them into the DataLoader.

        Args:
            benchmark: the benchmark dataset to be loaded, one of
            [Photoswitch, ESOL, FreeSolv, Lipophilicity]
            path: the path to the dataset in csv format

        """

        benchmarks = {
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

        if benchmark not in benchmarks.keys():

            raise Exception(
                f"The specified benchmark choice ({benchmark}) is not a valid option. "
                f"Choose one of {list(benchmarks.keys())}."
            )

        else:

            df = pd.read_csv(path)
            # drop nans from the datasets
            nans = df[benchmarks[benchmark]["labels"]].isnull().to_list()
            nan_indices = [nan for nan, x in enumerate(nans) if x]
            self.features = df[benchmarks[benchmark]["features"]].drop(
                nan_indices
            )  # .to_list()
            self.labels = (
                df[benchmarks[benchmark]["labels"]]
                .dropna()
                .to_numpy()
                .reshape(-1, 1)
            )


if __name__ == "__main__":
    loader = ReactionLoader()
    loader.load_benchmark(
        "SuzukiMiyauraRXN", "../../data/reactions/suzuki_miyaura_data.csv"
    )
    # loader.featurize('ohe')
    # print(loader.features.shape)
    loader.featurize("rxnfp")
    print(loader.features.shape)
    # loader.featurize('drfp')
    # print(loader.features.shape)
    print(loader)
