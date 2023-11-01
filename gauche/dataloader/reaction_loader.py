import pandas as pd
from gauche.dataloader import DataLoader


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
        invalid_idx = []

    def featurize(self, representation, nBits=2048):
        """Transforms reactions into the specified representation.

        :param representation: the desired reaction representation, one of [ohe, rxnfp, drfp, bag_of_smiles]
        :type representation: str
        :param nBits: int giving the bit vector length for drfp representation. Default is 2048
        :type nBits: int
        """

        valid_representations = [
            "ohe",
            "rxnfp",
            "drfp",
            "bag_of_smiles",
        ]

        if representation == "ohe":
            from gauche.representations.fingerprints import one_hot

            self.features = one_hot(self.features)

        elif representation == "rxnfp":
            from gauche.representations.fingerprints import rxnfp

            self.features = rxnfp(self.features.to_list())

        elif representation == "drfp":
            from gauche.representations.fingerprints import drfp

            self.features = drfp(self.features.to_list(), nBits=nBits)

        elif representation == "bag_of_smiles":
            from gauche.representations.strings import bag_of_characters

            self.features = bag_of_characters(self.features.to_list())

        else:
            raise Exception(
                f"The specified representation choice {representation} is not a valid option. "
                f"Choose between {valid_representations}."
            )

    def load_benchmark(self, benchmark, path):
        """Loads features and labels from one of the included benchmark datasets
                and feeds them into the DataLoader.

        :param benchmark: the benchmark dataset to be loaded, one of
            ``[DreherDoyle, SuzukiMiyaura, DreherDoyleRXN, SuzukiMiyauraRXN]``
              RXN suffix denotes that csv file contains reaction smiles in a dedicated column.
        :type benchmark: str
        :param path: the path to the dataset in csv format
        :type path: str
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
            self.features = df[benchmarks[benchmark]["features"]].drop(nan_indices)
            self.labels = (
                df[benchmarks[benchmark]["labels"]].dropna().to_numpy().reshape(-1, 1)
            )


if __name__ == "__main__":
    loader = ReactionLoader()
    loader.load_benchmark(
        "DreherDoyle", "../../data/reactions/dreher_doyle_science_aar5169.csv"
    )

    loader.featurize("ohe")
    print(loader.features.shape)
