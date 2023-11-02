"""
Instantiation of the abstract data loader class for
molecular property prediction datasets.
"""


import numpy as np
import pandas as pd

from gauche.dataloader import DataLoader
from rdkit.Chem import MolFromSmiles, MolToSmiles


class MolPropLoader(DataLoader):
    def __init__(self):
        super(MolPropLoader, self).__init__()
        self.task = "molecular_property_prediction"
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

        :param drop: whether to drop invalid entries
        :type drop: bool
        """

        invalid_idx = []

        # iterate through the features
        for i in range(len(self.features)):
            # try to convert each SMILES to an rdkit molecule
            mol = MolFromSmiles(self.features[i])

            # if it does not work, save the index and print its position to the console
            if mol is None:
                invalid_idx.append(i)
                print(f"Invalid SMILES at position {i+1}: {self.features[i]}")

        if drop:
            self.features = np.delete(self.features, invalid_idx).tolist()
            self.labels = np.delete(self.labels, invalid_idx)

    def featurize(
        self,
        representation,
        bond_radius=3,
        nBits=2048,
        graphein_config=None,
        max_ngram=5,
    ):
        """Transforms SMILES into the specified molecular representation.

        :param representation: the desired molecular representation, one of [fingerprints, fragments, fragprints]
        :type representation: str
        :param bond_radius: int giving the bond radius for Morgan fingerprints. Default is 3
        :type bond_radius: int
        :param nBits: int giving the bit vector length for Morgan fingerprints. Default is 2048
        :type nBits: int
        """

        valid_representations = [
            "ecfp_fingerprints",
            "fragments",
            "ecfp_fragprints",
            "graphs",
            "bag_of_smiles",
            "bag_of_selfies",
            "mqn",
        ]

        if representation == "ecfp_fingerprints":
            from gauche.representations.fingerprints import ecfp_fingerprints

            self.features = ecfp_fingerprints(
                self.features, bond_radius=bond_radius, nBits=nBits
            )

        elif representation == "fragments":
            from gauche.representations.fingerprints import fragments

            self.features = fragments(self.features)

        elif representation == "ecfp_fragprints":
            from gauche.representations.fingerprints import ecfp_fingerprints, fragments

            self.features = np.concatenate(
                (
                    ecfp_fingerprints(
                        self.features, bond_radius=bond_radius, nBits=nBits
                    ),
                    fragments(self.features),
                ),
                axis=1,
            )

        elif representation == "mqn":
            from gauche.representations.fingerprints import mqn_features

            self.features = mqn_features(self.features)

        elif representation == "bag_of_selfies":
            from gauche.representations.strings import bag_of_characters

            self.features = bag_of_characters(self.features, selfies=True)

        elif representation == "bag_of_smiles":
            from gauche.representations.strings import bag_of_characters

            self.features = bag_of_characters(self.features)

        elif representation == "molecular_graphs":
            from gauche.representations.graphs import molecular_graphs

            self.features = molecular_graphs(self.features, graphein_config)

        else:
            raise Exception(
                f"The specified representation choice {representation} is not a valid option."
                f"Choose between {valid_representations}."
            )

    def load_benchmark(self, benchmark, path):
        """Loads features and labels from one of the included benchmark datasets
        and feeds them into the DataLoader.

        :param benchmark: the benchmark dataset to be loaded, one of
            ``[Photoswitch, ESOL, FreeSolv, Lipophilicity]``.
        :type benchmark: str
        :param path: the path to the dataset in csv format
        :type path: str
        """

        benchmarks = {
            "Photoswitch": {
                "features": "SMILES",
                "labels": "E isomer pi-pi* wavelength in nm",
            },
            "Photoswitch_E_n_pi": {
                "features": "SMILES",
                "labels": "E isomer n-pi* wavelength in nm",
            },
            "Photoswitch_Z_pi_pi": {
                "features": "SMILES",
                "labels": "Z isomer pi-pi* wavelength in nm",
            },
            "Photoswitch_Z_n_pi": {
                "features": "SMILES",
                "labels": "Z isomer n-pi* wavelength in nm",
            },
            "ESOL": {
                "features": "smiles",
                "labels": "measured log solubility in mols per litre",
            },
            "FreeSolv": {"features": "smiles", "labels": "expt"},
            "Lipophilicity": {"features": "smiles", "labels": "exp"},
        }

        if benchmark in benchmarks:
            df = pd.read_csv(path)
            # drop nans from the datasets
            nans = df[benchmarks[benchmark]["labels"]].isnull().to_list()
            nan_indices = [nan for nan, x in enumerate(nans) if x]
            self.features = (
                df[benchmarks[benchmark]["features"]].drop(nan_indices).to_list()
            )
            self.labels = (
                df[benchmarks[benchmark]["labels"]].dropna().to_numpy().reshape(-1, 1)
            )

            # make SMILES canoncial
            self.features = [
                MolToSmiles(MolFromSmiles(smiles), isomericSmiles=False)
                for smiles in self.features
            ]

        else:
            raise ValueError(
                f"The specified benchmark choice ({benchmark}) is not a valid option. "
                f"Choose one of {list(benchmarks.keys())}."
            )
