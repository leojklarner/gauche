"""
Instantiation of the abstract data loader class for
molecular property prediction datasets.
"""

from typing import Optional, Union, Callable

import os
import numpy as np
import pandas as pd

from gauche.dataloader import DataLoader
from rdkit.Chem import MolFromSmiles, MolToSmiles


class MolPropLoader(DataLoader):
    """
    Data loader class for molecular property prediction
    datasets with a single regression target.
    Expects input to be a csv file with one column for
    SMILES strings and one column for labels.
    Contains methods to validate the dataset and to
    transform the SMILES strings into different
    molecular representations.
    """

    def __init__(self):
        super(MolPropLoader, self).__init__()
        self.task = "molecular_property_prediction"
        self._features = None
        self._labels = None

    def validate(
        self, drop: Optional[bool] = True, canonicalize: Optional[bool] = True
    ) -> None:
        """
        Utility function to validate a read-in dataset of smiles and labels by
        checking that all SMILES strings can be converted to rdkit molecules
        and that all labels are numeric and not NaNs.
        Optionally drops all invalid entries and makes the
        remaining SMILES strings canonical (default).

        :param drop: whether to drop invalid entries
        :type drop: bool
        :param canonicalize: whether to make the SMILES strings canonical
        :type canonicalize: bool
        """

        invalid_mols = np.array(
            [True if MolFromSmiles(x) is None else False for x in self.features]
        )
        if np.any(invalid_mols):
            print(
                f"Found invalid SMILES strings "
                f"{[x for i, x in enumerate(self.features) if invalid_mols[i]]} "
                f"at indices {np.where(invalid_mols)[0].tolist()}"
            )

        invalid_labels = np.isnan(self.labels).squeeze()
        if np.any(invalid_labels):
            print(
                f"Found invalid labels {self.labels[invalid_labels].squeeze()} "
                f"at indices {np.where(invalid_labels)[0].tolist()}"
            )

        invalid_idx = np.logical_or(invalid_mols, invalid_labels)

        if drop:
            self.features = [
                x for i, x in enumerate(self.features) if not invalid_idx[i]
            ]
            self.labels = self.labels[~invalid_idx]
            assert len(self.features) == len(self.labels)

        if canonicalize:
            self.features = [
                MolToSmiles(MolFromSmiles(smiles), isomericSmiles=False)
                for smiles in self.features
            ]

    def featurize(self, representation: Union[str, Callable], **kwargs) -> None:
        """Transforms SMILES into the specified molecular representation.

        :param representation: the desired molecular representation, one of [ecfp_fingerprints, fragments, ecfp_fragprints, molecular_graphs, bag_of_smiles, bag_of_selfies, mqn] or a callable that takes a list of SMILES strings as input and returns the desired featurization.

        :type representation: str or Callable
        :param kwargs: additional keyword arguments for the representation function
        :type kwargs: dict
        """

        assert isinstance(representation, (str, Callable)), (
            f"The specified representation choice {representation} is not "
            f"a valid type. Please choose a string from the list of available "
            f"representations or provide a callable that takes a list of "
            f"SMILES strings as input and returns the desired featurization."
        )

        valid_representations = [
            "ecfp_fingerprints",
            "fragments",
            "ecfp_fragprints",
            "molecular_graphs",
            "bag_of_smiles",
            "bag_of_selfies",
            "mqn",
        ]

        if isinstance(representation, Callable):
            self.features = representation(self.features, **kwargs)

        elif representation == "ecfp_fingerprints":
            from gauche.representations.fingerprints import ecfp_fingerprints

            self.features = ecfp_fingerprints(self.features, **kwargs)

        elif representation == "fragments":
            from gauche.representations.fingerprints import fragments

            self.features = fragments(self.features)

        elif representation == "ecfp_fragprints":
            from gauche.representations.fingerprints import ecfp_fingerprints, fragments

            self.features = np.concatenate(
                (
                    ecfp_fingerprints(self.features, **kwargs),
                    fragments(self.features),
                ),
                axis=1,
            )

        elif representation == "mqn":
            from gauche.representations.fingerprints import mqn_features

            self.features = mqn_features(self.features)

        elif representation == "bag_of_selfies":
            from gauche.representations.strings import bag_of_characters

            self.features = bag_of_characters(self.features, selfies=True, **kwargs)

        elif representation == "bag_of_smiles":
            from gauche.representations.strings import bag_of_characters

            self.features = bag_of_characters(self.features, **kwargs)

        elif representation == "molecular_graphs":
            from gauche.representations.graphs import molecular_graphs

            self.features = molecular_graphs(self.features, **kwargs)

        else:
            raise Exception(
                f"The specified representation choice {representation} is not a valid option."
                f"Choose between {valid_representations}."
            )

    def read_csv(self, path: str, smiles_column: str, labels_column: str) -> None:
        """
        Loads a dataset from a .csv file. The file must contain the two
        specified columns with the SMILES strings and labels.

        :param path: path to the csv file
        :type path: str
        :param smiles_column: name of the column containing the SMILES strings
        :type smiles_column: str
        :param labels_column: name of the column containing the labels
        :type labels_column: str
        """

        df = pd.read_csv(path, usecols=[smiles_column, labels_column])
        self.features = df[smiles_column].to_list()
        self.labels = df[labels_column].values.reshape(-1, 1)
        self.validate()

    def load_benchmark(
        self,
        benchmark: str,
        path=None,
    ) -> None:
        """
        Loads a selection of existing benchmarks data directory.

        :param benchmark: the benchmark dataset to be loaded, one of
            ``[Photoswitch, ESOL, FreeSolv, Lipophilicity]``.
        :type benchmark: str
        :param path: the path to the directory that contains the dataset,
            defaults to the data directory of the project if None
        :type path: str
        """

        benchmarks = {
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

        assert benchmark in benchmarks.keys(), (
            f"The specified benchmark choice ({benchmark}) is not a valid option. "
            f"Choose one of {list(benchmarks.keys())}."
        )

        # if no path is specified, use the default data directory
        if path is None:
            path = os.path.abspath(
                os.path.join(
                    os.path.abspath(__file__),
                    "..",
                    "..",
                    "..",
                    "data",
                    "property_prediction",
                    benchmark + ".csv",
                )
            )

        self.read_csv(
            path=path,
            smiles_column=benchmarks[benchmark]["features"],
            labels_column=benchmarks[benchmark]["labels"],
        )
