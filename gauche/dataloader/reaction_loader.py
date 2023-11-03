"""
Subclass of the abstract data loader class for
reaction yield prediction datasets.
"""

from typing import Optional, Union, Callable, List

import os
import numpy as np
import pandas as pd

from gauche.dataloader import DataLoader
from rdkit.Chem.AllChem import (
    ReactionFromSmarts,
    ReactionToSmarts,
    MolFromSmiles,
    MolToSmiles,
)


class ReactionLoader(DataLoader):
    def __init__(self):
        super(ReactionLoader, self).__init__()
        self.task = "reaction_yield_prediction"
        self._features = None
        self._labels = None

    def validate(
        self, drop: Optional[bool] = True, canonicalize: Optional[bool] = True
    ):
        """
        Utility function to validate a read-in dataset of
        reaction representations and reactions yield labels.
        Checks if all SMILES/reaction SMARTS strings can be
        converted to rdkit molecules/reactions and if all
        labels are numeric and not NaNs.
        Optionally drops all invalid entries and makes the
        remaining SMILES/SMARTS strings canonical (default).

        :param drop: whether to drop invalid entries
        :type drop: bool
        :param canonicalize: whether to make the SMILES/SMARTS strings canonical
        :type canonicalize: bool
        """

        if isinstance(self.features, pd.Series):
            # reaction SMARTS are provided as a single strings
            invalid_rxns = (
                self.features.apply(ReactionFromSmarts).isnull().values
            )

        elif isinstance(self.features, pd.DataFrame):
            # reactant SMILES are provided as list of strings
            invalid_rxns = (
                self.features.applymap(MolFromSmiles)
                .isnull()
                .any(axis=1)
                .values
            )
        else:
            raise ValueError(
                f"Invalid reaction representation type {type(self.features)}. Must be either str or List[str]"
            )
        if np.any(invalid_rxns):
            print(
                f"Found {invalid_rxns.sum()} invalid reaction strings "
                f"{[x for i, x in enumerate(self.features) if invalid_rxns[i]]} "
                f"at indices {np.where(invalid_rxns)[0].tolist()}"
            )
            print(
                "To turn validation off, use dataloader.read_csv(..., validate=False)."
            )

        invalid_labels = np.isnan(self.labels).squeeze()
        if np.any(invalid_labels):
            print(
                f"Found {invalid_labels.sum()} invalid labels {self.labels[invalid_labels].squeeze()} "
                f"at indices {np.where(invalid_labels)[0].tolist()}"
            )
            print(
                "To turn validation off, use dataloader.read_csv(..., validate=False)."
            )

        invalid_idx = np.logical_or(invalid_rxns, invalid_labels)

        if drop:
            self.features = self.features.loc[~invalid_idx]
            self.labels = self.labels[~invalid_idx]
            assert len(self.features) == len(self.labels)

        if canonicalize:
            if isinstance(self.features, pd.Series):
                # reaction SMARTS are provided as a single column
                self.features = self.features.apply(
                    lambda r: ReactionToSmarts(ReactionFromSmarts(r))
                )
            elif isinstance(self.features, pd.DataFrame):
                # reactants are provided as separate columns
                self.features = self.features.applymap(
                    lambda r: MolToSmiles(MolFromSmiles(r))
                )

    def featurize(self, representation: Union[str, Callable], **kwargs):
        """Transforms reactions into the specified representation.

        :param representation: the desired reaction representation, one of
            [ohe, rxnfp, drfp, bag_of_smiles] or a callable that takes a list
            of SMILES strings as input and returns the desired featurization.
        :type representation: str or Callable
        :param kwargs: additional keyword arguments for the representation function
        :type kwargs: dict
        """

        valid_representations = [
            "ohe",
            "rxnfp",
            "drfp",
            "bag_of_smiles",
        ]

        if isinstance(representation, Callable):
            self.features = representation(self.features, **kwargs)

        elif representation == "ohe":
            from gauche.representations.fingerprints import one_hot

            self.features = one_hot(self.features, **kwargs)

        elif representation == "rxnfp":
            from gauche.representations.fingerprints import rxnfp

            assert isinstance(self.features, pd.Series), (
                f"Reaction fingerprints can only be computed "
                f"for a single reaction SMARTS string. Received "
                f"{len(self.features.columns)} columns instead."
            )
            self.features = rxnfp(self.features.to_list(), **kwargs)

        elif representation == "drfp":
            from gauche.representations.fingerprints import drfp

            assert isinstance(self.features, pd.Series), (
                f"Reaction fingerprints can only be computed "
                f"for a single reaction SMARTS string. Received "
                f"{len(self.features.columns)} columns instead."
            )
            self.features = drfp(self.features.to_list(), **kwargs)

        elif representation == "bag_of_smiles":
            from gauche.representations.strings import bag_of_characters

            assert isinstance(self.features, pd.Series), (
                f"Reaction fingerprints can only be computed "
                f"for a single reaction SMARTS string. Received "
                f"{len(self.features.columns)} columns instead."
            )
            self.features = bag_of_characters(
                self.features.to_list(), **kwargs
            )

        else:
            raise Exception(
                f"The specified representation choice {representation} is not a valid option."
                f"Choose between {valid_representations}."
            )

    def read_csv(
        self,
        path: str,
        reactant_column: Union[str, List[str]],
        label_column: str,
        validate: bool = True,
    ) -> None:
        """
        Loads a dataset from a .csv file. Reactants must be provided
        as either multple SMILES columns or a single reaction SMARTS column.

        :param path: path to the csv file
        :type path: str
        :param reactant_column: name of the column(s) containing the reactants
        :type reactant_column: str or List[str]
        :param label_column: name of the column containing the labels
        :type label_column: str
        :param validate: whether to validate the loaded data
        :type validate: bool
        """

        assert isinstance(
            label_column, str
        ), "label_column must be a single string"

        df = pd.read_csv(
            path,
            usecols=[reactant_column, label_column]
            if isinstance(reactant_column, str)
            else reactant_column + [label_column],
        )
        # assumes that data is properly pre-processed and that
        # NaNs indicate valid reactions without optional reagents
        # that do not fit neatly into a tabular format
        self.features = df[reactant_column].fillna("")
        self.labels = df[label_column].values.reshape(-1, 1)
        if validate:
            self.validate()

    def load_benchmark(
        self,
        benchmark: str,
        path=None,
    ) -> None:
        """Loads features and labels from one of the included benchmark datasets
                and feeds them into the DataLoader.

        :param benchmark: the benchmark dataset to be loaded, one of
            ``[DreherDoyle, SuzukiMiyaura, DreherDoyleRXN, SuzukiMiyauraRXN]``
              RXN suffix denotes that csv file contains reaction smiles in a dedicated column.
        :type benchmark: str
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
                    "reactions",
                    benchmark.removesuffix("RXN") + ".csv",
                )
            )

        self.read_csv(
            path=path,
            reactant_column=benchmarks[benchmark]["features"],
            label_column=benchmarks[benchmark]["labels"],
        )
