"""
Implementation of the abstract DataLoader class for
molecular property prediction tasks.

Author: Leo Klarner (https://github.com/leojklarner), March 2021
"""

import numpy as np
import pandas as pd
from rdkit.Chem import MolFromSmiles

from gprotorch.dataloader import DataLoader
from gprotorch.dataloader.dataloader_utils import (molecule_fingerprints,
                                                   molecule_fragments)


class DataLoaderMP(DataLoader):
    """
    Task-specific implementation of the abstract DataLoader base class
    for molecular property prediction datasets.
    Reads in and stores molecules as SMILES representations and featurises them
    by calculating physicochemical descriptors, molecular fingerprints and others.
    """

    def __init__(self, validate_internal_rep=True):
        """
        Class initialisation.

        Args:
            validate_internal_rep: whether to verify that the provided objects are valid
            SMILES representations by attempting to parse them to rdkit Mol objects.

        """

        super(DataLoaderMP, self).__init__(validate_internal_rep=validate_internal_rep)

    def _validate(self, data):
        """
        Checks whether the provided SMILES represenations are valid by attempting to parse
        them to rdkit Mol objects, discarding invalid ones.

        Args:
            data: the SMILES representations to be checked

        Returns: (valid, invalid) tuple of lists, containing the
        valid and invalid SMILES representations

        """

        valid = []
        invalid = []

        # iterate through the provided SMILES representaions

        for i in range(len(data)):

            # denote the given SMILES representation as valid or invalid
            # depending on whether an rdkit Mol object could be parsed

            mol = MolFromSmiles(data[i])

            if mol is None:
                invalid.append(data[i])
            else:
                valid.append(data[i])

        return valid, invalid

    def featurize(self, representation, bond_radius=3, nBits=2048):
        """
        Applies the specified transformation to the currently loaded objects and
        stores the resulting features in self.features.

        Args:
            representation: the desired molecular representation,
            one of ["fingerprints", "fragments", "fragprints"]
            bond_radius: the bond radius for Morgan fingerprints, default is 3
            nBits: the bit vector length for Morgan fingerprints, default is 2048

        """

        valid_representations = ["fingerprints", "fragments", "fragprints"]

        if representation == "fingerprints":

            self.features = molecule_fingerprints(self.objects, bond_radius, nBits)

        elif representation == "fragments":

            self.features = molecule_fragments(self.objects)

        elif representation == "fragprints":

            self.features = np.concatenate(
                (
                    molecule_fingerprints(self.objects, bond_radius, nBits),
                    molecule_fragments(self._features),
                ),
                axis=1,
            )

        else:

            raise Exception(
                f"The specified representation choice {representation} is not a valid option."
                f"Choose between {valid_representations}."
            )

    def load_benchmark(self, benchmark, path):
        """
        Loads the SMILES representations and labels for the specified benchmark dataset.

        Args:
            benchmark: the benchmark dataset to be loaded,
            one of ['Photoswitch', 'ESOL', 'FreeSolv', 'Lipophilicity']
            path: the path to the dataset in csv format

        """

        # dictionary specifying which columns to read in for which benchmark dataset

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

        # read in the specified benchmark dataset, if a valid one was chosen

        if benchmark not in benchmarks.keys():

            raise Exception(
                f"The specified benchmark choice ({benchmark}) is not a valid option. "
                f"Choose one of {list(benchmarks.keys())}."
            )

        else:

            df = pd.read_csv(path)
            self.objects = df[benchmarks[benchmark]["features"]].to_list()
            self.labels = df[benchmarks[benchmark]["labels"]].to_numpy()
