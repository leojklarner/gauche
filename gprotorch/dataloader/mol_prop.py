"""
Implementation of the abstract data loader class for
molecular property prediction datasets.
"""

import numpy as np
import pandas as pd
from rdkit.Chem import AllChem, Descriptors, MolFromSmiles
from gprotorch.dataloader.dataloader_utils import molecule_fragments, molecule_fingerprints

from gprotorch.dataloader import DataLoader


class DataLoaderMP(DataLoader):
    """
    Instantiation of the abstract data loader class for
    molecular property prediction datasets.
    """

    def __init__(self):
        super(DataLoaderMP, self).__init__()
        self._features = None
        self._labels = None



    def validate(self, drop=True):
        """Checks if the features are valid SMILES strings and (potentially)
        drops the entries that are not.

        Args:
            drop: whether to drop invalid entries

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

    def featurize(self, representation, bond_radius=3, nBits=2048):
        """Transforms SMILES into the specified molecular representation.

        Args:
            representation: the desired molecular representation, one of ["fingerprints", "fragments", "fragprints"]
            bond_radius: int giving the bond radius for Morgan fingerprints. Default is 3
            nBits: int giving the bit vector length for Morgan fingerprints. Default is 2048

        """

        valid_representations = ["fingerprints", "fragments", "fragprints"]

        if representation == "fingerprints":

            self.features = molecule_fingerprints(self._features, bond_radius, nBits)

        elif representation == "fragments":

            self.features = molecule_fragments(self._features)

        elif representation == "fragprints":

            self.features = np.concatenate((molecule_fingerprints(self._features, bond_radius, nBits), molecule_fragments(self._features)), axis=1)

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

        if benchmark not in benchmarks.keys():

            raise Exception(
                f"The specified benchmark choice ({benchmark}) is not a valid option. "
                f"Choose one of {list(benchmarks.keys())}."
            )

        else:

            df = pd.read_csv(path)
            self.features = df[benchmarks[benchmark]["features"]].to_list()
            self.labels = df[benchmarks[benchmark]["labels"]].to_numpy()


if __name__ == '__main__':
    import os
    loader = DataLoaderMP()
    path_to_data = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), "data", "property_prediction", "ESOL.csv")
    loader.load_benchmark("ESOL", path_to_data)
    loader.featurize("fragments")
    print()