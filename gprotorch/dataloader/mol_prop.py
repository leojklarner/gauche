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

    def _validate(self, data):
        """Checks which of the given entries are valid SMILES representations and
        splits them into valid and invalid ones.

        Args:
            data: the data to be checked

        Returns: (valid, invalid) tuple of valid and invalid SMILES representations.

        """

        valid = []
        invalid = []

        # iterate through the given data
        for i in range(len(data)):

            # try to convert each SMILES to an rdkit molecule
            mol = MolFromSmiles(data[i])

            # denote molecule as either valid or invalid
            if mol is None:
                invalid.append(data[i])
            else:
                valid.append(data[i])

        return valid, invalid

    def featurize(self, representation, bond_radius=3, nBits=2048):
        """Transforms SMILES into the specified molecular representation.

        Args:
            representation: the desired molecular representation, one of ["fingerprints", "fragments", "fragprints"]
            bond_radius: int giving the bond radius for Morgan fingerprints. Default is 3
            nBits: int giving the bit vector length for Morgan fingerprints. Default is 2048

        """

        valid_representations = ["fingerprints", "fragments", "fragprints"]

        if representation == "fingerprints":

            self.features = molecule_fingerprints(self.objects, bond_radius, nBits)

        elif representation == "fragments":

            self.features = molecule_fragments(self.objects)

        elif representation == "fragprints":

            self.features = np.concatenate((molecule_fingerprints(self.objects, bond_radius, nBits), molecule_fragments(self._features)), axis=1)

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
            self.objects = df[benchmarks[benchmark]["features"]].to_list()
            self.labels = df[benchmarks[benchmark]["labels"]].to_numpy()


if __name__ == '__main__':
    import os
    loader = DataLoaderMP()
    path_to_data = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), "data", "property_prediction", "ESOL.csv")
    loader.load_benchmark("ESOL", path_to_data)
    loader.featurize("fragments")
    print()
