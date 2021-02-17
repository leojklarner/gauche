"""
Implementation of the abstract data loader class for
protein-ligand binding affinity prediction datasets.
"""

import numpy as np
import pandas as pd
import os
from Bio.PDB import *
from prody import *
import requests
from rdkit.Chem import SDWriter

from gprotorch.dataloader import DataLoader

# functions for splitting pdb and


class DataLoaderLB(DataLoader):
    """
    Implementation of the abstract data loader class for
    protein-ligand binding affinity prediction datasets.
    """

    def __init__(self):
        super(DataLoaderLB, self).__init__()
        self.task = "PL_binding_affinity"
        self._features = None
        self.labels = None

    @property
    def features(self):
        """
        Property for storing features.
        Returns: currently loaded features

        """
        return self._features

    @features.setter
    def features(self, value):
        """
        Setter to initialise or change features.
        Args:
            value: feature data

        """
        self._features = value

    @property
    def labels(self):
        """
        Property for storing labels
        Returns: currently loaded labels

        """
        return self._labels

    @labels.setter
    def labels(self, value):
        """
        Setter to initialise or change labels.
        Args:
            value: label data

        """
        self._labels = value

    def featurize(self, representation, data_dir):
        """
        Reads in the pdb files and calculates the specified representation

        Args:
            representation: the desired representation, one of ["rfscore"]
            data_dir:

        """

        features = pd.DataFrame()



        valid_representations = ["rfscore"]

        if representation == "rfscore":
            pass

        else:
            raise Exception(
                f"The specified representation choice {representation} is not a valid option."
                f"Choose between {valid_representations}."
            )

    def validate(self, drop=True):
        pass

    def download_dataset(self, download_dir, ligand_list=None):
        """
        Downloads the pdb files corresponding to the currently
        loaded pdb codes from the Protein Data Bank.

        Args:
            download_dir: the directory to which to download the files
            ligand_list: list of ligands to download

        """

        for pdb_code in self._features:

            # splitting the PDB entry into proteins and ligands
            pdb = parsePDB(pdb_code)
            protein = pdb.select('protein')
            ligand = pdb.select('not protein and not water')

            # writing the protein part to a separate .pdb file
            output_pdb_name = f"{pdb_code}_protein.pdb"
            writePDB(output_pdb_name, protein)

            # writing the non-water heteroatoms
            if ligand_list:
                output_ligand_name = f"{pdb_code}_{ligand_list[pdb_code]}_ligand.sdf"
                writer = SDWriter(output_ligand_name)
                writer.write()
            else:
                resList = list(set(ligand.getResnames()))
                for res in resList:
                    output_ligand_name = f"{pdb_code}_{res}_ligand.sdf"
                    writer = SDWriter(output_ligand_name)
                    writer.write(res)



        # check whether all files have been downloaded successfully
        downloaded_pdbs = [file[3:-4] for file in os.listdir(download_dir) if file.endswith(".ent")]

        if set(self._features).issubset(set(downloaded_pdbs)):
            print(f"Successfully downloaded all PDB files to {download_dir}.")
        else:
            print("Could not download the following PDB files:")
            failed_pdbs = set(self._features) - set(downloaded_pdbs)
            for failed_pdb in failed_pdbs:
                print(failed_pdb)

    def load_benchmark(self, benchmark, path):
        """Loads features and labels from one of the included benchmark datasets
        and feeds them into the DataLoader. PDB files still need to be downloaded
        before featurisation.

        Args:
            benchmark: the benchmark dataset to be loaded, one of
            [PDBbind_refined]
            path: the path to the dataset in csv format

        """

        benchmarks = {
            "PDBbind_refined": {
                "features": "pdb",
                "labels": "label",
            }
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
    loader = DataLoaderLB()
    path_to_data = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), 'data', 'binding_affinity', 'PDBbind')
    loader.load_benchmark("PDBbind_refined", os.path.join(path_to_data, 'pdbbind_test.csv'))
    loader.download_dataset(path_to_data)

    print(loader)
