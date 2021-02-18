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

from gprotorch.dataloader import DataLoader, read_ligand_expo, get_pdb_components, process_ligand, write_pdb, write_sdf


class DataLoaderLB(DataLoader):
    """
    Implementation of the abstract data loader class for
    protein-ligand binding affinity prediction datasets.
    """

    def __init__(self):
        super(DataLoaderLB, self).__init__()
        self.task = "PL_binding_affinity"
        self._features = None
        self._labels = None
        self._protein_paths = None
        self._ligand_paths = None

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

        protein_filenames = []
        ligand_filenames = []

        # set working directory to download_dir, as path handling for the prody
        # functions can be a bit tricky, will be reset to original cwd later on
        original_cwd = os.getcwd()
        os.chdir(download_dir)

        # read in the Ligand Expo dataset from the PDB
        expo = read_ligand_expo()

        for pdb_code in self._features:

            # split the PDB entry into proteins and ligands
            try:
                protein, ligand = get_pdb_components(pdb_code)
            except OSError:
                print(f"Could not download the PDB file for {pdb_code}.")
            else:
                # writing the protein part to a .pdb file
                protein_file = write_pdb(protein, pdb_code)
                protein_filenames.append(protein_file)

                # if ligand list is specified, save the ligands in there, otw save all
                if ligand_list:
                    sdf_ligands = ligand_list
                else:
                    sdf_ligands = list(set(ligand.getResnames()))

                # check if any ligands were found for the current pdb structure
                if sdf_ligands:

                    max_drug_likeness = 0
                    max_drug_likeness_mol = None
                    max_drug_likeness_res = None

                    for ligand_residue in sdf_ligands:
                        try:
                            # add bond-orders to the spatial file and calculate drug likeness
                            new_mol, drug_likeness = process_ligand(ligand, ligand_residue, expo)

                            # check if drug likeness of current ligand is higher than current max
                            if drug_likeness > max_drug_likeness:
                                max_drug_likeness = drug_likeness
                                max_drug_likeness_mol = new_mol
                                max_drug_likeness_res = ligand_residue

                        except ValueError:
                            # parsing errors happen when atoms have valences != 4 after bond-order augmentation
                            # such as e.g. modified residues that are denoted as hereoatoms or
                            # ions that are not filtered out by the ProDy ion selector (for example FE)
                            print(f"Could not parse the ligand denoted as {ligand_residue} for the PDB file {pdb_code}.")

                    # save bond-order augmented spatial structure of most drug-like ligand to SDF
                    ligand_filename = write_sdf(max_drug_likeness_mol, pdb_code, max_drug_likeness_res)
                    ligand_filenames.append(ligand_filename)

                else:
                    print(f"Could not find ligands for the PDB entry {pdb_code}.")

        # change back the working directory
        os.chdir(original_cwd)

        # check whether all files have been downloaded successfully
        downloaded_pdbs = [file[:-4] for file in os.listdir(download_dir) if file.endswith(".pdb")]

        if set(self._features).issubset(set(downloaded_pdbs)):
            print(f"Successfully downloaded all PDB files to {download_dir}.")
        else:
            print("Could not download the following PDB files:")
            failed_pdbs = set(self._features) - set(downloaded_pdbs)
            for failed_pdb in failed_pdbs:
                print(failed_pdb)

        # save paths to protein and selected ligand files
        self._protein_paths = [os.path.join(download_dir, protname) for protname in protein_filenames]
        self._ligand_paths = [os.path.join(download_dir, ligname) for ligname in ligand_filenames]

        return protein_filenames, ligand_filenames

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
    protein_filenames, ligand_filenames = loader.download_dataset(path_to_data)
    print(protein_filenames)
    print(ligand_filenames)

    protein_filenames, ligand_filenames = loader.download_dataset(path_to_data)
    print(protein_filenames)
    print(ligand_filenames)

    print(loader)
