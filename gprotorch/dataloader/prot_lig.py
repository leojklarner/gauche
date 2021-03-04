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
import oddt
from oddt.toolkits import rdk
from oddt.scoring.descriptors import binana
from oddt.fingerprints import PLEC

from gprotorch.dataloader import DataLoader

from gprotorch.dataloader.dataloader_utils import \
    read_ligand_expo, get_pdb_components, process_ligand, write_pdb, write_sdf, \
    vina_binana_features, plec_fingerprints


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
        self._pdbcodes = None

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

    def featurize(self, representations, concatenate=True, save_features=True, kwargs=None):
        """
        Reads in the pdb files and calculates the specified representation.
        The representations are segmented into three main groups:
        continuous features, categorical features/fingerprints and graph features.
        Continuous features include: AutoDock Vina, BINANA
        BIANANA: finds protein-ligand atom pairs in close contact and calculates
        numeric features for electrostatic interactions, binding-pocket flexibility,
        hydrophobic contacts, hydrogen bonds, salt bridges, various pi interactions
         Described in Durrant and McCammon, JMGM, Vol. 29, Issue 6, 2011
        PLEC: a combination of ligand and receptor ECFPs of close-contact atoms,
        described in Wojcikowski et al, Bioinformatics Vol. 35, Issue 8, 2019

        Args:
            representations: list of desired representations,
            must belong to the same group, i.e. continuous/fingerprints/graph
            concatenate: whether to concatenate the features to one data frame or return them separately
            save_features: whether to save the computed features
            kwargs: dictionary with specific keywords to update

        """

        if kwargs is None:
            kwargs = {}

        featurisations = {
            'vina': {
                'func': vina_binana_features,
                'args': [self._protein_paths, self._ligand_paths, 'vina']
            },
            'binana': {
                'func': vina_binana_features,
                'args': [self._protein_paths, self._ligand_paths, 'binana']
            },
            'nnscore': {
                'func': vina_binana_features,
                'args': [self._protein_paths, self._ligand_paths, 'all']
            },
            'plec': {
                'func': plec_fingerprints,
                'args': [self._protein_paths, self._ligand_paths, kwargs]
            }
        }

        if not set(representations).issubset(featurisations.keys()):

            raise Exception(f"The specified featurisation choice(s) {set(featurisations.keys())-set(representations)}"
                            f"are not supported. Please choose between {featurisations.keys()}.")

        else:

            result_dfs = []

            for representation in representations:

                feature_func = featurisations[representation]['func']
                feature_args = featurisations[representation]['args']

                result_dfs.append(pd.DataFrame(data=feature_func(feature_args), index=self._features))


            # clean features


            # Drop 'Ipc' from RDKit feature set
            # It generates very large values for larger molecules. https://github.com/rdkit/rdkit/issues/1527
            #rdkit_features = rdkit_features.drop(['Ipc'], axis='columns')

            # drop datapoints for which one or more features are NA
            features = features.replace([np.inf, -np.inf], pd.NA)
            features = features.dropna()

            # keep only points for which all features could be calculated

            # drop features with almost zero variance
            zero_var_cols = features.columns[(features.var() < 1e-2)]
            features = features.drop(labels=zero_var_cols, axis='columns')

            print(features)

    def validate(self, drop=True):
        """
        Check whether all of the provided protein and ligand files can be parsed
        into the rdkit implementation from ODDT.

        Args:
            drop: whether to drop files that cannot be parsed from the file list

        Returns: None

        """
        pass

    def download_dataset(self, download_dir, ligand_codes=None):
        """
        Iterate though the currently loaded PDB codes and download the respective PDB entries.
        Extract the protein and write it into a separate .pdb file.
        Split all (or the specified) heteroatoms, select the most drug-like (or specified) one
        and write it into a separate .sdf file.
        Check which PDB entries (if any) could not be downloaded.

        Args:
            download_dir: the directory to which to download the files
            ligand_codes: the ligand codes of one or more molecules that should be split off as ligands.
            Either None (select the most drug-like molecule for each structure) or
            list/dict of lists/tuples (select the given heteroatom residues for all/the specified PDB entries).

        """

        protein_filenames = []
        ligand_filenames = []

        # set working directory to download_dir, as path handling for the prody
        # functions can be a bit tricky, will be reset to original cwd later on
        original_cwd = os.getcwd()
        os.chdir(download_dir)

        # create a pdb_code:ligand_code dictionary for all pdb codes if a list is passed
        if type(ligand_codes) is list:
            assert len(ligand_codes) == len(self._features), "Ligand code list must have same length as pdb code list."
            ligand_dict = {self._features[i]: ligand_codes[i] for i in range(len(self._features))}

        # or create a pdb_code:"" dictionary of empty strings
        # (in which case most drug-like ligand will be selected)
        # and selectively update it with ligand codes if dict is passed
        else:
            ligand_dict = {self._features[i]: "" for i in range(len(self._features))}
            if type(ligand_codes) is dict:
                # capitalise keys before merging ligand codes with empty dictionary
                ligand_dict.update({k.upper(): v for k, v in ligand_codes.items()})

        # read in the Ligand Expo dataset from the PDB
        expo = read_ligand_expo()

        for pdb_code in self._features:

            print(pdb_code)

            # split the PDB entry into proteins and ligands
            try:
                protein, ligand = get_pdb_components(pdb_code)
            except OSError:
                print(f"Could not download the PDB file for {pdb_code}.")
            else:
                # writing the protein part to a .pdb file
                protein_file = write_pdb(protein, pdb_code)
                protein_filenames.append(protein_file)

                # if a ligand code is specified, proceed with the given code(s)
                if ligand_dict[pdb_code]:
                    # check if dict entry is a list or tuple, in case user enters multiple ligand codes
                    if type(ligand_dict[pdb_code]) in [list, tuple]:
                        sdf_ligands = ligand_dict[pdb_code]
                    else:
                        sdf_ligands = [ligand_dict[pdb_code]]

                # if no ligand code is specified, proceed with all hereroatomic molecules
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
                            # parsing errors happen when atoms have valences != 4 after bond-order augmentation,
                            # such as molecules that are covalently bound to the protein (e.g. covalent ligands
                            # or modified residues that are denoted as hereoatoms) or
                            # ions that are not filtered out by the ProDy ion selector (e.g FE)
                            # all of them are of limited relevance to binding affinity prediction
                            print(f"Could not parse the ligand denoted as {ligand_residue} for the PDB file {pdb_code}.")

                    # save bond-order augmented spatial structure of most drug-like ligand to SDF
                    ligand_filename = write_sdf(max_drug_likeness_mol, pdb_code, max_drug_likeness_res)
                    ligand_filenames.append(ligand_filename)

                else:
                    ligand_filenames.append(None)
                    print(f"Could not find ligands for the PDB entry {pdb_code}.")

        # change the working directory back to the original
        os.chdir(original_cwd)

        # check whether all files have been downloaded successfully
        downloaded_pdbs = [file[:-4].upper() for file in os.listdir(download_dir) if file.endswith(".pdb")]
        print(downloaded_pdbs)

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
            pdb_code_list = df[benchmarks[benchmark]["features"]].to_list()
            self._features = [pdb_code.upper() for pdb_code in pdb_code_list]
            self._labels = df[benchmarks[benchmark]["labels"]].to_numpy()


if __name__ == '__main__':
    loader = DataLoaderLB()
    path_to_data = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), 'data', 'binding_affinity', 'PDBbind')
    loader.load_benchmark("PDBbind_refined", os.path.join(path_to_data, 'pdbbind_test.csv'))
    loader.download_dataset(path_to_data)
    loader.featurize(['all'])

    print(loader)
