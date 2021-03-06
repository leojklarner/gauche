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
from rdkit.Chem.PandasTools import LoadSDF
import re

from gprotorch.dataloader import DataLoader

from gprotorch.dataloader.dataloader_utils import *


class DataLoaderLB(DataLoader):
    """
    Implementation of the abstract data loader class for
    protein-ligand binding affinity prediction datasets.
    """

    def __init__(self, validate_internal_rep):
        super(DataLoaderLB, self).__init__(validate_internal_rep=validate_internal_rep)
        self._pdb_codes = None

    @property
    def pdb_codes(self):
        """
        Property for storing a list of pdb codes.

        Returns: currently loaded pdb codes

        """
        return self._pdb_codes

    @pdb_codes.setter
    def pdb_codes(self, value):
        """
        Method to load a list of of pdb codes.

        Args:
            value: list of of pdb codes

        """
        self._pdb_codes = value

    def featurize(self, representations, concatenate=True, save_features=True, params=None):
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
            params: dictionary with specific keywords to update

        """

        if type(representations) is not list:
            representations = [representations]

        if params is None:
            params = {}

        featurisations = {
            'vina': {
                'func': vina_features,
                'args': [self.objects]
            },
            'binana': {
                'func': binana_nnscore_features,
                'args': [self.objects, 'binana']
            },
            'nnscore_v2': {
                'func': binana_nnscore_features,
                'args': [self.objects, 'nnscorev2']
            },
            'plec': {
                'func': plec_fingerprints,
                'args': [self.objects, params]
            },
            'rfscore_v3': {
                'func': rfscore_descriptors,
                'args': [self.objects, params]
            },
            'fragments': {
                'func': molecule_fragments,
                'args': sdf_to_smiles(
                    [pdb_codes for pdb_codes, _, _ in self.objects],
                    [lig_path for _, _, lig_path in self.objects]
                )
            },
        }

        if not set(representations).issubset(set(featurisations.keys())):
            print(set(representations))
            print(set(featurisations.keys()))

            raise Exception(f"The specified featurisation choice(s) {set(representations)-set(featurisations.keys())} "
                            f"are not supported. Please choose between {set(featurisations.keys())}.")

        else:

            result_dfs = []

            for representation in representations:

                feature_func = featurisations[representation]['func']
                feature_args = featurisations[representation]['args']

                result_df = feature_func(*feature_args)

                # remove entries with NA features
                result_df = result_df.replace([np.inf, -np.inf], pd.NA)
                result_df = result_df.dropna()

                # remove features with low variance
                # shouldn't cause any complications with fingerprints that
                # are analysed via the Tanimoto index.
                zero_var_cols = result_df.columns[(result_df.var() < 1e-2)]
                result_df = result_df.drop(labels=zero_var_cols, axis='columns')

                result_dfs.append(result_df)

                print(f'Calculated {representation} features.')

        # remove entries for which not all features could be calculated
        indices_to_use = set.intersection(*[set(df.index.to_list()) for df in result_dfs])
        result_dfs = [df.loc[indices_to_use] for df in result_dfs]

        # concatenate them if needed
        if concatenate:
            result_dfs = pd.concat(result_dfs, axis=1)

        self.features = result_dfs

    def _validate(self, protein_ligand_path_list):
        """
        Check whether all of the provided protein and ligand files can be parsed
        into the rdkit implementation from ODDT.

        Args:
            protein_ligand_path_list: a list of (protein_path, ligand_path) tuples

        Returns: (valid, invalid) tuple of parseable and inparseable pdb and sdf files.

        """

        valid = []
        invalid = []

        # for each of the given path pairs
        for protein_ligand_pair in protein_ligand_path_list:

            # try to parse them into oddt pdb/sdf files and add them to valid/invalid
            # valid/invalid list depending on result

            try:
                next(oddt.toolkit.readfile('pdb', protein_ligand_pair[1]))
                next(oddt.toolkit.readfile('sdf', protein_ligand_pair[2]))

                valid.append(protein_ligand_pair)

            except StopIteration:

                invalid.append(protein_ligand_pair)

        return valid, invalid

    def download_dataset(self, download_dir, ligand_codes=None, keep_source=False, solvent_blacklist=True):
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
            keep_source: whether to keep the .pdb.gz file containing both the protein and ligend
            or delete it and only keep the split files
            solvent_blacklist: whether to exclude commonly used solvents from consideration

        Returns: None

        """

        results = []

        if solvent_blacklist:
            # load a list of commonly used solvents to exclude during automatic
            # residue selection, in case of parsing errors for the actual ligands
            with open('solvent_blacklist.txt', 'r') as f:
                blacklist = set(f.read().splitlines())
        else:
            blacklist = {}

        # set working directory to download_dir, as path handling for the prody
        # functions can be a bit tricky, will be reset to original cwd later on
        original_cwd = os.getcwd()
        os.chdir(download_dir)

        # create a pdb_code:ligand_code dictionary for all pdb codes if a list is passed
        if type(ligand_codes) is list:
            assert len(ligand_codes) == len(self.pdb_codes), "Ligand code list must have same length as pdb code list."
            ligand_dict = {self.pdb_codes[i]: ligand_codes[i] for i in range(len(self.pdb_codes))}

        # or create a pdb_code:"" dictionary of empty strings
        # (in which case most drug-like ligand will be selected)
        # and selectively update it with ligand codes if dict is passed
        else:
            ligand_dict = {self.pdb_codes[i]: "" for i in range(len(self.pdb_codes))}
            if type(ligand_codes) is dict:
                # capitalise keys before merging ligand codes with empty dictionary
                ligand_dict.update({k.upper(): v for k, v in ligand_codes.items()})

        # read in the Ligand Expo dataset from the PDB
        expo = read_ligand_expo()

        # list files for checking which files already exist
        file_list = os.listdir(download_dir)

        for pdb_code in self.pdb_codes:

            if os.path.isfile(f"{pdb_code}_protein.pdb"):
                print(f"Protein file for {pdb_code} already exists, ligand extraction is skipped.")

                try:
                    # find ligand file
                    pattern = re.compile(pdb_code+r'_..._ligand\.sdf')
                    ligfile = list(filter(pattern.match, file_list))[0]

                    results.append(
                        (
                            'pdb_code',
                            os.path.join(download_dir, f"{pdb_code}_protein.pdb"),
                            os.path.join(download_dir, ligfile)
                        )
                    )

                except IndexError:
                    print(f"Found protein file for entry {pdb_code}, but no ligand file. "
                          f"Protein will be deleted.")
                    os.remove(os.path.join(download_dir, f"{pdb_code}_protein.pdb"))

            else:
                # split the PDB entry into proteins and ligands
                try:
                    protein, ligand = get_pdb_components(pdb_code)
                except OSError:
                    print(f"Could not download the PDB file for {pdb_code}.")
                else:

                    if protein is not None and ligand is not None:

                        # writing the protein part to a .pdb file
                        protein_filename = write_pdb(protein, pdb_code)

                        # if a ligand code is specified, proceed with the given code(s)
                        if ligand_dict[pdb_code]:
                            # check if dict entry is a list or tuple, in case user enters multiple ligand codes
                            if type(ligand_dict[pdb_code]) in [list, tuple]:
                                sdf_ligands = ligand_dict[pdb_code]
                            else:
                                sdf_ligands = [ligand_dict[pdb_code]]

                        # if no ligand code is specified, proceed with all hereroatomic molecules
                        else:
                            sdf_ligands = list(set(ligand.getResnames())-blacklist)

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

                                except (ValueError, TypeError) as e:
                                    # parsing errors happen when atoms have valences != 4 after bond-order augmentation,
                                    # such as molecules that are covalently bound to the protein (e.g. covalent ligands
                                    # or modified residues that are denoted as hereoatoms) or
                                    # ions that are not filtered out by the ProDy ion selector (e.g FE)
                                    # all of them are of limited relevance to binding affinity prediction
                                    print(f"Could not parse the heteroatom denoted as {ligand_residue} for the PDB file {pdb_code}.")

                            if max_drug_likeness_mol is not None and max_drug_likeness_res is not None:

                                # save bond-order augmented spatial structure of most drug-like ligand to SDF
                                ligand_filename = write_sdf(max_drug_likeness_mol, pdb_code, max_drug_likeness_res)

                                # a pdb file with a suitable ligand was found, add them to results
                                results.append((
                                    pdb_code,
                                    os.path.join(download_dir, protein_filename),
                                    os.path.join(download_dir, ligand_filename)))

                            else:
                                print(f"Could not parse any ligands for PDB entry {pdb_code}.")

                        else:
                            print(f"Could not find ligands for the PDB entry {pdb_code}.")

                    else:
                        print(f"Could not separate protein and ligand for PDB entry {pdb_code}.")

                # delete the .pdb.gz source file, if selected
                if not keep_source:
                    os.remove(f'{pdb_code}.pdb.gz')

        # change the working directory back to the original
        os.chdir(original_cwd)

        # check whether all files have been downloaded successfully
        downloaded_pdbs = [file[:4].upper() for file in os.listdir(download_dir) if file.endswith("_protein.pdb")]

        if set(self.pdb_codes).issubset(set(downloaded_pdbs)):
            print(f"Successfully processed all PDB files to {download_dir}.")
        else:
            failed_pdbs = set(self.pdb_codes) - set(downloaded_pdbs)
            print(f"Could not process the following {failed_pdbs} PDB files:")
            print(failed_pdbs)

        # save paths to protein and selected ligand files
        self.objects = results

    def load_benchmark(self, benchmark, path):
        """Loads pdb codes and labels from one of the included benchmark datasets
        and feeds them into the DataLoader. PDB files still need to be downloaded
        before featurisation.

        Args:
            benchmark: the benchmark dataset to be loaded, one of
            [PDBbind_refined]
            path: the path to the dataset in csv format

        """

        if benchmark == 'PDBbind_refined':

            index_df = pd.read_csv(
                os.path.join(path, 'index', 'INDEX_refined_data.2019'),
                delim_whitespace=True,
                skiprows=6,
                header=None,
                index_col=False,
                names=['pdb_code', 'resolution', 'release_year', 'label', 'Kd/Ki', 'slashes', 'reference', 'ligand_name']
            )

            self.pdb_codes = index_df['pdb_code'].to_list()

            index_df = index_df.set_index(index_df['pdb_code'])
            self.labels = index_df['label']

            protein_paths = [
                os.path.join(path, pdb_code, f'{pdb_code}_pocket.pdb') for pdb_code in self.pdb_codes
            ]
            ligand_paths = [
                os.path.join(path, pdb_code, f'{pdb_code}_ligand.sdf') for pdb_code in self.pdb_codes
            ]

            self.objects = list(zip(self.pdb_codes, protein_paths, ligand_paths))

        else:

            raise Exception(
                f"The specified benchmark choice ({benchmark}) is not a valid option. "
                f"Choose one of [PDBbind_refined]."
            )

    def save_objects(self, file_path):
        """
        Saves the paths to the downloaded .pdb protein and .sdf ligand files,
        as well as the labels. Only saves labels for valid protein-ligand pairs
        (might be fewer than initial list in case of parsing errors).

        Args:
            file_path: the path to which the file should be saved

        Returns: None

        """

        with open(file_path, 'w') as file:
            file.write('pdb_code,protein_path,ligand_path,label\n')
            for pair in self.objects:
                file.write(f'{pair[0]},{pair[1]},{pair[2]},{self.labels.loc[pair[0]].to_numpy()[0]}\n')

    def load_objects(self, file_path):
        """
        Loads the log created by the save_paths method to load all pdb_codes
        and the corresponding protein and ligand file paths, as well as the labels.

        Args:
            file_path: file path of the saved paths

        Returns: None

        """

        df = pd.read_csv(file_path, index_col='pdb_code')
        self.objects = list(
            zip(
                df.index,
                df['protein_path'],
                df['ligand_path']
            )
        )

        self.labels = df['label']

    def save_features(self, file_paths):
        """
        A method for saving the calculated features to a .csv. Creates multiple
        .csv files if features are not concatenated.

        Args:
            file_paths: list of file paths to store the feature dataframes to

        Returns: None

        """

        assert self.features is not None, "Tried to save features, but none were found."

        if type(self.features) is list:
            assert len(self.features) == len(file_paths), \
                "Number of given file paths does not match number of feature dataframes. " \
                "Check whether features were concatenated during featurisation."

            for df, filepath in zip(self.features, file_paths):
                df = df.join(self.labels, how='inner')
                df.to_csv(filepath, index_label='pdb_codes')

        else:

            df = self.features.join(self.labels, how='inner')
            df.to_csv(file_paths, index_label='pdb_codes')

    def load_features(self, file_path):
        """
        Loads a single dataframe containing features and labels.

        Args:
            file_path: a single file path to the DataFrame to be loaded

        Returns: None

        """

        df = pd.read_csv(file_path, index_col='pdb_codes')
        self.features = df.loc[:, df.columns != 'label']
        self.labels = df[['label']]


if __name__ == '__main__':
    loader = DataLoaderLB()
    path_to_data = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), 'data', 'binding_affinity', 'PDBbind')
    loader.load_benchmark("PDBbind_refined", os.path.join(path_to_data, 'pdbbind_test.csv'))
    loader.download_dataset(path_to_data)
    loader.featurize(['vina'])
    loader.save_features('features.csv')
    loader.load_features('features.csv')

    print(loader)
