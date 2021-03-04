"""
A script containing utilities for the different data loader classes.
"""

import sys
from prody import *
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Descriptors import qed
from io import StringIO
import requests
import os
import oddt
from oddt.toolkits import rdk
from oddt.scoring.descriptors import binana
from oddt.fingerprints import PLEC
from rdkit.Chem import AllChem, Descriptors, MolFromSmiles

# -------------------------------- PDB scraping utilities --------------------------------

# functions adapted from Pat Walters (https://gist.github.com/PatWalters/3c0a483c030a2c75cb22c4234f206973)
# that split a PDB entry into a .pdb file of the protein and a .sdf file of the ligand(s)


def read_ligand_expo():
    """
    Read PDB Ligand Expo data, try to find a file called
    Components-smiles-stereo-oe.smi in the current directory.
    If you can't find the file, grab it from the RCSB

    Returns: Ligand Expo as a dictionary with ligand id as the key

    """

    file_name = "Components-smiles-stereo-oe.smi"

    # read file if it already exists
    try:
        df = pd.read_csv(file_name, sep="\t",
                         header=None,
                         names=["SMILES", "ID", "Name"])

    # otherwise download it and read it in afterwards
    except FileNotFoundError:
        url = f"http://ligand-expo.rcsb.org/dictionaries/{file_name}"
        r = requests.get(url, allow_redirects=True)
        open('Components-smiles-stereo-oe.smi', 'wb').write(r.content)
        df = pd.read_csv(file_name, sep="\t",
                         header=None,
                         names=["SMILES", "ID", "Name"])
    df.set_index("ID", inplace=True)

    return df.to_dict()


def get_pdb_components(pdb_id):
    """
    Split a protein-ligand pdb into protein and ligand components.
    A useful dictionary of possible flags can be found here
    http://prody.csb.pitt.edu/manual/reference/atomic/flags.html#flags.

    Args:
        pdb_id: 4-letter pdb code

    Returns: tuple of ProDy Selections for proteins and ligands

    """

    pdb = parsePDB(pdb_id)
    protein = pdb.select('protein')
    ligand = pdb.select('not protein and not water and not ion')
    return protein, ligand


def process_ligand(ligand, res_name, expo_dict):
    """
    Add bond orders to a pdb ligand through the following process
    1. Select the ligand component with name "res_name"
    2. Get the corresponding SMILES from the Ligand Expo dictionary
    3. Create a template molecule from the SMILES in step 2
    4. Write the PDB file to a stream
    5. Read the stream into an RDKit molecule
    6. Assign the bond orders from the template from step 3

    Args:
        ligand: ligand as generated by prody
        res_name: residue name of ligand to extract
        expo_dict: dictionary with LigandExpo

    Returns: molecule with bond orders assigned

    """

    output = StringIO()

    # select res_name residue
    sub_mol = ligand.select(f"resname {res_name}")

    # extract corresponding SMILES and read it into rdkit
    sub_smiles = expo_dict['SMILES'][res_name]
    template = AllChem.MolFromSmiles(sub_smiles)

    # calculate the rdkit drug likeness descripot for the ligand expo template
    drug_likeness = qed(template)

    # stream selected ligand
    writePDBStream(output, sub_mol)
    pdb_string = output.getvalue()

    # add bond orders
    rd_mol = AllChem.MolFromPDBBlock(pdb_string)
    new_mol = AllChem.AssignBondOrdersFromTemplate(template, rd_mol)

    return new_mol, drug_likeness


def write_pdb(protein, pdb_name, overwrite=False):
    """
    Write a prody protein to a pdb file.

    Args:
        protein: protein object from prody
        pdb_name: base name for the pdb file
        overwrite: whether to overwrite an already existing file

    Returns: file name

    """

    output_protein_name = f"{pdb_name}.pdb"

    # check if file already exists
    if not overwrite and os.path.isfile(os.path.join(os.getcwd(), output_protein_name)):
        print(f"The protein file for the PDB entry {pdb_name} already exists. "
              f"Set overwrite=True if you want to overwrite it.")
    else:
        writePDB(output_protein_name, protein)
        print(f"Wrote the protein file for the PDB entry {output_protein_name}.")

    return output_protein_name


def write_sdf(new_mol, pdb_name, res_name, overwrite=False):
    """
    Write an RDKit molecule to an SD file.

    Args:
        new_mol: the molecule to write to a file
        pdb_name: the PDB entry from which it was extracted
        res_name: its residue identifier in the PDB entry
        overwrite: whether to overwrite an already existing file

    Returns: file name

    """

    outfile_ligand_name = f"{pdb_name}_{res_name}_ligand.sdf"

    # check if file already exists
    if not overwrite and os.path.isfile(os.path.join(os.getcwd(), outfile_ligand_name)):
        print(f"The ligand file for the ligand {res_name} in PDB entry {pdb_name} already exists. "
              f"Set overwrite=True if you want to overwrite it.")
    else:
        writer = Chem.SDWriter(outfile_ligand_name)
        writer.write(new_mol)
        print(f"wrote the ligand file for the ligand {res_name} in PDB entry {pdb_name}.")

    return outfile_ligand_name


# -------------------------------- featurisation utilities --------------------------------

def molecule_fingerprints(input_mols, bond_radius, nBits):
    """
    Auxiliary function to transform the loaded features to a fingerprint representation

    Returns: numpy array of features in fingerprint representation

    """

    rdkit_mols = [MolFromSmiles(smiles) for smiles in input_mols]
    fps = [
        AllChem.GetMorganFingerprintAsBitVect(mol, bond_radius, nBits=nBits)
        for mol in rdkit_mols
    ]

    return np.asarray(fps)


def molecule_fragments(input_mols):
    """
    Auxiliary function to transform the loaded features to a fragment representation

    Returns: numpy array of features in fragment representation

    """

    # extract all fragment rdkit descriptors
    # (https://www.rdkit.org/docs/source/rdkit.Chem.Fragments.html)
    fragList = [desc for desc in Descriptors.descList if desc[0].startswith('fr_')]

    fragments = {d[0]: d[1] for d in fragList}
    frags = np.zeros((len(input_mols), len(fragments)))
    for i in range(len(input_mols)):
        mol = MolFromSmiles(input_mols[i])
        try:
            features = [fragments[d](mol) for d in fragments]
        except:
            raise Exception("molecule {}".format(i) + " is not canonicalised")
        frags[i, :] = features

    return frags


def vina_binana_features(protein_paths, ligand_paths, feature_group):
    """
    Calculates the AutoDock Vina and/or BINANA features for the given
    protein and ligand, as implemented in ODDT.

    Args:
        protein_paths: list of paths to the protein .pdb files
        ligand_paths: list of paths to the ligand .sdf files
        feature_group: whether to extract 'vina', 'binana' or 'all' features

    Returns: list of specified features

    """

    results = []

    for protein_path, ligand_path in zip(protein_paths, ligand_paths):

        # initialise protein and ligand
        protein = next(rdk.readfile('pdb', protein_path))
        protein.protein = True
        ligand = next(rdk.readfile('sdf', ligand_path))

        # initialise binana engine
        binana_engine = binana.binana_descriptor(protein)

        features_all = {name: value for name, value in zip(binana_engine.titles, binana_engine.build([ligand])[0])}

        # the ODDT names for the VINA features, missing 'num_rotors'
        vina_feature_names = ['vina_gauss1', 'vina_gauss2', 'vina_hydrogen',
                              'vina_hydrophobic', 'vina_repulsion', 'vina_num_rotors']

        # NOTE: the feature 'num_rotors' is included in both the Vina and BINANA feature sets
        # it will be renamed to 'vina_num_rotors' or 'binana_num_rotors' when calculating the
        # features separately and will be renamed to 'vina_num_rotors' when calculating them both
        features_all['vina_num_rotors'] = features_all.pop('num_rotors')

        # split off the vina feature set
        features_vina = {k: v for k, v in features_all.items() if k in vina_feature_names}

        # split off the binana feature set and add 'binana_' to the feature names
        features_binana = {'binana_' + k: v for k, v in features_all.items() if k not in vina_feature_names}
        features_binana['binana_num_rotors'] = features_all['vina_num_rotors']

        if feature_group == 'vina':

            # extract AutoDock Vina features
            result = features_vina

        elif feature_group == 'binana':

            # extract BINANA features
            result = features_binana

        elif feature_group == 'all':

            # combine both feature sets (to keep the 'binana_' prefix for binana features)
            result = {**features_vina, **features_binana}
            result.pop('binana_num_rotors')

        else:
            raise Exception(
                f"Internal error: feature selection {feature_group} not in "
                f"['vina','binana','all'] for vina_binana_features function."
            )

        results.append(result)

    return results


def plec_fingerprints(protein_paths, ligand_paths, **params):
    """
    Calculates the protein-ligand extended conncetivity fingerprints
    for the given proteins. The arguments and their standard values are
    plec_depth_ligand=2, plec_depth_protein=4,
    plec_distance_cutoff=4.5, plec_size=16384

    Args:
        protein_paths: list of paths to the protein .pdb files
        ligand_paths: list of paths to the ligand .sdf files
        plec_params: custom parameters passed to calculate PLEC FPs

    Returns: list of specified features

    """

    results = []

    for protein_path, ligand_path in zip(protein_paths, ligand_paths):

        # set standard parameters
        plec_params = {
            'plec_depth_ligand': 2,
            'plec_depth_protein': 4,
            'plec_distance_cutoff': 4.5,
            'plec_size': 16384,
        }

        # if parameter changes are passed through kwargs, apply them
        plec_params = {k: params[k] for k, v in plec_params if k in params}

        protein = next(rdk.readfile('pdb', protein_path))
        protein.protein = True
        ligand = next(rdk.readfile('sdf', ligand_path))

        result = PLEC(
            ligand=ligand,
            protein=protein,
            depth_ligand=plec_params['plec_depth_ligand'],
            depth_protein=plec_params['plec_depth_protein'],
            distance_cutoff=plec_params['plec_distance_cutoff'],
            size=plec_params['plec_size']
        )

        results.append(result)

    return results