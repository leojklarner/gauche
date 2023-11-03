"""
Contains methods to generate fingerprint representations 
of molecules, chemical reactions and proteins.
"""

from typing import List, Optional

import numpy as np
import pandas as pd
from rdkit.Chem import MolFromSmiles, AllChem, rdMolDescriptors, Descriptors


### ------------------ Reactions ------------------ ###


def one_hot(df: pd.DataFrame) -> np.ndarray:
    """
    Builds reaction representation as a bit vector which indicates whether
    a certain condition, reagent, reactant etc. is present in the reaction.

    :param df: pandas DataFrame with columns representing different
    parameters of the reaction (e.g. reactants, reagents, conditions).
    :type df: pandas DataFrame
    :return: array of shape [len(reaction_smiles), sum(unique values for different columns in df)]
     with one-hot encoding of reactions
    """
    df_ohe = pd.get_dummies(df)
    return df_ohe.to_numpy(dtype=np.float64)


def rxnfp(reaction_smiles: List[str]) -> np.ndarray:
    """
    https://rxn4chemistry.github.io/rxnfp/

    Builds reaction representation as a continuous RXNFP fingerprints.
    :param reaction_smiles: list of reaction smiles
    :type reaction_smiles: list
    :return: array of shape [len(reaction_smiles), 256] with rxnfp featurised reactions

    """

    from rxnfp.transformer_fingerprints import (
        get_default_model_and_tokenizer,
        RXNBERTFingerprintGenerator,
    )

    rxn_model, tokenizer = get_default_model_and_tokenizer()
    rxnfp_generator = RXNBERTFingerprintGenerator(rxn_model, tokenizer)
    rxnfps = [rxnfp_generator.convert(smiles) for smiles in reaction_smiles]
    return np.array(rxnfps, dtype=np.float64)


def drfp(
    reaction_smiles: List[str], nBits: Optional[int] = 2048
) -> np.ndarray:
    """
    https://github.com/reymond-group/drfp

    Builds reaction representation as a binary DRFP fingerprints.
    :param reaction_smiles: list of reaction smiles
    :type reaction_smiles: list
    :return: array of shape [len(reaction_smiles), nBits] with drfp featurised reactions

    """

    from drfp import DrfpEncoder

    fps = DrfpEncoder.encode(reaction_smiles, n_folded_length=nBits)
    return np.array(fps, dtype=np.float64)


### ------------------ Molecules ------------------ ###


def ecfp_fingerprints(
    smiles: List[str],
    bond_radius: Optional[int] = 3,
    nBits: Optional[int] = 2048,
) -> np.ndarray:
    """
    Builds molecular representation as a binary ECFP fingerprints.

    :param smiles: list of molecular smiles
    :type smiles: list
    :param bond_radius: int giving the bond radius for Morgan fingerprints. Default is 3
    :type bond_radius: int
    :param nBits: int giving the bit vector length for Morgan fingerprints. Default is 2048
    :type nBits: int
    :return: array of shape [len(smiles), nBits] with ecfp featurised molecules

    """

    rdkit_mols = [MolFromSmiles(s) for s in smiles]
    fpgen = AllChem.GetMorganGenerator(radius=bond_radius, fpSize=nBits)
    fps = [fpgen.GetFingerprint(mol) for mol in rdkit_mols]
    return np.array(fps)


def fragments(smiles: List[str]) -> np.ndarray:
    """
    Builds molecular representation as a vector of fragment counts.

    :param smiles: list of molecular smiles
    :type smiles: list
    :return: array of shape [len(smiles), 85] with fragment featurised molecules
    """
    # get fragment descriptors from RDKit descriptor list
    # (https://www.rdkit.org/docs/source/rdkit.Chem.Fragments.html)

    frag_descriptors = {
        desc_name: desc_fn
        for desc_name, desc_fn in Descriptors.descList
        if desc_name.startswith("fr_")
    }

    frags = []

    for s in smiles:
        mol = MolFromSmiles(s)
        try:
            features = [frag_descriptors[d](mol) for d in frag_descriptors]
        except:
            raise Exception("molecule {}".format(i) + " is not canonicalised")
        frags.append(features)

    return np.array(frags)


def mqn_features(smiles: List[str]) -> np.ndarray:
    """
    Builds molecular representation as a vector of Molecular Quantum Numbers.

    :param reaction_smiles: list of molecular smiles
    :type reaction_smiles: list
    :return: array of mqn featurised molecules

    """
    molecules = [MolFromSmiles(smile) for smile in smiles]
    mqn_descriptors = [
        rdMolDescriptors.MQNs_(molecule) for molecule in molecules
    ]
    return np.array(mqn_descriptors)
