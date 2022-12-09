import numpy as np
from rdkit.Chem import MolFromSmiles, AllChem, Descriptors
import pandas as pd
from rxnfp.transformer_fingerprints import (
    get_default_model_and_tokenizer,
    RXNBERTFingerprintGenerator,
)
from drfp import DrfpEncoder
from sklearn.feature_extraction.text import CountVectorizer
import selfies as sf
import graphein.molecule as gm
from rdkit.Chem import rdMolDescriptors


# Reactions
def one_hot(df):
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


def rxnfp(reaction_smiles):
    """
    https://rxn4chemistry.github.io/rxnfp/

    Builds reaction representation as a continuous RXNFP fingerprints.
    :param reaction_smiles: list of reaction smiles
    :type reaction_smiles: list
    :return: array of shape [len(reaction_smiles), 256] with rxnfp featurised reactions

    """
    rxn_model, tokenizer = get_default_model_and_tokenizer()
    rxnfp_generator = RXNBERTFingerprintGenerator(rxn_model, tokenizer)
    rxnfps = [rxnfp_generator.convert(smile) for smile in reaction_smiles]
    return np.array(rxnfps, dtype=np.float64)


def drfp(reaction_smiles, nBits=2048):
    """
    https://github.com/reymond-group/drfp

    Builds reaction representation as a binary DRFP fingerprints.
    :param reaction_smiles: list of reaction smiles
    :type reaction_smiles: list
    :return: array of shape [len(reaction_smiles), nBits] with drfp featurised reactions

    """
    fps = DrfpEncoder.encode(reaction_smiles, n_folded_length=nBits)
    return np.asarray(fps, dtype=np.float64)


# Molecules
def fingerprints(smiles, bond_radius=3, nBits=2048):
    rdkit_mols = [MolFromSmiles(smiles) for smiles in smiles]
    fps = [
        AllChem.GetMorganFingerprintAsBitVect(mol, bond_radius, nBits=nBits)
        for mol in rdkit_mols
    ]
    return np.asarray(fps)


# auxiliary function to calculate the fragment representation of a molecule
def fragments(smiles):
    # descList[115:] contains fragment-based features only
    # (https://www.rdkit.org/docs/source/rdkit.Chem.Fragments.html)
    # Update: in the new RDKit version the indices are [124:]
    fragments = {d[0]: d[1] for d in Descriptors.descList[124:]}
    frags = np.zeros((len(smiles), len(fragments)))
    for i in range(len(smiles)):
        mol = MolFromSmiles(smiles[i])
        try:
            features = [fragments[d](mol) for d in fragments]
        except:
            raise Exception("molecule {}".format(i) + " is not canonicalised")
        frags[i, :] = features

    return frags


# auxiliary function to calculate bag of character representation of a molecular string
def bag_of_characters(smiles, max_ngram=5, selfies=False):
    if selfies:  # convert SMILES to SELFIES
        strings = [sf.encoder(smiles[i]) for i in range(len(smiles))]
    else:  # otherwise stick with SMILES
        strings = smiles

    # extract bag of character (boc) representation from strings
    cv = CountVectorizer(
        ngram_range=(1, max_ngram), analyzer="char", lowercase=False
    )
    return cv.fit_transform(strings).toarray()


def graphs(smiles, graphein_config=None):
    return [
        gm.construct_graph(smiles=i, config=graphein_config) for i in smiles
    ]


def mqn_features(smiles):
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
    return np.asarray(mqn_descriptors)
