import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import rdChemReactions, MolFromSmiles, AllChem, Descriptors
import pandas as pd
from rxnfp.transformer_fingerprints import (
    get_default_model_and_tokenizer,
    RXNBERTFingerprintGenerator,
)
from drfp import DrfpEncoder
from sklearn.feature_extraction.text import CountVectorizer
from transformers import AutoModelWithLMHead, AutoTokenizer
import selfies as sf
import graphein.molecule as gm


# Reactions
def rxnfp(reaction_smiles):
    rxn_model, tokenizer = get_default_model_and_tokenizer()
    rxnfp_generator = RXNBERTFingerprintGenerator(rxn_model, tokenizer)
    rxnfps = [rxnfp_generator.convert(smile) for smile in reaction_smiles]
    return np.array(rxnfps, dtype=float)


def one_hot(df):
    df_ohe = pd.get_dummies(df)
    return df_ohe.to_numpy(dtype=float)


def drfp(reaction_smiles):
    fps = DrfpEncoder.encode(reaction_smiles)
    return np.asarray(fps, dtype=float)


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


def chemberta_features(smiles):
    # any model weights from the link above will work here
    model = AutoModelWithLMHead.from_pretrained(
        "seyonec/ChemBERTa-zinc-base-v1"
    )
    tokenizer = AutoTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
    tokenized_smiles = [
        tokenizer(smile, return_tensors="pt") for smile in smiles
    ]
    outputs = [
        model(
            input_ids=tokenized_smile["input_ids"],
            attention_mask=tokenized_smile["attention_mask"],
            output_hidden_states=True,
        )
        for tokenized_smile in tokenized_smiles
    ]

    embeddings = torch.cat(
        [output["hidden_states"][0].sum(axis=1) for output in outputs], axis=0
    )
    return embeddings.detach().numpy()


def graphs(smiles, graphein_config=None):
    return [
        gm.construct_graph(smiles=i, config=graphein_config) for i in smiles
    ]


def random_features():
    # todo random continous random bit vector
    pass
