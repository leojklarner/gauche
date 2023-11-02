"""
Contains methods to generate string representations 
of molecules, chemical reactions and proteins.
"""

import numpy as np
from typing import List, Optional
from sklearn.feature_extraction.text import CountVectorizer


def bag_of_characters(
    strings: List[str], max_ngram: Optional[int] = 5, selfies: Optional[bool] = False
) -> np.ndarray:
    """
    Featursises any string representation (molecules/chemical reactions/proteins) into a bag of characters (boc) representation.

    :param strings: list of molecular strings
    :type strings: list
    :param max_ngram: maximum length of ngrams to be considered
    :type max_ngram: int
    :param selfies: when using molecular SMILES, optionally convert them into SELFIES
    :type selfies: bool

    :return: array of shape [len(strings), n_features] with bag of characters featurised molecules
    """

    assert isinstance(strings, list), "strings must be a list of strings"
    assert (
        isinstance(max_ngram, int) and max_ngram > 0
    ), "max_ngram must be a positive integer"

    if selfies:
        # when using molecular SMILES, optionally convert them into SELFIES
        import selfies as sf

        strings = [sf.encoder(strings[i]) for i in range(len(strings))]

    # extract bag of character (boc) representation from strings
    cv = CountVectorizer(ngram_range=(1, max_ngram), analyzer="char", lowercase=False)
    return cv.fit_transform(strings).toarray()
