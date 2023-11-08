"""
Contains methods to generate graph representations 
of molecules, chemical reactions and proteins.
"""

from typing import List, Optional

import networkx as nx


def molecular_graphs(
    smiles: List[str], graphein_config: Optional[bool] = None
) -> List[nx.Graph]:
    """
    Convers a list of SMILES strings into molecular graphs
    using the feautrisation utilities of graphein.

    :param smiles: list of molecular SMILES
    :type smiles: list
    :param graphein_config: graphein configuration object
    :type graphein_config: graphein/config/graphein_config
    :return: list of molecular graphs
    """

    import graphein.molecule as gm

    return [
        gm.construct_graph(smiles=i, config=graphein_config) for i in smiles
    ]
