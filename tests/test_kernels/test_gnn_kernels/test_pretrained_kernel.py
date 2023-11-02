"""
Tests to check whether the refactored implementation of the code from
Hu et al. Strategies for Pre-training Graph Neural Networks. ICLR 2020
(https://github.com/snap-stanford/pretrain-gnns)
runs and produces node embeddings with the provided pre-trained parameters.
"""

import os

import pytest
import torch
from gauche.dataloader import DataLoaderMP
from gauche.kernels.gnn_kernels.pretrained_kernel import GNN, mol_to_pyg
from rdkit.Chem import MolFromSmiles
from torch_geometric.data import Batch

# load and featurise ESOL benchmark for tests
benchmark_path = os.path.abspath(
    os.path.join(
        os.getcwd(),
        "..",
        "..",
        "..",
        "data",
        "property_prediction",
        "ESOL.csv",
    )
)

benchmark_loader = DataLoaderMP()
benchmark_loader.load_benchmark(benchmark="ESOL", path=benchmark_path)

# get PyTorch Geometric featurisation of molecules
benchmark_mols = [
    mol_to_pyg(MolFromSmiles(smiles)) for smiles in benchmark_loader.features
]

# combine molecules to PyG batch to speed up embedding generation
benchmark_batch = Batch().from_data_list(benchmark_mols)


@pytest.mark.parametrize(
    "gnn_type, pretrain_method",
    [
        ("gcn", "contextpred"),
        ("gin", "attrmasking"),
        ("gin", "contextpred"),
        ("gin", "edgepred"),
        ("gin", "infomax"),
    ],
)
def test_embedding_generation(gnn_type, pretrain_method):
    """
    Tests whether the pretrained parameters for a given GNN type
    and a given pretraining approach can be loaded and used
    to generate node embeddings for a set of ESOL training molecules.
    Args:
        gnn_type: type of GNN to use, either GIN or GCN
        pretrain_method: method used to pretrain parameter sets

    Returns: None

    """

    model = GNN(gnn_type=gnn_type)
    model.load_pretrained(pretrain_method, device=torch.device("cpu"))

    # check that embeddings are properly generated for a single molecule
    node_embed = model(
        x=benchmark_mols[0].x,
        edge_index=benchmark_mols[0].edge_index,
        edge_attr=benchmark_mols[0].edge_attr,
    )

    assert node_embed.shape[0] == benchmark_mols[0].num_nodes
    assert node_embed.shape[1] == model.embed_dim

    # check that embeddings are properly generated for a batch of molecules
    node_embeds = model(
        x=benchmark_batch.x,
        edge_index=benchmark_batch.edge_index,
        edge_attr=benchmark_batch.edge_attr,
    )

    # split node embedding batch back into separate molecules
    node_embeds = torch.tensor_split(
        node_embeds, benchmark_batch.ptr[1:-1], dim=0
    )

    # check that shapes match after embedding and unbatching
    for embed, mol in zip(node_embeds, benchmark_mols):
        assert embed.shape[0] == mol.num_nodes
        assert embed.shape[1] == model.embed_dim