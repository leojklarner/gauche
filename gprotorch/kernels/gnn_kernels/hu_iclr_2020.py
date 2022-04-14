"""
Reformatted and cleaned up GNN modules from Hu et al.
Strategies for Pre-training Graph Neural Networks. ICLR 2020
(https://github.com/snap-stanford/pretrain-gnns)

Author: Leo Klarner (https://github.com/leojklarner), April 2022
"""

import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops
import torch.nn.functional as F
from torch_scatter import scatter_add


num_atom_type = 120  # including the extra mask tokens
num_chirality_tag = 3

num_bond_type = 6  # including aromatic and self-loop edge, and extra masked tokens
num_bond_direction = 3
self_loop_token = 4  # bond type for self-loop edge
masked_bond_token = 5  # bond type for masked edges


class GINConv(MessagePassing):
    """
    Extension of the Graph Isomorphism Network to incorporate
    edge information by concatenating edge embeddings.
    """

    def __init__(self, emb_dim, aggr="add"):
        super(GINConv, self).__init__()

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(emb_dim, 2 * emb_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(2 * emb_dim, emb_dim)
        )

        self.edge_embedding1 = torch.nn.Embedding(num_bond_type, emb_dim)
        self.edge_embedding2 = torch.nn.Embedding(num_bond_direction, emb_dim)
        torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)

        self.aggr = aggr

    def forward(self, x, edge_index, edge_attr):

        # add self loops to edge index
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # update edge attributes to represent self-loop edges
        self_loop_attr = torch.zeros(x.size(0), 2)
        self_loop_attr[:, 0] = self_loop_token
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)

        # generate edge embeddings and propagate
        edge_embeddings = self.edge_embedding1(edge_attr[:, 0]) + self.edge_embedding2(edge_attr[:, 1])
        return self.propagate(edge_index, x=x, edge_attr=edge_embeddings)

    def message(self, x_j, edge_attr):
        return x_j + edge_attr

    def update(self, aggr_out):
        return self.mlp(aggr_out)


class GCNConv(MessagePassing):
    """
    Extension of the Graph Convolutional Network to incorporate
    edge information by concatenating edge embeddings.
    """

    def __init__(self, emb_dim, aggr="add"):
        super(GCNConv, self).__init__(aggr=aggr)

        self.linear = torch.nn.Linear(emb_dim, emb_dim)
        self.edge_embedding1 = torch.nn.Embedding(num_bond_type, emb_dim)
        self.edge_embedding2 = torch.nn.Embedding(num_bond_direction, emb_dim)

        torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)

    @staticmethod
    def norm(edge_index, num_nodes, dtype):

        # symmetrically normalise edge weights
        edge_weight = torch.ones((edge_index.size(1),), dtype=dtype, device=edge_index.device)
        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        return deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def forward(self, x, edge_index, edge_attr):

        # add self loops to edge index
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), 2)
        self_loop_attr[:, 0] = self_loop_token
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)

        # generate edge embeddings and norm, and propagate
        edge_embeddings = self.edge_embedding1(edge_attr[:, 0]) + self.edge_embedding2(edge_attr[:, 1])
        return self.propagate(
            edge_index,
            x=self.linear(x),
            edge_attr=edge_embeddings,
            norm=self.norm(edge_index, x.size(0), x.dtype)
        )

    def message(self, x_j, edge_attr, norm):
        return norm.view(-1, 1) * (x_j + edge_attr)


class GNN(torch.nn.Module):
    """
    Combine multiple GNN layers into a network.
    """

    def __init__(self, args):

        super(GNN, self).__init__()
        self.args = args

        if self.args.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        # initialise label embeddings
        self.x_embedding1 = torch.nn.Embedding(num_atom_type, self.args.emb_dim)
        self.x_embedding2 = torch.nn.Embedding(num_chirality_tag, self.args.emb_dim)
        torch.nn.init.xavier_uniform_(self.x_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.x_embedding2.weight.data)

        # initialise GNN layers
        self.gnns = torch.nn.ModuleList()
        for layer in range(self.args.num_layer):
            if self.args.gnn_type == "gin":
                self.gnns.append(GINConv(emb_dim=self.args.emb_dim))
            elif self.args.gnn_type == "gcn":
                self.gnns.append(GCNConv(emb_dim=self.args.emb_dim))
            else:
                raise NotImplementedError('Invalid GNN layer type.')

        # initialise BatchNorm layers
        self.batch_norms = torch.nn.ModuleList()
        for layer in range(self.args.num_layer):
            self.batch_norms.append(torch.nn.BatchNorm1d(self.args.emb_dim))

    def forward(self, x, edge_index, edge_attr):

        # x[:, 0] corresponds to 'possible_atomic_num_list',
        # x[:, 1] corresponds to 'possible_chirality_list'
        x = self.x_embedding1(x[:, 0]) + self.x_embedding2(x[:, 1])

        for layer in range(self.args.num_layer):
            # x are atom features of the molecule and edge_attr the atomic features of the molecule
            x = self.gnns[layer](x, edge_index, edge_attr)
            x = self.batch_norms[layer](x)
            if layer != self.args.num_layer - 1:
                x = F.relu(x)
            x = F.dropout(x, self.args.dropout_ratio, training=self.training)

        return x


if __name__ == "__main__":
    pass
