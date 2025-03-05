import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as tg
import torch_geometric.nn as geom_nn
from torch_geometric.nn.conv import TransformerConv, GCNConv, RGCNConv
from dtegnn.model.layers.layers import EGNNLayer


class EGNN(nn.Module):
    def __init__(
            self,
            depth,
            hidden_features,
            node_features,
            out_features,
            norm,
            activation="swish",
            aggr="sum",
            pool="add",
            residual=True,
            RFF_dim=None,
            RFF_sigma=None,
            return_pos=False,
            **kwargs
    ):
        """
        E(n) Equivariant GNN model

        Args:
            depth: (int) - number of message passing layers
            hidden_features: (int) - hidden dimension
            node_features: (int) - initial node feature dimension
            out_features: (int) - output number of classes
            activation: (str) - non-linearity within MLPs (swish/relu)
            norm: (str) - normalisation layer (layer/batch)
            aggr: (str) - aggregation function `\oplus` (sum/mean/max)
            pool: (str) - global pooling function (sum/mean)
            residual: (bool) - whether to use residual connections
        """
        super().__init__()
        self.name = "EGNN"
        
        self.emb_in = nn.Linear(node_features, hidden_features)
        
        self.make_dist = PBCConvLayer()

        self.convs = torch.nn.ModuleList()
        for layer in range(depth):
            self.convs.append(EGNNLayer(hidden_features, activation, norm, aggr, RFF_dim, RFF_sigma))

        self.pool = {"mean": tg.nn.global_mean_pool, "add": tg.nn.global_add_pool, "none": None}[pool]

        self.pred = torch.nn.Sequential(
            torch.nn.Linear(hidden_features, hidden_features),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_features, out_features)
        )
        self.residual = residual

    def forward(self, batch, train_flag=True):
        h = self.emb_in(batch.x)
        batch.pos = torch.autograd.Variable(batch.pos, requires_grad=True)
        distances = self.make_dist(batch.pos, batch.edge_index, batch.cell_offset ,batch.unit_cell[:3])
        for conv in self.convs:
            h_update = conv(h, batch.edge_index, distances)

            h = h + h_update if self.residual else h_update

        out = h
        if self.pool is not None:
            out = self.pool(h, batch.batch)
        
        energy = self.pred(out)
        
        if train_flag:
            force = -1.0 * torch.autograd.grad(
                        energy,
                        batch.pos,
                        grad_outputs=torch.ones_like(energy),
                        create_graph=True,
                        retain_graph=True
                    )[0]
        else:
            force = -1.0 * torch.autograd.grad(
            energy,
            batch.pos,
            grad_outputs=torch.ones_like(energy),
            create_graph=False, 
            retain_graph=False 
        )[0]

        return energy, force

class PBCConvLayer(nn.Module):
    def __init__(self):
        super(PBCConvLayer, self).__init__()

    def forward(self, pos, edge_index, offsets, cell_vectors):

            to_move = pos[edge_index[1]]

            pbc_adjustments = torch.matmul(offsets, cell_vectors)
            corrected = to_move - pbc_adjustments        
            distances = torch.linalg.vector_norm(corrected - pos[edge_index[0]],dim=-1)

            return distances
