import math
import torch
from torch import Tensor
import torch.nn as nn
import torch_geometric as tg
import torch.nn.functional as F

from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import softmax
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import Adj, OptTensor, PairTensor, SparseTensor
from torch_scatter import scatter

from typing import Optional, Tuple, Union

class RFF(nn.Module):
    def __init__(self, in_features, out_features, sigma=1.0):
        super().__init__()
        self.sigma = sigma
        self.in_features = in_features
        self.out_features = out_features

        if out_features % 2 != 0:
            self.compensation = 1
        else:
            self.compensation = 0

        B = torch.randn(int(out_features / 2) + self.compensation, in_features) * sigma
        B /= math.sqrt(2)
        self.register_buffer("B", B)

    def forward(self, x):
        x = F.linear(x, self.B)
        x = torch.cat((x.sin(), x.cos()), dim=-1)
        if self.compensation:
            x = x[..., :-1]
        return x

    def extra_repr(self) -> str:
        return "in_features={}, out_features={}, sigma={}".format(
            self.in_features, self.out_features, self.sigma
        )



class EGNNLayer(tg.nn.MessagePassing):
    def __init__(self, emb_dim, activation="relu", norm="layer", aggr="add", RFF_dim=None, RFF_sigma=None, mask=None):
        super().__init__(aggr=aggr)
        self.emb_dim = emb_dim
        self.activation = {"swish": nn.SiLU(), "relu": nn.ReLU()}[activation]
        self.norm = {"layer": torch.nn.LayerNorm, "batch": torch.nn.BatchNorm1d, "none": nn.Identity}[norm]
        self.RFF_dim = RFF_dim
        self.RFF_sigma = RFF_sigma
        self.mask = mask

        self.mlp_msg = nn.Sequential(
            nn.Linear(2 * emb_dim + 1 if self.RFF_dim is None else 2 * emb_dim + RFF_dim, emb_dim),
            self.norm(emb_dim),
            self.activation,
            nn.Linear(emb_dim, emb_dim),
            self.norm(emb_dim),
            self.activation,
        )

        self.mlp_upd = nn.Sequential(
            nn.Linear(2 * emb_dim, emb_dim),
            self.norm(emb_dim),
            self.activation,
            nn.Linear(emb_dim, emb_dim),
            self.norm(emb_dim) if norm != "none" else nn.Identity(),
            self.activation,
        )
        if self.RFF_dim is not None:
            self.RFF = RFF(1, RFF_dim, RFF_sigma)

    def forward(self, h, edge_index, distances, mask=None):
        self.mask = mask
        out = self.propagate(edge_index, h=h, distances=distances, mask=mask)
        return out

    def message(self, h_i, h_j, distances):
        dists = distances.unsqueeze(1)
        if self.RFF_dim is not None:
            dists = self.RFF(dists)
        msg = torch.cat([h_i, h_j, dists], dim=-1)
        msg = self.mlp_msg(msg)
        return msg

    def update(self, aggr_out, h):
        msg_aggr = aggr_out
        upd_out = self.mlp_upd(torch.cat([h, msg_aggr], dim=-1))
        if self.mask is not None:
            upd_out = torch.where(self.mask.unsqueeze(-1), upd_out, h)
        return upd_out

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(emb_dim={self.emb_dim}, aggr={self.aggr})"
