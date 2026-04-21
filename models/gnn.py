from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


class GraphSAGEEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 2,
        *,
        dropout: float = 0.1,
    ):
        super().__init__()
        if num_layers < 2:
            raise ValueError("num_layers must be >= 2")
        self.dropout_p = float(dropout)
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout_p, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x


class NodeClassifier(nn.Module):
    def __init__(self, encoder: GraphSAGEEncoder, emb_dim: int):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(emb_dim, 1)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x, edge_index)
        logits = self.head(h).squeeze(-1)
        return logits, h


class EdgeClassifier(nn.Module):
    def __init__(self, emb_dim: int, edge_feat_dim: int, hidden_dim: int = 128, *, dropout: float = 0.1):
        super().__init__()
        dp = float(dropout)
        self.mlp = nn.Sequential(
            nn.Linear(2 * emb_dim + edge_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dp),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dp),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_idx: torch.Tensor,
    ) -> torch.Tensor:
        # edge_idx indexes into columns of edge_index/edge_attr
        src = edge_index[0, edge_idx]
        dst = edge_index[1, edge_idx]
        z = torch.cat([h[src], h[dst], edge_attr[edge_idx]], dim=-1)
        return self.mlp(z).squeeze(-1)

