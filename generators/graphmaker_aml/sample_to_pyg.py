"""Convert one GraphMaker Async sample to PyG Data aligned with AMLWorld feature widths."""
from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
from torch_geometric.data import Data


def graphmaker_sample_to_pyg(
    X_t_one_hot: torch.Tensor,
    Y_0_one_hot: torch.Tensor,
    E_t: torch.Tensor,
    *,
    ref_x_dim: int,
    ref_edge_attr_dim: int,
    edge_fraud_rate: float,
    seed: int = 7,
) -> Data:
    """
    X_t_one_hot: (F, N, 2) from ModelAsync.sample()
    Y_0_one_hot: (N, C)
    E_t: (N, N) adjacency (upper-tri process in paper; may be symmetric)
    """
    rng = np.random.default_rng(seed)
    F, N, _ = X_t_one_hot.shape
    x_bin = X_t_one_hot.argmax(dim=-1).float().T  # (N, F)
    if x_bin.size(1) < ref_x_dim:
        pad = torch.zeros(N, ref_x_dim - x_bin.size(1), dtype=torch.float32)
        x = torch.cat([x_bin, pad], dim=1)
    else:
        x = x_bin[:, :ref_x_dim]

    y_node = Y_0_one_hot.argmax(dim=1).long()

    # Directed multigraph from undirected adjacency: emit both (u,v) and (v,u) for each edge.
    ei = []
    uu, vv = torch.nonzero(E_t, as_tuple=True)
    for u, v in zip(uu.tolist(), vv.tolist()):
        if u >= v:
            continue
        ei.append([u, v])
        ei.append([v, u])
    if not ei:
        # isolated nodes — add self-loops? avoid empty edge_index breaking GNN
        ei = [[0, 0]]
    edge_index = torch.tensor(ei, dtype=torch.long).T
    E = edge_index.size(1)
    edge_attr = torch.zeros(E, ref_edge_attr_dim, dtype=torch.float32)
    y_edge = torch.tensor(rng.binomial(1, edge_fraud_rate, size=E), dtype=torch.long)

    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y_node=y_node,
        y_edge=y_edge,
    )
