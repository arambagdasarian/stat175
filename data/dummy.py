from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch
from torch_geometric.data import Data

from data.amlworld import _build_splits_binary


@dataclass(frozen=True)
class DummyConfig:
    num_nodes: int = 5000
    num_edges: int = 50000
    node_feat_dim: int = 16
    edge_feat_dim: int = 8
    node_pos_rate: float = 0.01
    edge_pos_rate: float = 0.002
    seed: int = 7
    train_size: float = 0.6
    val_size: float = 0.2


def make_dummy_pyg(cfg: DummyConfig = DummyConfig()) -> Tuple[Data, Dict[str, str]]:
    rng = np.random.default_rng(cfg.seed)

    src = rng.integers(0, cfg.num_nodes, size=cfg.num_edges, dtype=np.int64)
    dst = rng.integers(0, cfg.num_nodes, size=cfg.num_edges, dtype=np.int64)
    edge_index = torch.tensor(np.stack([src, dst], axis=0), dtype=torch.long)

    x = torch.tensor(rng.normal(size=(cfg.num_nodes, cfg.node_feat_dim)).astype(np.float32))
    edge_attr = torch.tensor(rng.normal(size=(cfg.num_edges, cfg.edge_feat_dim)).astype(np.float32))

    y_node = torch.tensor((rng.random(cfg.num_nodes) < cfg.node_pos_rate).astype(np.int64))
    y_edge = torch.tensor((rng.random(cfg.num_edges) < cfg.edge_pos_rate).astype(np.int64))

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y_node=y_node, y_edge=y_edge)

    n_tr, n_va, n_te = _build_splits_binary(
        y_node.numpy(), cfg.train_size, cfg.val_size, cfg.seed
    )
    data.node_train_mask = torch.zeros(cfg.num_nodes, dtype=torch.bool)
    data.node_val_mask = torch.zeros(cfg.num_nodes, dtype=torch.bool)
    data.node_test_mask = torch.zeros(cfg.num_nodes, dtype=torch.bool)
    data.node_train_mask[n_tr] = True
    data.node_val_mask[n_va] = True
    data.node_test_mask[n_te] = True

    e_tr, e_va, e_te = _build_splits_binary(
        y_edge.numpy(), cfg.train_size, cfg.val_size, cfg.seed
    )
    data.edge_train_idx = torch.tensor(e_tr, dtype=torch.long)
    data.edge_val_idx = torch.tensor(e_va, dtype=torch.long)
    data.edge_test_idx = torch.tensor(e_te, dtype=torch.long)

    meta = {"dataset": "dummy", "note": "Synthetic dummy graph for pipeline smoke testing."}
    return data, meta

