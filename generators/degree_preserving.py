from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from torch_geometric.data import Data

from data.amlworld import _build_splits_binary


@dataclass(frozen=True)
class DegreePreservingGeneratorConfig:
    seed: int = 7
    train_size: float = 0.6
    val_size: float = 0.2
    allow_self_loops: bool = True


def generate_degree_preserving_synthetic(
    real: Data,
    *,
    config: Optional[DegreePreservingGeneratorConfig] = None,
) -> Tuple[Data, Dict[str, Any]]:
    """
    Baseline generator:
    - Preserves exact in/out degree sequences via stub matching (configuration-style sampling).
    - Bootstraps node features and node labels from empirical distribution.
    - Bootstraps edge attributes and edge labels from empirical distribution.

    This is intentionally simple and is meant as a first synthetic baseline.
    """
    cfg = config or DegreePreservingGeneratorConfig()
    rng = np.random.default_rng(cfg.seed)

    num_nodes = int(real.num_nodes)
    num_edges = int(real.edge_index.size(1))

    src_real = real.edge_index[0].cpu().numpy()
    dst_real = real.edge_index[1].cpu().numpy()

    out_deg = np.bincount(src_real, minlength=num_nodes).astype(np.int64)
    in_deg = np.bincount(dst_real, minlength=num_nodes).astype(np.int64)

    out_stubs = np.repeat(np.arange(num_nodes, dtype=np.int64), out_deg)
    in_stubs = np.repeat(np.arange(num_nodes, dtype=np.int64), in_deg)
    rng.shuffle(in_stubs)

    if out_stubs.shape[0] != in_stubs.shape[0]:
        raise ValueError("Degree sequences mismatch; cannot generate synthetic edges.")

    src = out_stubs
    dst = in_stubs

    if not cfg.allow_self_loops:
        # Simple repair: resample dst positions that create self-loops (bounded iterations).
        for _ in range(5):
            bad = src == dst
            if not bad.any():
                break
            swap_idx = np.where(bad)[0]
            rng.shuffle(swap_idx)
            # rotate dst among bad positions
            dst_bad = dst[swap_idx]
            dst[swap_idx] = np.roll(dst_bad, shift=1)

    edge_index = torch.tensor(np.stack([src, dst], axis=0), dtype=torch.long)

    # Bootstrap node features/labels (preserve marginal distributions)
    x_real = real.x.cpu().numpy()
    y_node_real = real.y_node.cpu().numpy()
    node_boot = rng.integers(0, num_nodes, size=num_nodes)
    x_syn = torch.tensor(x_real[node_boot], dtype=torch.float32)
    y_node_syn = torch.tensor(y_node_real[node_boot], dtype=torch.long)

    # Bootstrap edge attributes/labels
    edge_attr_real = real.edge_attr.cpu().numpy()
    y_edge_real = real.y_edge.cpu().numpy()
    edge_boot = rng.integers(0, num_edges, size=num_edges)
    edge_attr_syn = torch.tensor(edge_attr_real[edge_boot], dtype=torch.float32)
    y_edge_syn = torch.tensor(y_edge_real[edge_boot], dtype=torch.long)

    syn = Data(
        x=x_syn,
        edge_index=edge_index,
        edge_attr=edge_attr_syn,
        y_node=y_node_syn,
        y_edge=y_edge_syn,
    )

    # splits/masks on synthetic for training
    n_y = y_node_syn.cpu().numpy()
    n_tr, n_va, n_te = _build_splits_binary(
        n_y, train_size=cfg.train_size, val_size=cfg.val_size, seed=cfg.seed
    )
    syn.node_train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    syn.node_val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    syn.node_test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    syn.node_train_mask[n_tr] = True
    syn.node_val_mask[n_va] = True
    syn.node_test_mask[n_te] = True

    e_y = y_edge_syn.cpu().numpy()
    e_tr, e_va, e_te = _build_splits_binary(
        e_y, train_size=cfg.train_size, val_size=cfg.val_size, seed=cfg.seed
    )
    syn.edge_train_idx = torch.tensor(e_tr, dtype=torch.long)
    syn.edge_val_idx = torch.tensor(e_va, dtype=torch.long)
    syn.edge_test_idx = torch.tensor(e_te, dtype=torch.long)

    meta = {
        "generator": "degree_preserving_bootstrap",
        "seed": cfg.seed,
        "preserves": ["in_degree_sequence", "out_degree_sequence", "marginal_node_feature_dist", "marginal_edge_feature_dist"],
    }
    return syn, meta

