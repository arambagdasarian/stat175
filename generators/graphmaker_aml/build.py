"""
Convert AMLWorld (PyG) to a DGL graph suitable for GraphMaker's preprocess():
- undirected edges (symmetrized transaction directions)
- integer node features (quantile bins per dimension)
- integer node labels y_node (fraud vs not)
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import dgl
import numpy as np
import torch

from data.amlworld import load_amlworld_hi_small_pyg
from generators.fraud_enriched_subgraph import extract_fraud_enriched_subgraph
from generators.snowball_sampling import snowball_sample


def _induced_subgraph(
    edge_index: torch.Tensor,
    y_node: torch.Tensor,
    y_edge: torch.Tensor,
    x: torch.Tensor,
    edge_attr: torch.Tensor,
    max_nodes: int,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Randomly keep max_nodes nodes; keep edges with both ends inside; relabel 0..n-1."""
    rng = np.random.default_rng(seed)
    n_all = y_node.size(0)
    if n_all <= max_nodes:
        return edge_index, y_node, y_edge, x, edge_attr
    pos = torch.where(y_node == 1)[0].cpu().numpy()
    keep = set(rng.choice(np.arange(n_all), size=max_nodes, replace=False).tolist())
    if len(pos) > 0 and not any(int(p) in keep for p in pos):
        # Keep at least one positive node if any exist (GraphMaker expects labeled classes).
        keep.discard(rng.choice(list(keep)))
        keep.add(int(rng.choice(pos)))
    ei = edge_index.cpu().numpy()
    ye = y_edge.cpu().numpy()
    ea = edge_attr.cpu().numpy()
    rows = []
    for e in range(ei.shape[1]):
        u, v = int(ei[0, e]), int(ei[1, e])
        if u in keep and v in keep:
            rows.append(e)
    if not rows:
        raise ValueError("Induced subgraph has no edges; increase max_transactions or max_nodes.")
    rows = np.array(rows, dtype=np.int64)
    old_nodes = sorted(keep)
    remap = {old: i for i, old in enumerate(old_nodes)}
    new_ei = np.stack(
        [
            [remap[int(ei[0, e])] for e in rows],
            [remap[int(ei[1, e])] for e in rows],
        ]
    )
    new_ye = ye[rows]
    new_ea = ea[rows]
    new_yn = y_node.cpu().numpy()[old_nodes]
    new_x = x.cpu().numpy()[old_nodes]
    n = len(old_nodes)
    return (
        torch.tensor(new_ei, dtype=torch.long),
        torch.tensor(new_yn, dtype=torch.long),
        torch.tensor(new_ye, dtype=torch.long),
        torch.tensor(new_x, dtype=torch.float32),
        torch.tensor(new_ea, dtype=torch.float32),
    )


def _binarize_features_median(x: torch.Tensor) -> torch.Tensor:
    """
    GraphMaker's reference code assumes **binary** node attributes (2 classes per column);
    see `MarginalTransition` in `model/diffusion.py` (X_marginal shape (F, 2)).
    We threshold each continuous column at its median to obtain 0/1.
    """
    x = x.cpu().numpy()
    n, d = x.shape
    out = np.zeros((n, d), dtype=np.int64)
    for j in range(d):
        med = np.median(x[:, j])
        out[:, j] = (x[:, j] > med).astype(np.int64)
    return torch.tensor(out, dtype=torch.long)


def build_dgl_for_graphmaker(
    data_dir: Path,
    *,
    max_transactions: int = 200_000,
    max_nodes: int = 3500,
    seed: int = 7,
    fraud_enriched: bool = False,
    neighbor_hops: int = 2,
    sampling: str = "random_induced",
    snowball_top_k: int = 20,
    snowball_wave_limit: int = 15,
    slice_mode: str = "prefix",
    balance_scan_rows: int = 2_000_000,
    target_edge_pos_fraction: float = 0.05,
    stratify_edges_if_possible: bool = True,
    stratify_nodes_if_possible: bool = True,
) -> dgl.DGLGraph:
    """
    ``sampling`` choices
    --------------------
    ``'random_induced'``   – uniform random node subset (original default).
    ``'fraud_enriched'``   – BFS from all SAR nodes + random filler (legacy flag).
    ``'snowball'``         – wave-by-wave expansion from top-k SAR hubs; no filler.
    ``'fraud_enriched'`` is still accepted for backward compatibility;
    setting ``fraud_enriched=True`` (the old CLI flag) is equivalent to
    ``sampling='fraud_enriched'``.
    """
    # Legacy flag maps to the named strategy.
    if fraud_enriched and sampling == "random_induced":
        sampling = "fraud_enriched"

    pyg, _ = load_amlworld_hi_small_pyg(
        data_dir,
        max_transactions=max_transactions,
        seed=seed,
        slice_mode=slice_mode,
        balance_scan_rows=balance_scan_rows,
        target_edge_pos_fraction=target_edge_pos_fraction,
        stratify_edges_if_possible=stratify_edges_if_possible,
        stratify_nodes_if_possible=stratify_nodes_if_possible,
    )
    if sampling == "snowball":
        sub, sub_meta = snowball_sample(
            pyg,
            max_nodes=max_nodes,
            seed=seed,
            top_k_seeds=int(snowball_top_k),
            wave_limit=int(snowball_wave_limit),
        )
        print(
            f"[snowball] nodes={sub_meta['num_nodes']}  edges={sub_meta['num_edges']}  "
            f"density={sub_meta['density']:.2e}  "
            f"SAR-frac={sub_meta['node_sar_fraction']:.3f}  "
            f"fraud-edge-frac={sub_meta['edge_fraud_fraction']:.3f}  "
            f"waves={sub_meta['waves_expanded']}",
            flush=True,
        )
        ei, yn, ye, xf, eattr = (
            sub.edge_index,
            sub.y_node,
            sub.y_edge,
            sub.x,
            sub.edge_attr,
        )
    elif sampling == "fraud_enriched":
        sub, _sub_meta = extract_fraud_enriched_subgraph(
            pyg, max_nodes=max_nodes, seed=seed, neighbor_hops=neighbor_hops
        )
        ei, yn, ye, xf, eattr = (
            sub.edge_index,
            sub.y_node,
            sub.y_edge,
            sub.x,
            sub.edge_attr,
        )
    else:  # random_induced
        ei, yn, ye, xf, eattr = _induced_subgraph(
            pyg.edge_index,
            pyg.y_node,
            pyg.y_edge,
            pyg.x,
            pyg.edge_attr,
            max_nodes=max_nodes,
            seed=seed,
        )
    n = yn.size(0)
    feat = _binarize_features_median(xf).float()
    # Drop constant columns (GraphMaker / one_hot expects variability; mirrors Cora path).
    nz = feat.sum(dim=0) != 0
    feat = feat[:, nz]
    no = feat.sum(dim=0) != feat.size(0)
    feat = feat[:, no]
    feat = feat.long()
    label = yn.long().clamp(min=0)
    # Symmetrize directed edges for GraphMaker (undirected adjacency).
    src, dst = ei[0].tolist(), ei[1].tolist()
    us, vs = [], []
    for u, v in zip(src, dst):
        if u == v:
            continue
        us.extend([u, v])
        vs.extend([v, u])
    g = dgl.graph((us, vs), num_nodes=n)
    g = dgl.remove_self_loop(g)
    g.ndata["feat"] = feat
    g.ndata["label"] = label
    return g


def save_training_graph(g: dgl.DGLGraph, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    dgl.save_graphs(str(path), [g])
