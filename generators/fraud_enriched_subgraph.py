"""
Fraud-enriched node-induced subgraphs for smaller / denser training graphs.

Picks (almost) all node-level positives, expands by k hops on an undirected view of
the transaction graph, optionally fills up to ``max_nodes`` with random other nodes,
then induces edges and relabels vertices 0..n-1. Splits are rebuilt on the subgraph.
"""

from __future__ import annotations

from typing import Any, Dict, List, Set, Tuple

import numpy as np
import torch
from torch_geometric.data import Data

from data.amlworld import _build_splits_binary


def _undirected_neighbors(edge_index: np.ndarray, num_nodes: int) -> List[Set[int]]:
    adj: List[Set[int]] = [set() for _ in range(num_nodes)]
    for e in range(edge_index.shape[1]):
        u, v = int(edge_index[0, e]), int(edge_index[1, e])
        adj[u].add(v)
        adj[v].add(u)
    return adj


def extract_fraud_enriched_subgraph(
    data: Data,
    *,
    max_nodes: int,
    seed: int,
    neighbor_hops: int = 2,
    train_size: float = 0.6,
    val_size: float = 0.2,
) -> Tuple[Data, Dict[str, Any]]:
    """
    Return a new ``Data`` on at most ``max_nodes`` vertices with elevated fraud density
    versus a uniform random node subset.
    """
    rng = np.random.default_rng(int(seed))
    y_node = data.y_node.cpu().numpy().reshape(-1)
    n_all = int(data.num_nodes)
    pos_ix = np.where(y_node == 1)[0]
    ei = data.edge_index.cpu().numpy()

    keep: Set[int] = set(int(x) for x in pos_ix.tolist())
    adj = _undirected_neighbors(ei, n_all)
    boundary: Set[int] = set(keep)
    for _ in range(int(neighbor_hops)):
        nxt: Set[int] = set()
        for u in boundary:
            for v in adj[u]:
                nxt.add(v)
        nxt -= keep
        keep |= nxt
        boundary = nxt
        if not boundary:
            break

    filler_pool = [i for i in range(n_all) if i not in keep]

    if len(keep) > int(max_nodes):
        need_drop = len(keep) - int(max_nodes)
        non_pos = [i for i in keep if y_node[i] == 0]
        rng.shuffle(non_pos)
        for j in range(min(need_drop, len(non_pos))):
            keep.discard(non_pos[j])
    elif len(keep) < int(max_nodes) and filler_pool:
        need = int(max_nodes) - len(keep)
        rng.shuffle(filler_pool)
        for v in filler_pool[:need]:
            keep.add(int(v))

    old_nodes = sorted(keep)
    remap = {old: i for i, old in enumerate(old_nodes)}
    sset = set(old_nodes)
    rows: List[int] = []
    for e in range(ei.shape[1]):
        u, v = int(ei[0, e]), int(ei[1, e])
        if u in sset and v in sset:
            rows.append(e)
    if not rows:
        raise ValueError(
            "Fraud-enriched subgraph has no edges; raise neighbor_hops or max_transactions."
        )

    rows_np = np.asarray(rows, dtype=np.int64)
    new_ei = np.stack(
        [
            np.array([remap[int(ei[0, e])] for e in rows_np], dtype=np.int64),
            np.array([remap[int(ei[1, e])] for e in rows_np], dtype=np.int64),
        ]
    )
    new_x = data.x.cpu().numpy()[old_nodes]
    new_yn = y_node[old_nodes]
    new_ea = data.edge_attr.cpu().numpy()[rows_np]
    new_ye = data.y_edge.cpu().numpy()[rows_np]

    syn = Data(
        x=torch.tensor(new_x, dtype=torch.float32),
        edge_index=torch.tensor(new_ei, dtype=torch.long),
        edge_attr=torch.tensor(new_ea, dtype=torch.float32),
        y_node=torch.tensor(new_yn, dtype=torch.long),
        y_edge=torch.tensor(new_ye, dtype=torch.long),
    )

    n_tr, n_va, n_te = _build_splits_binary(syn.y_node.cpu().numpy(), train_size, val_size, seed)
    syn.node_train_mask = torch.zeros(syn.num_nodes, dtype=torch.bool)
    syn.node_val_mask = torch.zeros(syn.num_nodes, dtype=torch.bool)
    syn.node_test_mask = torch.zeros(syn.num_nodes, dtype=torch.bool)
    syn.node_train_mask[n_tr] = True
    syn.node_val_mask[n_va] = True
    syn.node_test_mask[n_te] = True

    e_y = syn.y_edge.cpu().numpy()
    e_tr, e_va, e_te = _build_splits_binary(e_y, train_size, val_size, seed + 1)
    syn.edge_train_idx = torch.tensor(e_tr, dtype=torch.long)
    syn.edge_val_idx = torch.tensor(e_va, dtype=torch.long)
    syn.edge_test_idx = torch.tensor(e_te, dtype=torch.long)

    meta: Dict[str, Any] = {
        "subgraph": "fraud_enriched_bfs",
        "neighbor_hops": int(neighbor_hops),
        "max_nodes_cap": int(max_nodes),
        "num_nodes": int(syn.num_nodes),
        "num_edges": int(syn.edge_index.size(1)),
        "node_label_pos_rate": float(syn.y_node.float().mean().item()),
        "edge_label_pos_rate": float(syn.y_edge.float().mean().item()),
    }
    return syn, meta
