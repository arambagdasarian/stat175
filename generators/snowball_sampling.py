"""
Snowball sampling for AMLWorld transaction graphs.

Starting from the highest-out-degree SAR (fraud hub) nodes, expands the sample
wave-by-wave through the undirected adjacency until the node budget is reached.
Every node in the returned subgraph has a directed path back to at least one
fraud seed — unlike the BFS flood-fill in fraud_enriched_subgraph.py (which
starts from ALL SAR nodes simultaneously and pads with random filler), snowball
sampling here is:

  1. Seeded from a small, well-chosen set of fraud hubs (configurable via
     ``seed_strategy`` and ``top_k_seeds``).
  2. Expanded one wave at a time with shuffle-based fair capping so the node
     budget is respected without biasing toward the first frontier node.
  3. Never padded with random disconnected nodes: if the budget is not met
     the subgraph is simply smaller, but fully connected to the seed community.

The result is a denser, more community-cohesive subgraph than random induced
sampling, and better focused on actual laundering clusters than the broad BFS.

Main entry-point
----------------
    snowball_sample(data, *, max_nodes, seed, ...) -> (Data, meta_dict)
"""

from __future__ import annotations

from typing import Any, Dict, List, Set, Tuple

import numpy as np
import torch
from torch_geometric.data import Data

from data.amlworld import _build_splits_binary


def _undirected_adj(ei: np.ndarray, num_nodes: int) -> List[List[int]]:
    """Build an undirected adjacency list from a directed edge_index (O(E))."""
    adj: List[List[int]] = [[] for _ in range(num_nodes)]
    for e in range(ei.shape[1]):
        u, v = int(ei[0, e]), int(ei[1, e])
        if u != v:
            adj[u].append(v)
            adj[v].append(u)
    return adj


def snowball_sample(
    data: Data,
    *,
    max_nodes: int,
    seed: int,
    top_k_seeds: int = 20,
    seed_strategy: str = "top_sar_by_degree",
    wave_limit: int = 15,
    train_size: float = 0.6,
    val_size: float = 0.2,
) -> Tuple[Data, Dict[str, Any]]:
    """Return a node-induced subgraph grown by snowball expansion from fraud hubs.

    Parameters
    ----------
    data : Data
        Full PyG graph (AMLWorld HI-Small or subslice thereof).
    max_nodes : int
        Hard cap on the number of nodes in the returned subgraph.
    top_k_seeds : int
        Number of seed nodes to start the snowball from.  Ranked by total
        degree within the SAR set for ``seed_strategy='top_sar_by_degree'``.
    seed_strategy : str
        ``'top_sar_by_degree'``  — top-k SAR (node-positive) nodes by total
                                   degree in the undirected graph.  Default.
        ``'all_sar'``            — every SAR node is a seed; matches the BFS
                                   flood-fill of fraud_enriched_subgraph.py but
                                   with wave-based tracking and no random filler.
        ``'top_degree'``         — top-k nodes by total degree regardless of
                                   SAR label; useful if SAR nodes are rare.
    wave_limit : int
        Maximum number of expansion waves (depth cap).  In practice, the budget
        is hit long before this on AMLWorld.
    train_size, val_size : float
        Fractions for rebuilding train/val/test masks on the subgraph.

    Returns
    -------
    sub : Data
        The node-induced subgraph with rebuilt masks.
    meta : dict
        Sampling diagnostics (node/edge counts, density, fraud rates, wave stats).
    """
    rng = np.random.default_rng(int(seed))
    y_node = data.y_node.cpu().numpy().reshape(-1)
    n_all = int(data.num_nodes)
    ei = data.edge_index.cpu().numpy()

    # Degree in the directed graph (for seed ranking)
    out_deg = np.bincount(ei[0], minlength=n_all)
    in_deg = np.bincount(ei[1], minlength=n_all)
    total_deg = out_deg + in_deg

    sar_idx = np.flatnonzero(y_node == 1)

    # ── Seed selection ────────────────────────────────────────────────────────
    if seed_strategy == "all_sar":
        initial_seeds: List[int] = sar_idx.tolist()
    elif seed_strategy == "top_degree":
        k = min(int(top_k_seeds), n_all)
        initial_seeds = np.argsort(total_deg)[::-1][:k].tolist()
    else:  # top_sar_by_degree (default)
        if len(sar_idx) == 0:
            k = min(int(top_k_seeds), n_all)
            initial_seeds = np.argsort(total_deg)[::-1][:k].tolist()
        else:
            k = min(int(top_k_seeds), len(sar_idx))
            ranked = sar_idx[np.argsort(total_deg[sar_idx])[::-1]]
            initial_seeds = ranked[:k].tolist()

    # ── Build undirected adjacency (O(E), done once) ──────────────────────────
    adj = _undirected_adj(ei, n_all)

    # ── Wave-by-wave snowball expansion ───────────────────────────────────────
    visited: Set[int] = set(int(x) for x in initial_seeds)
    frontier: List[int] = [int(x) for x in initial_seeds]
    wave_stats: List[Dict[str, int]] = [
        {"wave": 0, "new_nodes": len(frontier), "total_nodes": len(frontier)}
    ]

    for wave_num in range(1, int(wave_limit) + 1):
        if len(visited) >= int(max_nodes):
            break
        next_frontier: List[int] = []
        # Shuffle frontier order so the node-budget cut is not biased to any
        # particular seed's neighbourhood.
        idxs = rng.permutation(len(frontier)).tolist()
        for fi in idxs:
            u = frontier[fi]
            nbrs = adj[u]
            nbr_idxs = rng.permutation(len(nbrs)).tolist()
            for ni in nbr_idxs:
                v = nbrs[ni]
                if v not in visited:
                    visited.add(v)
                    next_frontier.append(v)
                    if len(visited) >= int(max_nodes):
                        break
            if len(visited) >= int(max_nodes):
                break
        wave_stats.append(
            {
                "wave": wave_num,
                "new_nodes": len(next_frontier),
                "total_nodes": len(visited),
            }
        )
        if not next_frontier:
            break
        frontier = next_frontier

    # ── Induce subgraph on retained nodes ─────────────────────────────────────
    old_nodes = sorted(visited)
    remap = {old: i for i, old in enumerate(old_nodes)}
    sset = set(old_nodes)

    rows = [
        e
        for e in range(ei.shape[1])
        if int(ei[0, e]) in sset and int(ei[1, e]) in sset
    ]
    if not rows:
        raise ValueError(
            "Snowball subgraph has no edges. "
            "Try increasing max_transactions, top_k_seeds, or wave_limit."
        )

    rows_np = np.asarray(rows, dtype=np.int64)
    new_ei = np.stack(
        [
            np.array([remap[int(ei[0, e])] for e in rows_np], dtype=np.int64),
            np.array([remap[int(ei[1, e])] for e in rows_np], dtype=np.int64),
        ]
    )

    x_np = data.x.cpu().numpy()
    ea_np = data.edge_attr.cpu().numpy()
    y_edge_np = data.y_edge.cpu().numpy()

    new_x = x_np[old_nodes]
    new_yn = y_node[old_nodes]
    new_ea = ea_np[rows_np]
    new_ye = y_edge_np[rows_np]

    sub = Data(
        x=torch.tensor(new_x, dtype=torch.float32),
        edge_index=torch.tensor(new_ei, dtype=torch.long),
        edge_attr=torch.tensor(new_ea, dtype=torch.float32),
        y_node=torch.tensor(new_yn, dtype=torch.long),
        y_edge=torch.tensor(new_ye, dtype=torch.long),
    )

    # Rebuild splits on the subgraph
    n_tr, n_va, n_te = _build_splits_binary(sub.y_node.cpu().numpy(), train_size, val_size, seed)
    sub.node_train_mask = torch.zeros(sub.num_nodes, dtype=torch.bool)
    sub.node_val_mask = torch.zeros(sub.num_nodes, dtype=torch.bool)
    sub.node_test_mask = torch.zeros(sub.num_nodes, dtype=torch.bool)
    sub.node_train_mask[n_tr] = True
    sub.node_val_mask[n_va] = True
    sub.node_test_mask[n_te] = True

    e_y = sub.y_edge.cpu().numpy()
    e_tr, e_va, e_te = _build_splits_binary(e_y, train_size, val_size, seed + 1)
    sub.edge_train_idx = torch.tensor(e_tr, dtype=torch.long)
    sub.edge_val_idx = torch.tensor(e_va, dtype=torch.long)
    sub.edge_test_idx = torch.tensor(e_te, dtype=torch.long)

    n_sub = int(sub.num_nodes)
    e_sub = int(sub.edge_index.size(1))
    n_sar = int(sub.y_node.sum().item())
    n_fraud_e = int(sub.y_edge.sum().item())

    meta: Dict[str, Any] = {
        "sampler": "snowball",
        "seed_strategy": seed_strategy,
        "top_k_seeds_requested": int(top_k_seeds),
        "initial_seeds_used": len(initial_seeds),
        "wave_limit": int(wave_limit),
        "waves_expanded": len(wave_stats) - 1,
        "wave_stats": wave_stats,
        "num_nodes": n_sub,
        "num_edges": e_sub,
        "density": float(e_sub) / max(1, n_sub * (n_sub - 1)),
        "avg_degree": float(2 * e_sub) / max(1, n_sub),
        "node_sar_count": n_sar,
        "node_sar_fraction": float(n_sar) / max(1, n_sub),
        "edge_fraud_count": n_fraud_e,
        "edge_fraud_fraction": float(n_fraud_e) / max(1, e_sub),
        "max_nodes_cap": int(max_nodes),
    }
    return sub, meta
