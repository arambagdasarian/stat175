from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import networkx as nx
from scipy.stats import ks_2samp
import torch
from torch_geometric.data import Data


def _to_nx_digraph(data: Data, max_edges: Optional[int] = None) -> nx.MultiDiGraph:
    edge_index = data.edge_index.cpu().numpy()
    if max_edges is not None and edge_index.shape[1] > max_edges:
        edge_index = edge_index[:, :max_edges]
    g = nx.MultiDiGraph()
    g.add_nodes_from(range(int(data.num_nodes)))
    g.add_edges_from(edge_index.T.tolist())
    return g


def degree_histogram(data: Data) -> Dict[str, Any]:
    src = data.edge_index[0].cpu().numpy()
    dst = data.edge_index[1].cpu().numpy()
    n = int(data.num_nodes)
    out_deg = np.bincount(src, minlength=n)
    in_deg = np.bincount(dst, minlength=n)
    return {
        "in_deg": in_deg,
        "out_deg": out_deg,
        "in_deg_hist": np.bincount(in_deg).tolist(),
        "out_deg_hist": np.bincount(out_deg).tolist(),
        "in_deg_mean": float(in_deg.mean()),
        "out_deg_mean": float(out_deg.mean()),
    }


def approx_clustering_undirected(data: Data, max_edges: int = 2_000_000) -> float:
    g = _to_nx_digraph(data, max_edges=max_edges)
    ug = nx.Graph(g)  # undirected projection
    return float(nx.average_clustering(ug))


def approx_path_length(data: Data, sample_nodes: int = 2000, seed: int = 7, max_edges: int = 2_000_000) -> Dict[str, Any]:
    g = _to_nx_digraph(data, max_edges=max_edges)
    ug = nx.Graph(g)
    rng = np.random.default_rng(seed)
    nodes = list(ug.nodes())
    if len(nodes) == 0:
        return {"mean": None, "median": None, "n_pairs": 0}
    sample = rng.choice(nodes, size=min(sample_nodes, len(nodes)), replace=False)
    lengths = []
    for s in sample:
        sp = nx.single_source_shortest_path_length(ug, s, cutoff=10)
        # exclude self-distance
        lengths.extend([d for t, d in sp.items() if t != s])
    if not lengths:
        return {"mean": None, "median": None, "n_pairs": 0}
    arr = np.array(lengths, dtype=float)
    return {"mean": float(arr.mean()), "median": float(np.median(arr)), "n_pairs": int(arr.size)}


def ks_feature_similarity(real: np.ndarray, syn: np.ndarray) -> Dict[str, Any]:
    if real.ndim != 2 or syn.ndim != 2:
        raise ValueError("Expected 2D arrays")
    d = min(real.shape[1], syn.shape[1])
    stats = []
    pvals = []
    for j in range(d):
        r = real[:, j]
        s = syn[:, j]
        res = ks_2samp(r, s, alternative="two-sided", mode="auto")
        stats.append(float(res.statistic))
        pvals.append(float(res.pvalue))
    return {
        "ks_stat_mean": float(np.mean(stats)),
        "ks_stat_max": float(np.max(stats)),
        "ks_pvalue_mean": float(np.mean(pvals)),
        "per_dim": {"ks_stat": stats, "ks_pvalue": pvals},
    }


def feature_correlation_similarity(real: np.ndarray, syn: np.ndarray) -> Dict[str, Any]:
    # Compare correlation matrices via Frobenius norm on overlapping dims
    d = min(real.shape[1], syn.shape[1])
    r = np.corrcoef(real[:, :d], rowvar=False)
    s = np.corrcoef(syn[:, :d], rowvar=False)
    # nan-safe: replace nan correlations (constant columns) with 0
    r = np.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0)
    s = np.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0)
    diff = r - s
    return {"corr_fro_norm": float(np.linalg.norm(diff, ord="fro")), "dims": int(d)}


def structural_and_feature_report(real: Data, syn: Data) -> Dict[str, Any]:
    real_deg = degree_histogram(real)
    syn_deg = degree_histogram(syn)

    rep: Dict[str, Any] = {
        "degree": {
            "real": {"in_deg_mean": real_deg["in_deg_mean"], "out_deg_mean": real_deg["out_deg_mean"]},
            "syn": {"in_deg_mean": syn_deg["in_deg_mean"], "out_deg_mean": syn_deg["out_deg_mean"]},
            "in_deg_hist_l1": float(
                np.sum(np.abs(np.array(real_deg["in_deg_hist"]) - np.array(syn_deg["in_deg_hist"][: len(real_deg["in_deg_hist"])])))
            )
            if len(syn_deg["in_deg_hist"]) >= len(real_deg["in_deg_hist"])
            else None,
        },
        "clustering_undirected": {
            "real": approx_clustering_undirected(real),
            "syn": approx_clustering_undirected(syn),
        },
        "path_length": {
            "real": approx_path_length(real),
            "syn": approx_path_length(syn),
        },
        "node_feature_ks": ks_feature_similarity(real.x.cpu().numpy(), syn.x.cpu().numpy()),
        "edge_feature_ks": ks_feature_similarity(real.edge_attr.cpu().numpy(), syn.edge_attr.cpu().numpy()),
        "node_feature_corr": feature_correlation_similarity(real.x.cpu().numpy(), syn.x.cpu().numpy()),
        "edge_feature_corr": feature_correlation_similarity(real.edge_attr.cpu().numpy(), syn.edge_attr.cpu().numpy()),
        "label_rates": {
            "node_pos_rate_real": float(real.y_node.float().mean().item()),
            "node_pos_rate_syn": float(syn.y_node.float().mean().item()),
            "edge_pos_rate_real": float(real.y_edge.float().mean().item()),
            "edge_pos_rate_syn": float(syn.y_edge.float().mean().item()),
        },
    }
    return rep


def fraud_pattern_report(real: Data, syn: Data) -> Dict[str, Any]:
    # Simple first-pass: compare degree distributions for fraudulent vs non-fraudulent nodes.
    def node_group_deg(data: Data) -> Dict[str, Any]:
        src = data.edge_index[0].cpu().numpy()
        dst = data.edge_index[1].cpu().numpy()
        n = int(data.num_nodes)
        out_deg = np.bincount(src, minlength=n)
        in_deg = np.bincount(dst, minlength=n)
        y = data.y_node.cpu().numpy().astype(int)
        sar = y == 1
        clean = y == 0
        return {
            "sar_in_mean": float(in_deg[sar].mean()) if sar.any() else None,
            "sar_out_mean": float(out_deg[sar].mean()) if sar.any() else None,
            "clean_in_mean": float(in_deg[clean].mean()) if clean.any() else None,
            "clean_out_mean": float(out_deg[clean].mean()) if clean.any() else None,
            "sar_frac": float(sar.mean()),
        }

    return {"node_degree_by_label": {"real": node_group_deg(real), "syn": node_group_deg(syn)}}

