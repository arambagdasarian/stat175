"""
Link-level empirical privacy audit for synthetic transaction graphs.

Threat model: the attacker observes the synthetic graph (structure + released node
features) and tries to predict whether a candidate pair (u, v) is a real edge in the
original graph. Adapted from link-stealing style settings (e.g. He et al., 2021) but
with the *synthetic graph* as the information source instead of GNN API outputs.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data


def _undirected_edge_set(edge_index: torch.Tensor) -> Set[Tuple[int, int]]:
    ei = edge_index.cpu().numpy()
    out: Set[Tuple[int, int]] = set()
    for i in range(ei.shape[1]):
        u, v = int(ei[0, i]), int(ei[1, i])
        if u == v:
            continue
        a, b = (u, v) if u < v else (v, u)
        out.add((a, b))
    return out


def _adj_sets(n: int, edge_index: torch.Tensor) -> List[Set[int]]:
    adj: List[Set[int]] = [set() for _ in range(n)]
    ei = edge_index.cpu().numpy()
    for i in range(ei.shape[1]):
        u, v = int(ei[0, i]), int(ei[1, i])
        if u == v:
            continue
        adj[u].add(v)
        adj[v].add(u)
    return adj


def _shortest_path_len(
    adj: List[Set[int]], u: int, v: int, cutoff: int = 20, max_visits: int = 4000
) -> int:
    if u == v:
        return 0
    if v in adj[u]:
        return 1
    q: deque[Tuple[int, int]] = deque([(u, 0)])
    seen = {u}
    while q:
        if len(seen) > max_visits:
            return cutoff + 2  # truncated search — treat as "far / unknown"
        x, d = q.popleft()
        if d >= cutoff:
            continue
        for w in adj[x]:
            if w == v:
                return d + 1
            if w not in seen:
                seen.add(w)
                q.append((w, d + 1))
    return cutoff + 1


def _cosine_sim(x: torch.Tensor, u: int, v: int) -> float:
    a = x[u].float().numpy()
    b = x[v].float().numpy()
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _precision_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    if k <= 0:
        return float("nan")
    order = np.argsort(-y_score)
    top = order[:k]
    return float(np.mean(y_true[top]))


@dataclass(frozen=True)
class LinkLeakageConfig:
    n_positive: int = 4000
    n_negative: int = 4000
    test_size: float = 0.25
    random_state: int = 7
    path_cutoff: int = 20


def run_link_leakage_audit(
    real: Data,
    syn: Data,
    *,
    cfg: Optional[LinkLeakageConfig] = None,
) -> Dict[str, object]:
    cfg = cfg or LinkLeakageConfig()
    n = int(real.num_nodes)
    if int(syn.num_nodes) != n:
        raise ValueError(
            f"Link leakage audit requires aligned node IDs (same num_nodes). "
            f"Got real={n}, syn={int(syn.num_nodes)}."
        )
    if int(syn.x.size(1)) != int(real.x.size(1)):
        raise ValueError(
            f"Node feature dim mismatch: real x {tuple(real.x.shape)} vs syn {tuple(syn.x.shape)}."
        )

    rng = np.random.default_rng(cfg.random_state)
    real_edges = _undirected_edge_set(real.edge_index)
    syn_edges = _undirected_edge_set(syn.edge_index)
    syn_adj = _adj_sets(n, syn.edge_index)

    degs = [len(syn_adj[i]) for i in range(n)]

    pos = list(real_edges)
    if len(pos) == 0:
        raise ValueError("No undirected edges in real graph.")
    rng.shuffle(pos)
    pos = pos[: min(cfg.n_positive, len(pos))]

    neg: List[Tuple[int, int]] = []
    attempts = 0
    max_attempts = max(50 * cfg.n_negative, 100_000)
    while len(neg) < cfg.n_negative and attempts < max_attempts:
        attempts += 1
        u = int(rng.integers(0, n))
        v = int(rng.integers(0, n))
        if u == v:
            continue
        a, b = (u, v) if u < v else (v, u)
        if (a, b) in real_edges:
            continue
        neg.append((a, b))
    # dedupe neg
    neg = list(dict.fromkeys(neg))[: cfg.n_negative]

    pairs = pos + neg
    y = np.array([1] * len(pos) + [0] * len(neg), dtype=np.int64)

    rows: List[List[float]] = []
    edge_ind_baseline: List[float] = []
    for u, v in pairs:
        und = (u, v) if u < v else (v, u)
        e_syn = 1.0 if und in syn_edges else 0.0
        edge_ind_baseline.append(e_syn)
        cn = float(len(syn_adj[u] & syn_adj[v]))
        sp = float(_shortest_path_len(syn_adj, u, v, cutoff=cfg.path_cutoff))
        du, dv = float(degs[u]), float(degs[v])
        csim = _cosine_sim(syn.x, u, v)
        rows.append([e_syn, sp, cn, du, dv, csim])

    X = np.asarray(rows, dtype=np.float64)
    y_score_edge = np.asarray(edge_ind_baseline, dtype=np.float64)

    X_tr, X_te, y_tr, y_te, s_tr, s_te = train_test_split(
        X, y, y_score_edge, test_size=cfg.test_size, random_state=cfg.random_state, stratify=y
    )
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=cfg.random_state)
    clf.fit(X_tr, y_tr)
    prob = clf.predict_proba(X_te)[:, 1]

    def pack_metrics(y_true: np.ndarray, scores: np.ndarray) -> Dict[str, float]:
        out: Dict[str, float] = {}
        if len(np.unique(y_true)) >= 2:
            out["roc_auc"] = float(roc_auc_score(y_true, scores))
            out["pr_auc"] = float(average_precision_score(y_true, scores))
        else:
            out["roc_auc"] = float("nan")
            out["pr_auc"] = float("nan")
        n_te = len(y_true)
        for frac in (0.01, 0.05, 0.10):
            k = max(1, int(frac * n_te))
            out[f"precision_at_{int(frac * 100)}pct"] = _precision_at_k(y_true, scores, k)
        return out

    metrics_full = pack_metrics(y_te, prob)
    metrics_edge_only = pack_metrics(y_te, s_te)

    return {
        "threat_model": (
            "Attacker observes synthetic graph (undirected edge presence, structure, "
            "released node features) and predicts held-out real edges vs sampled non-edges."
        ),
        "reference_attack_family": (
            "Link stealing / link inference (e.g. He et al., Stealing Links from GNNs, "
            "arXiv:2005.02131) adapted to synthetic-graph release instead of model outputs."
        ),
        "n_nodes": n,
        "n_pairs_total": int(len(pairs)),
        "n_positive": int(len(pos)),
        "n_negative": int(len(neg)),
        "feature_names": [
            "syn_edge_present",
            "syn_shortest_path_len",
            "syn_common_neighbors",
            "syn_deg_u",
            "syn_deg_v",
            "syn_cosine_feature_sim",
        ],
        "metrics_logistic_all_features": metrics_full,
        "metrics_baseline_syn_edge_indicator_only": metrics_edge_only,
        "metrics_trivial_random_guess": {"roc_auc": 0.5, "pr_auc": float(y_te.mean())},
        "note_multiple_samples": (
            "Optional: average syn_edge_present / scores over multiple GraphMaker draws "
            "— not used in this run (single synthetic graph)."
        ),
    }
