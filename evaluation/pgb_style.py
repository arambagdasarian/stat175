"""
PGB-style *structural* graph statistics (Liu et al., arXiv:2408.02928, Tables III–IV).

These are fidelity / utility diagnostics comparing a synthetic graph to the real graph on an
undirected simple projection restricted to vertices that appear in the **real** edge list
(and lie in the intersection of valid node ids for both graphs). They are **not** formal
(ε, δ)-DP guarantees for GraphMaker or degree-preserving generators.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import networkx as nx
from scipy.stats import ks_2samp
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score, normalized_mutual_info_score
from torch_geometric.data import Data

# Greedy modularity / diameter blow up on huge GCCs; keep PGB-style checks tractable.
_MAX_NODES_COMMUNITY = 10_000
_MAX_NODES_DIAMETER = 25_000
_MAX_GCC_EVC = 5_000
# Cap directed edges scanned when building NetworkX graphs (full AMLWorld ≈ 5M rows).
PGB_DEFAULT_MAX_DIRECTED_EDGE_ROWS = 600_000


def _nx_from_edge_index_capped(
    ei: np.ndarray,
    *,
    inv: np.ndarray,
    n_graph_nodes: int,
    max_directed_rows: int,
    seed: int,
    label: str,
) -> Tuple[nx.Graph, Dict[str, Any]]:
    """
    Undirected simple graph: map endpoints with `inv` (-1 = drop), drop self-loops, subsample rows.
    """
    info: Dict[str, Any] = {
        "edge_rows_input": int(ei.shape[1]),
        "label": label,
    }
    s = ei[0].astype(np.int64, copy=False)
    d = ei[1].astype(np.int64, copy=False)
    su = inv[s]
    dv = inv[d]
    ok = (su >= 0) & (dv >= 0) & (su != dv)
    su, dv = su[ok], dv[ok]
    if su.size == 0:
        G = nx.Graph()
        G.add_nodes_from(range(n_graph_nodes))
        info.update({"edge_rows_used": 0, "edge_subsampled": False, "undirected_unique_edges": 0})
        return G, info
    if su.size > max_directed_rows:
        rng = np.random.default_rng(seed)
        pick = rng.choice(su.size, size=max_directed_rows, replace=False)
        su, dv = su[pick], dv[pick]
        info["edge_subsampled"] = True
    else:
        info["edge_subsampled"] = False
    a = np.minimum(su, dv)
    b = np.maximum(su, dv)
    pairs = np.stack([a, b], axis=1)
    pairs = np.unique(pairs, axis=0)
    pairs = pairs[pairs[:, 0] != pairs[:, 1]]
    G = nx.Graph()
    G.add_nodes_from(range(n_graph_nodes))
    if pairs.size:
        G.add_edges_from(map(tuple, pairs))
    info["edge_rows_used"] = int(su.size)
    info["undirected_unique_edges"] = int(pairs.shape[0])
    return G, info


def _scalar_re(real: float, syn: float) -> float:
    if not (np.isfinite(real) and np.isfinite(syn)):
        return float("nan")
    ar = abs(float(real))
    if ar < 1e-15 and abs(float(syn)) < 1e-15:
        return 0.0
    if ar < 1e-15:
        return float("inf")
    return abs(float(syn) - float(real)) / ar


def _mean_re_scalar(res: Dict[str, Any]) -> float:
    keys = [
        "Q1_|V|",
        "Q2_|E|",
        "Q3_triangles",
        "Q4_avg_degree",
        "Q5_degree_variance",
        "Q7_diameter",
        "Q8_avg_shortest_path",
        "Q10_GCC_transitivity",
        "Q11_ACC",
        "Q13_modularity",
        "Q14_assortativity",
    ]
    acc: List[float] = []
    for k in keys:
        v = res.get(k, {}).get("RE")
        if v is None or not np.isfinite(v) or v == float("inf"):
            continue
        acc.append(min(float(v), 1e6))
    return float(np.mean(acc)) if acc else float("nan")


def _build_graphs_subsampled(
    real: Data,
    syn: Data,
    *,
    seed: int = 0,
    max_directed_edges: int = PGB_DEFAULT_MAX_DIRECTED_EDGE_ROWS,
) -> Tuple[nx.Graph, nx.Graph, Dict[str, Any]]:
    """
    When syn.num_nodes << real.num_nodes, align by taking the `k = syn.num_nodes` **highest-degree**
    real vertices (by total degree in the real transaction slice) and inducing the undirected
    simple subgraph on that set; compare to full `syn`. This avoids empty induced subgraphs from
    purely random sampling on sparse graphs.
    """
    _ = seed
    k = int(syn.num_nodes)
    ei_r = real.edge_index.cpu().numpy()
    ei_s = syn.edge_index.cpu().numpy()
    deg = np.bincount(ei_r.flatten(), minlength=int(real.num_nodes))
    top = np.argsort(deg)[::-1][:k]
    pick = np.sort(np.unique(top.astype(np.int64)))
    if pick.size < k:
        pad = np.arange(int(real.num_nodes), dtype=np.int64)
        pad = np.setdiff1d(pad, pick, assume_unique=False)[: k - int(pick.size)]
        pick = np.sort(np.concatenate([pick, pad]))
    pick = pick[:k]
    pick_set = {int(x) for x in pick.tolist()}
    old_to_new = {int(n): i for i, n in enumerate(sorted(pick_set))}

    inv_r = np.full(int(real.num_nodes), -1, dtype=np.int32)
    for n in pick_set:
        inv_r[n] = old_to_new[n]
    Gr, xr = _nx_from_edge_index_capped(
        ei_r,
        inv=inv_r,
        n_graph_nodes=len(pick_set),
        max_directed_rows=max_directed_edges,
        seed=seed + 11,
        label="real_subsampled",
    )

    inv_s = np.arange(k, dtype=np.int32)
    Gs, xs = _nx_from_edge_index_capped(
        ei_s,
        inv=inv_s,
        n_graph_nodes=k,
        max_directed_rows=max_directed_edges,
        seed=seed + 17,
        label="syn",
    )

    meta = {
        "method": "subsampled_real_nodes_to_match_syn",
        "k": k,
        "n_real_nodes_pyG": int(real.num_nodes),
        "n_syn_nodes_pyG": int(syn.num_nodes),
        "n_edges_undirected_real": int(Gr.number_of_edges()),
        "n_edges_undirected_syn": int(Gs.number_of_edges()),
        "pgb_edge_cap_directed": int(max_directed_edges),
        "real_edge_projection_stats": xr,
        "syn_edge_projection_stats": xs,
    }
    return Gr, Gs, meta


def _build_graphs_on_real_active(
    real: Data,
    syn: Data,
    *,
    seed: int = 0,
    max_directed_edges: int = PGB_DEFAULT_MAX_DIRECTED_EDGE_ROWS,
) -> Tuple[nx.Graph, nx.Graph, Dict[str, Any]]:
    n_cap = min(int(real.num_nodes), int(syn.num_nodes))
    ei_r = real.edge_index.cpu().numpy()
    ei_s = syn.edge_index.cpu().numpy()
    src_r, dst_r = ei_r[0], ei_r[1]
    m = (src_r < n_cap) & (dst_r < n_cap) & (src_r != dst_r)
    V_star = np.unique(np.concatenate([src_r[m], dst_r[m]])).astype(np.int64)
    old_to_new = {int(o): i for i, o in enumerate(V_star.tolist())}

    inv = np.full(max(int(real.num_nodes), int(syn.num_nodes)), -1, dtype=np.int32)
    for node, j in old_to_new.items():
        inv[int(node)] = int(j)

    inv_r = inv[: int(real.num_nodes)]
    inv_s = inv[: int(syn.num_nodes)]

    Gr, xr = _nx_from_edge_index_capped(
        ei_r,
        inv=inv_r,
        n_graph_nodes=len(V_star),
        max_directed_rows=max_directed_edges,
        seed=seed + 3,
        label="real_aligned",
    )
    Gs, xs = _nx_from_edge_index_capped(
        ei_s,
        inv=inv_s,
        n_graph_nodes=len(V_star),
        max_directed_rows=max_directed_edges,
        seed=seed + 5,
        label="syn_aligned",
    )

    meta = {
        "method": "real_active_endpoints_intersection",
        "n_real_nodes_pyG": int(real.num_nodes),
        "n_syn_nodes_pyG": int(syn.num_nodes),
        "n_active_vertices_V_star": int(len(V_star)),
        "n_edges_undirected_real": int(Gr.number_of_edges()),
        "n_edges_undirected_syn": int(Gs.number_of_edges()),
        "pgb_edge_cap_directed": int(max_directed_edges),
        "real_edge_projection_stats": xr,
        "syn_edge_projection_stats": xs,
    }
    return Gr, Gs, meta


def _largest_cc(g: nx.Graph) -> nx.Graph:
    if g.number_of_nodes() == 0:
        return g
    gcc = max(nx.connected_components(g), key=len)
    return g.subgraph(gcc).copy()


def _avg_shortest_path_sample(g: nx.Graph, *, sample: int = 400, seed: int = 7) -> Optional[float]:
    if g.number_of_nodes() == 0:
        return None
    rng = np.random.default_rng(seed)
    nodes = list(g.nodes())
    take = min(sample, len(nodes))
    picks = rng.choice(nodes, size=take, replace=False)
    lengths: List[float] = []
    for s in picks:
        sp = nx.single_source_shortest_path_length(g, s, cutoff=30)
        lengths.extend(float(d) for t, d in sp.items() if t != s)
    if not lengths:
        return None
    return float(np.mean(lengths))


def _degree_hist_l1(g1: nx.Graph, g2: nx.Graph) -> Tuple[float, float, float, float]:
    d1 = [g1.degree(i) for i in g1.nodes()]
    d2 = [g2.degree(i) for i in g2.nodes()]
    mx = max(max(d1, default=0), max(d2, default=0)) + 1
    h1 = np.bincount(np.array(d1, dtype=int), minlength=mx).astype(np.float64)
    h2 = np.bincount(np.array(d2, dtype=int), minlength=mx).astype(np.float64)
    s1 = h1.sum() or 1.0
    s2 = h2.sum() or 1.0
    p = h1 / s1
    q = h2 / s2
    eps = 1e-12
    kl_pr = float(np.sum(q * np.log((q + eps) / (p + eps))))
    kl_rp = float(np.sum(p * np.log((p + eps) / (q + eps))))
    ks = ks_2samp(np.array(d1, dtype=float), np.array(d2, dtype=float), alternative="two-sided", mode="auto")
    return kl_pr, kl_rp, float(ks.statistic), float(ks.pvalue)


def _distance_hist_l1(g1: nx.Graph, g2: nx.Graph, *, seed: int = 7) -> float:
    rng = np.random.default_rng(seed)
    bins = np.arange(0.5, 31.5, 1.0)

    def hist(g: nx.Graph) -> np.ndarray:
        if g.number_of_nodes() == 0:
            return np.zeros(len(bins) - 1, dtype=np.float64)
        nodes = list(g.nodes())
        take = min(250, len(nodes))
        picks = rng.choice(nodes, size=take, replace=False)
        lengths: List[int] = []
        for s in picks:
            sp = nx.single_source_shortest_path_length(g, s, cutoff=30)
            lengths.extend(int(d) for t, d in sp.items() if t != s and d <= 30)
        if not lengths:
            return np.zeros(len(bins) - 1, dtype=np.float64)
        h, _ = np.histogram(lengths, bins=bins)
        h = h.astype(np.float64)
        s = h.sum() or 1.0
        return h / s

    h1 = hist(g1)
    h2 = hist(g2)
    return float(np.sum(np.abs(h1 - h2)))


def _modularity(g: nx.Graph) -> Optional[float]:
    if g.number_of_edges() == 0:
        return None
    gcc = _largest_cc(g)
    if gcc.number_of_nodes() == 0:
        return None
    try:
        comms = nx.community.greedy_modularity_communities(gcc, weight=None)
        return float(nx.community.modularity(gcc, comms))
    except Exception:
        return None


def _paired_community_labels(
    g1: nx.Graph, g2: nx.Graph, *, max_nodes: int = _MAX_NODES_COMMUNITY
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    gcc1 = _largest_cc(g1)
    gcc2 = _largest_cc(g2)
    nodes = sorted(set(gcc1.nodes()) & set(gcc2.nodes()))
    m = len(nodes)
    if m == 0 or m > max_nodes:
        return None, None
    mapping = {old: i for i, old in enumerate(nodes)}
    s1 = nx.relabel_nodes(g1.subgraph(nodes).copy(), mapping, copy=True)
    s2 = nx.relabel_nodes(g2.subgraph(nodes).copy(), mapping, copy=True)
    try:
        c1 = nx.community.greedy_modularity_communities(s1, weight=None)
        c2 = nx.community.greedy_modularity_communities(s2, weight=None)
    except Exception:
        return None, None
    lab1 = np.zeros(m, dtype=np.int64)
    lab2 = np.zeros(m, dtype=np.int64)
    for i, c in enumerate(c1):
        for v in c:
            lab1[int(v)] = i
    for i, c in enumerate(c2):
        for v in c:
            lab2[int(v)] = i
    return lab1, lab2


def _diameter(g: nx.Graph) -> Optional[float]:
    gcc = _largest_cc(g)
    if gcc.number_of_nodes() == 0:
        return None
    if gcc.number_of_nodes() > _MAX_NODES_DIAMETER:
        return None
    try:
        return float(nx.diameter(gcc))
    except Exception:
        return None


def _triangles_count(g: nx.Graph) -> int:
    return int(sum(nx.triangles(g).values()) // 3)


def _assortativity(g: nx.Graph) -> Optional[float]:
    if g.number_of_edges() == 0:
        return None
    try:
        return float(nx.degree_assortativity_coefficient(g))
    except Exception:
        return None


def pgb_fifteen_style(
    real: Data,
    syn: Data,
    *,
    seed: int = 7,
    max_directed_edges: int = PGB_DEFAULT_MAX_DIRECTED_EDGE_ROWS,
) -> Dict[str, Any]:
    n_r, n_s = int(real.num_nodes), int(syn.num_nodes)
    if n_r == n_s:
        Gr, Gs, meta = _build_graphs_on_real_active(real, syn, seed=seed, max_directed_edges=max_directed_edges)
        if int(meta.get("n_active_vertices_V_star", 0)) == 0 or int(meta.get("n_edges_undirected_real", 0)) == 0:
            Gr, Gs, meta = _build_graphs_subsampled(
                real, syn, seed=seed, max_directed_edges=max_directed_edges
            )
    else:
        Gr, Gs, meta = _build_graphs_subsampled(real, syn, seed=seed, max_directed_edges=max_directed_edges)
    n = Gr.number_of_nodes()

    tr = _triangles_count(Gr)
    ts = _triangles_count(Gs)
    dr = np.array([Gr.degree(i) for i in Gr.nodes()], dtype=float) if n else np.array([], dtype=float)
    ds = np.array([Gs.degree(i) for i in Gs.nodes()], dtype=float) if n else np.array([])

    vr = float(n)
    vs = float(n)  # same vertex set
    er = float(Gr.number_of_edges())
    es = float(Gs.number_of_edges())
    mean_dr = float(dr.mean()) if n else 0.0
    mean_ds = float(ds.mean()) if n else 0.0
    var_dr = float(dr.var()) if n else 0.0
    var_ds = float(ds.var()) if n else 0.0

    gcc_r = _largest_cc(Gr)
    gcc_s = _largest_cc(Gs)
    trans_r = float(nx.transitivity(gcc_r)) if gcc_r.number_of_nodes() else 0.0
    trans_s = float(nx.transitivity(gcc_s)) if gcc_s.number_of_nodes() else 0.0
    acc_r = float(nx.average_clustering(Gr)) if n else 0.0
    acc_s = float(nx.average_clustering(Gs)) if n else 0.0

    diam_r = _diameter(Gr)
    diam_s = _diameter(Gs)
    asp_r = _avg_shortest_path_sample(Gr, seed=seed)
    asp_s = _avg_shortest_path_sample(Gs, seed=seed + 1)

    mod_r = _modularity(Gr)
    mod_s = _modularity(Gs)
    ass_r = _assortativity(Gr)
    ass_s = _assortativity(Gs)

    queries: Dict[str, Any] = {
        "Q1_|V|": {"real": vr, "syn": vs, "RE": _scalar_re(vr, vs), "metric": "RE"},
        "Q2_|E|": {"real": er, "syn": es, "RE": _scalar_re(er, es), "metric": "RE"},
        "Q3_triangles": {"real": float(tr), "syn": float(ts), "RE": _scalar_re(float(tr), float(ts)), "metric": "RE"},
        "Q4_avg_degree": {"real": mean_dr, "syn": mean_ds, "RE": _scalar_re(mean_dr, mean_ds), "metric": "RE"},
        "Q5_degree_variance": {"real": var_dr, "syn": var_ds, "RE": _scalar_re(var_dr, var_ds), "metric": "RE"},
        "Q7_diameter": {
            "real": diam_r,
            "syn": diam_s,
            "RE": _scalar_re(diam_r, diam_s) if diam_r is not None and diam_s is not None else None,
            "metric": "RE",
        },
        "Q8_avg_shortest_path": {
            "real": asp_r,
            "syn": asp_s,
            "RE": _scalar_re(asp_r, asp_s) if asp_r is not None and asp_s is not None else None,
            "metric": "RE",
        },
        "Q10_GCC_transitivity": {"real": trans_r, "syn": trans_s, "RE": _scalar_re(trans_r, trans_s), "metric": "RE"},
        "Q11_ACC": {"real": acc_r, "syn": acc_s, "RE": _scalar_re(acc_r, acc_s), "metric": "RE"},
        "Q13_modularity": {
            "real": mod_r,
            "syn": mod_s,
            "RE": _scalar_re(mod_r, mod_s) if mod_r is not None and mod_s is not None else None,
            "metric": "RE",
        },
        "Q14_assortativity": {
            "real": ass_r,
            "syn": ass_s,
            "RE": _scalar_re(ass_r, ass_s) if ass_r is not None and ass_s is not None else None,
            "metric": "RE",
        },
    }

    if n > 0:
        kl_pr, kl_rp, ks_stat, ks_p = _degree_hist_l1(Gr, Gs)
        queries["Q6_degree_distribution"] = {
            "KL_syn_vs_real": kl_pr,
            "KL_real_vs_syn": kl_rp,
            "KS_statistic": ks_stat,
            "KS_pvalue": ks_p,
            "metric": "KL,KS",
        }
        queries["Q9_distance_distribution"] = {
            "L1_hist_diff": _distance_hist_l1(Gr, Gs, seed=seed),
            "metric": "L1(normalized hists)",
        }
    else:
        queries["Q6_degree_distribution"] = {"metric": "KL,KS", "skipped": "empty V*"}
        queries["Q9_distance_distribution"] = {"metric": "L1", "skipped": "empty V*"}

    lr, ls = _paired_community_labels(Gr, Gs)
    if lr is not None and ls is not None and lr.shape == ls.shape:
        queries["Q12_community_detection"] = {
            "NMI": float(normalized_mutual_info_score(lr, ls)),
            "ARI": float(adjusted_rand_score(lr, ls)),
            "AMI": float(adjusted_mutual_info_score(lr, ls)),
            "metric": "NMI,ARI,AMI",
        }
    else:
        queries["Q12_community_detection"] = {
            "NMI": None,
            "ARI": None,
            "AMI": None,
            "metric": "NMI,ARI,AMI",
            "skipped": "GCC intersection empty or too large for greedy modularity",
        }

    evc_mae: Optional[float] = None
    evc_skip = "GCC intersection empty or too large for EVC"
    nodes_ev = sorted(set(gcc_r.nodes()) & set(gcc_s.nodes()))
    if 0 < len(nodes_ev) <= _MAX_GCC_EVC:
        try:
            mp = {old: i for i, old in enumerate(nodes_ev)}
            hr = nx.relabel_nodes(gcc_r.subgraph(nodes_ev).copy(), mp, copy=True)
            hs = nx.relabel_nodes(gcc_s.subgraph(nodes_ev).copy(), mp, copy=True)
            er_vec = nx.eigenvector_centrality_numpy(hr, max_iter=100)
            es_vec = nx.eigenvector_centrality_numpy(hs, max_iter=100)
            order = list(range(len(nodes_ev)))
            vr_ = np.array([er_vec[i] for i in order], dtype=float)
            vs_ = np.array([es_vec[i] for i in order], dtype=float)
            evc_mae = float(np.mean(np.abs(vr_ - vs_)))
        except Exception:
            evc_mae = None
            evc_skip = "eigenvector computation failed"
    queries["Q15_eigenvector_centrality"] = {"MAE": evc_mae, "metric": "MAE", **({"skipped": evc_skip} if evc_mae is None else {})}

    mre_acc: Optional[float] = None
    if n > 0:
        mre_acc = float(np.mean(np.abs(ds - dr) / np.maximum(dr, 1.0)))

    queries["Q11_ACC_MRE_per_node"] = {"MRE": mre_acc, "metric": "MRE (Table IV)"}

    out = {
        "reference": "Liu et al., PGB arXiv:2408.02928 Table III-IV",
        "graph": (
            "Undirected simple graphs: if num_nodes matches, V* = endpoints of non-self-loop real edges "
            "restricted to shared id range; else top-degree-k real induced subgraph (k=num_syn_nodes) vs full syn."
        ),
        "vertex_graph_meta": meta,
        "queries": queries,
        "summary_mean_RE_scalar_queries": _mean_re_scalar(queries),
        "privacy_note": (
            "PGB privacy in the paper is ε-DP from their synthetic mechanism; our generators do not "
            "emit ε. These numbers are structural fidelity / utility only."
        ),
    }
    return out


def empirical_privacy_proxies(
    real: Data,
    syn: Data,
    *,
    neighbor_samples: int = 500,
    seed: int = 7,
    max_directed_edges: int = PGB_DEFAULT_MAX_DIRECTED_EDGE_ROWS,
) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    er = real.edge_index.cpu().numpy()
    es = syn.edge_index.cpu().numpy()
    n = min(int(real.num_nodes), int(syn.num_nodes))

    def directed_set_cap(ei: np.ndarray, cap: int, rng_seed: int) -> Set[Tuple[int, int]]:
        e = int(ei.shape[1])
        rng = np.random.default_rng(rng_seed)
        if e <= cap:
            idx = np.arange(e, dtype=np.int64)
        else:
            idx = rng.choice(e, size=cap, replace=False)
        u = ei[0, idx].astype(np.int64, copy=False)
        v = ei[1, idx].astype(np.int64, copy=False)
        return set(zip(u.tolist(), v.tolist()))

    Sr = directed_set_cap(er, max_directed_edges, seed)
    Ss = directed_set_cap(es, max_directed_edges, seed + 101)
    inter = len(Sr & Ss)
    union = len(Sr | Ss) or 1
    jacc = inter / union

    feat_block: Dict[str, Any] = {"compatible": int(real.num_nodes) == int(syn.num_nodes)}
    if feat_block["compatible"]:
        xr = real.x.cpu().numpy()
        xs = syn.x.cpu().numpy()
        diff = np.abs(xr - xs).mean(axis=1)
        feat_block["mean_l1"] = float(np.mean(diff))
        feat_block["median_l1"] = float(np.median(diff))
    else:
        feat_block["mean_l1"] = None
        feat_block["median_l1"] = None
        feat_block["reason"] = "num_nodes mismatch"

    out_deg_r: Dict[int, Set[int]] = {}
    for i in range(er.shape[1]):
        u, v = int(er[0, i]), int(er[1, i])
        if u < n and v < n:
            out_deg_r.setdefault(u, set()).add(v)
    out_deg_s: Dict[int, Set[int]] = {}
    for i in range(es.shape[1]):
        u, v = int(es[0, i]), int(es[1, i])
        if u < n and v < n:
            out_deg_s.setdefault(u, set()).add(v)
    common = [u for u in out_deg_r if u in out_deg_s and (out_deg_r[u] or out_deg_s[u])]
    n_sampled = 0
    if not common:
        nj = None
    else:
        take = min(neighbor_samples, len(common))
        n_sampled = int(take)
        pick = rng.choice(common, size=take, replace=False)
        jacs = []
        for u in pick:
            a, b = out_deg_r[u], out_deg_s[u]
            uu = len(a | b) or 1
            jacs.append(len(a & b) / uu)
        nj = float(np.mean(jacs))

    return {
        "directed_edge_jaccard": {
            "intersection": inter,
            "union": int(union),
            "jaccard": float(jacc),
            "n_real_edges": int(real.edge_index.size(1)),
            "n_syn_edges": int(syn.edge_index.size(1)),
            "approximate_on_sample": int(max_directed_edges),
        },
        "aligned_node_feature_l1": feat_block,
        "neighbor_jaccard_out": {"mean_jaccard": nj, "n_samples": n_sampled},
        "disclaimer": "Empirical proxies only; not epsilon-DP.",
    }


def pgb_style_bundle(
    real: Data,
    syn: Data,
    *,
    syn_meta: Optional[Dict[str, Any]] = None,
    seed: int = 7,
    max_directed_edges: int = PGB_DEFAULT_MAX_DIRECTED_EDGE_ROWS,
) -> Dict[str, Any]:
    return {
        "synthetic_meta": syn_meta or {},
        "pgb_fifteen": pgb_fifteen_style(real, syn, seed=seed, max_directed_edges=max_directed_edges),
        "empirical_privacy": empirical_privacy_proxies(
            real, syn, seed=seed, max_directed_edges=max_directed_edges
        ),
    }
