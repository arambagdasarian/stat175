"""
Generate figures for README: subgraph sample, degree histograms, path-length bar,
and refresh utility/degree bar charts from the latest transfer JSON.

Run from project root:
  python3 -m scripts.visualize_outputs
  python3 -m scripts.visualize_outputs --transfer_json outputs/transfer_..._graphmaker.json
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Set, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch

from data.amlworld import load_amlworld_hi_small_pyg
from generators.degree_preserving import DegreePreservingGeneratorConfig, generate_degree_preserving_synthetic
from generators.graphmaker_bridge import load_synthetic_from_torch


def _find_latest_transfer_json(outputs_dir: Path) -> Optional[Path]:
    cands = sorted(outputs_dir.glob("transfer_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0] if cands else None


def _find_transfer_json_for_figures(outputs_dir: Path, explicit: Optional[Path]) -> Optional[Path]:
    if explicit is not None:
        p = explicit.expanduser()
        return p if p.is_file() else None
    gm = sorted(outputs_dir.glob("transfer_*_graphmaker.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if gm:
        return gm[0]
    dp = sorted(outputs_dir.glob("transfer_*_degree_preserving.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if dp:
        return dp[0]
    return _find_latest_transfer_json(outputs_dir)


def _extract_subgraph_nodes(
    edge_index: torch.Tensor,
    y_edge: torch.Tensor,
    max_nodes: int = 90,
    seed: int = 42,
) -> Set[int]:
    """Seed from laundering edges, expand by one hop, cap size."""
    rng = np.random.default_rng(seed)
    ei = edge_index.cpu().numpy()
    ye = y_edge.cpu().numpy()
    launder_idx = np.where(ye == 1)[0]
    if len(launder_idx) == 0:
        # fallback: random edges
        launder_idx = rng.choice(ei.shape[1], size=min(30, ei.shape[1]), replace=False)

    picked = set()
    for _ in range(min(25, len(launder_idx))):
        j = int(rng.choice(launder_idx))
        picked.add(int(ei[0, j]))
        picked.add(int(ei[1, j]))

    # one-hop expansion
    neighbors: Set[int] = set(picked)
    for e in range(ei.shape[1]):
        u, v = int(ei[0, e]), int(ei[1, e])
        if u in picked or v in picked:
            neighbors.add(u)
            neighbors.add(v)

    nodes = list(neighbors)
    if len(nodes) > max_nodes:
        nodes = list(rng.choice(nodes, size=max_nodes, replace=False))
    return set(int(x) for x in nodes)


def plot_subgraph_sample(
    data_dir: Path,
    out_path: Path,
    *,
    max_transactions: int = 120000,
    max_nodes: int = 85,
) -> None:
    real, _ = load_amlworld_hi_small_pyg(data_dir, max_transactions=max_transactions, seed=7)
    nodes = _extract_subgraph_nodes(real.edge_index, real.y_edge, max_nodes=max_nodes)
    node_list = sorted(nodes)
    idx_map = {old: i for i, old in enumerate(node_list)}
    ei = real.edge_index.cpu().numpy()
    ye = real.y_edge.cpu().numpy()
    yn = real.y_node.cpu().numpy()

    G = nx.DiGraph()
    for n in node_list:
        G.add_node(idx_map[n], fraud=bool(yn[n]))
    for e in range(ei.shape[1]):
        u, v = int(ei[0, e]), int(ei[1, e])
        if u in nodes and v in nodes:
            G.add_edge(idx_map[u], idx_map[v], laundering=bool(ye[e]))

    pos = nx.spring_layout(G, seed=42, k=0.45, iterations=50)
    fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True)

    node_colors = ["#c0392b" if G.nodes[n]["fraud"] else "#3498db" for n in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=120, alpha=0.9, ax=ax)
    edge_colors = []
    widths = []
    for u, v in G.edges():
        lau = G[u][v].get("laundering", False)
        edge_colors.append("#e74c3c" if lau else "#bdc3c7")
        widths.append(2.0 if lau else 0.6)
    nx.draw_networkx_edges(
        G, pos, edge_color=edge_colors, width=widths, alpha=0.75, arrows=True, arrowsize=10, ax=ax
    )
    ax.set_title(
        "Sample induced subgraph (red nodes: derived SAR accounts; red edges: laundering tx)",
        fontsize=11,
    )
    ax.axis("off")
    from matplotlib.patches import Patch

    leg = [
        Patch(facecolor="#3498db", label="Account (clean)"),
        Patch(facecolor="#c0392b", label="Account (touches laundering)"),
        Patch(facecolor="#e74c3c", label="Edge: laundering"),
        Patch(facecolor="#bdc3c7", label="Edge: non-laundering"),
    ]
    ax.legend(handles=leg, loc="upper left", frameon=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _synthetic_for_figures(data_dir: Path, root: Path, max_transactions: int, seed: int):
    """Prefer GraphMaker export if present; otherwise degree-preserving baseline."""
    real, _ = load_amlworld_hi_small_pyg(data_dir, max_transactions=max_transactions, seed=seed)
    pt = root / "outputs/graphmaker/synthetic_from_graphmaker.pt"
    if pt.is_file():
        return real, load_synthetic_from_torch(pt, real, seed=seed)[0]
    return real, generate_degree_preserving_synthetic(
        real, config=DegreePreservingGeneratorConfig(seed=seed)
    )[0]


def plot_degree_histograms(
    data_dir: Path,
    out_path: Path,
    *,
    root: Path,
    max_transactions: int = 120000,
) -> None:
    real, syn = _synthetic_for_figures(data_dir, root, max_transactions, seed=7)
    n = int(real.num_nodes)
    src_r = real.edge_index[0].cpu().numpy()
    dst_r = real.edge_index[1].cpu().numpy()
    src_s = syn.edge_index[0].cpu().numpy()
    dst_s = syn.edge_index[1].cpu().numpy()
    in_r = np.bincount(dst_r, minlength=n)
    out_r = np.bincount(src_r, minlength=n)
    in_s = np.bincount(dst_s, minlength=n)
    out_s = np.bincount(src_s, minlength=n)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    bins = np.arange(0, min(40, max(in_r.max(), out_r.max(), in_s.max(), out_s.max()) + 2))

    for ax, arr_r, arr_s, title in [
        (axes[0], in_r, in_s, "In-degree distribution (real vs synthetic)"),
        (axes[1], out_r, out_s, "Out-degree distribution (real vs synthetic)"),
    ]:
        ax.hist(arr_r, bins=bins, alpha=0.55, label="Real", color="#2980b9", density=True)
        ax.hist(arr_s, bins=bins, alpha=0.55, label="Synthetic", color="#e67e22", density=True)
        ax.set_xlabel("Degree")
        ax.set_ylabel("Density")
        ax.set_title(title)
        ax.legend(frameon=False)
        ax.set_yscale("log")

    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _metric_float(x) -> float:
    if x is None:
        return 0.0
    return float(x)


def plot_path_length_bar(json_path: Path, out_path: Path) -> None:
    obj = json.loads(json_path.read_text())
    pl = obj["result"]["similarity"]["path_length"]
    real_m = _metric_float(pl["real"].get("mean"))
    syn_m = _metric_float(pl["syn"].get("mean"))
    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    x = [0, 1]
    ax.bar(x, [real_m, syn_m], width=0.55, color=["#2980b9", "#e67e22"])
    ax.set_xticks(x, ["Real graph", "Synthetic graph"])
    ax.set_ylabel("Mean shortest-path length (approx., undirected projection)")
    ax.set_title("Higher-order structure: approximate mean path length (undefined → 0 if no pairs)")
    ax.grid(axis="y", alpha=0.25)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_utility_and_degree_means(
    json_path: Path,
    fig_dir: Path,
    *,
    utility_png: str = "utility_transfer.png",
    degree_png: Optional[str] = "degree_means.png",
) -> None:
    """Build utility bar chart (and optionally degree means) from a transfer JSON."""
    res = json.loads(json_path.read_text())["result"]
    fig_dir.mkdir(parents=True, exist_ok=True)

    def bar_two(ax, labels, a, b, a_name, b_name, title):
        x = np.arange(len(labels))
        w = 0.35
        aa = [_metric_float(v) for v in a]
        bb = [_metric_float(v) for v in b]
        ax.bar(x - w / 2, aa, width=w, label=a_name)
        ax.bar(x + w / 2, bb, width=w, label=b_name)
        ax.set_xticks(x, labels)
        ax.set_title(title)
        ax.legend(frameon=False)
        ax.grid(axis="y", alpha=0.25)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    node_real = res["node"]["real_train_eval"]["test"]
    node_tr = res["node"]["transfer_eval_on_real_test"]
    bar_two(
        axes[0],
        ["ROC-AUC", "PR-AUC", "F1"],
        [node_real["roc_auc"], node_real["pr_auc"], node_real["f1"]],
        [node_tr["roc_auc"], node_tr["pr_auc"], node_tr["f1"]],
        "Real→Real (test)",
        "Synthetic→Real (test)",
        "Node task: account fraud",
    )
    edge_real = res["edge"]["real_train_eval"]["test"]
    edge_tr = res["edge"]["transfer_eval_on_real_test"]
    bar_two(
        axes[1],
        ["ROC-AUC", "PR-AUC", "F1"],
        [edge_real["roc_auc"], edge_real["pr_auc"], edge_real["f1"]],
        [edge_tr["roc_auc"], edge_tr["pr_auc"], edge_tr["f1"]],
        "Real→Real (test)",
        "Synthetic→Real (test)",
        "Edge task: transaction fraud",
    )
    fig.suptitle("Utility transfer: train on real vs synthetic, evaluate on real", fontsize=12)
    fig.savefig(fig_dir / utility_png, dpi=200)
    plt.close(fig)

    if degree_png is None:
        return

    sim = res["similarity"]
    fig, ax = plt.subplots(1, 1, figsize=(10, 4), constrained_layout=True)
    keys = ["in_deg_mean", "out_deg_mean"]
    real_vals = [sim["degree"]["real"][k] for k in keys]
    syn_vals = [sim["degree"]["syn"][k] for k in keys]
    bar_two(
        ax,
        ["Mean in-degree", "Mean out-degree"],
        real_vals,
        syn_vals,
        "Real",
        "Synthetic",
        "Degree means (real vs synthetic training graph)",
    )
    fig.savefig(fig_dir / degree_png, dpi=200)
    plt.close(fig)


def main() -> int:
    import argparse

    root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description="Generate README figures from transfer JSONs.")
    ap.add_argument(
        "--transfer_json",
        type=Path,
        default=None,
        help="Single transfer JSON for utility/degree/path figures (default: prefer *_graphmaker.json).",
    )
    args = ap.parse_args()

    data_dir = root / "data" / "raw"
    fig_dir = root / "outputs" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    out_dir = root / "outputs"

    json_path = _find_transfer_json_for_figures(out_dir, args.transfer_json)
    if json_path is None:
        print("No outputs/transfer_*.json found; skip metric figures.")
    else:
        plot_utility_and_degree_means(json_path, fig_dir)
        plot_path_length_bar(json_path, fig_dir / "path_length_mean.png")
        print(f"Wrote utility_transfer.png, degree_means.png, path_length_mean.png using {json_path.name}")

    for tag in ("graphmaker", "degree_preserving"):
        matches = sorted(out_dir.glob(f"transfer_*_{tag}.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        if matches:
            plot_utility_and_degree_means(
                matches[0], fig_dir, utility_png=f"utility_transfer_{tag}.png", degree_png=None
            )
            print(f"Wrote utility_transfer_{tag}.png from {matches[0].name}")

    if not (data_dir / "HI-Small_Trans.csv").exists():
        print("Missing data/raw CSVs; skip subgraph and degree histograms.")
        return 0

    plot_subgraph_sample(data_dir, fig_dir / "subgraph_sample.png")
    print(f"Wrote {fig_dir / 'subgraph_sample.png'}")
    plot_degree_histograms(data_dir, fig_dir / "degree_hist_in_out.png", root=root)
    print(f"Wrote {fig_dir / 'degree_hist_in_out.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
