"""
Standalone snowball-sampling diagnostic — requires only numpy + csv module.

Reads HI-Small_Trans.csv directly, builds the transaction graph, runs snowball
sampling from the top-k highest-out-degree SAR seed nodes, and prints a
side-by-side comparison of the raw slice vs the snowball subgraph.

This mirrors exactly what generators/snowball_sampling.py does inside the
full PyTorch pipeline, but with no torch/DGL dependency so it can run in
the base environment.

Usage:
    python3 scripts/snowball_stats.py --data_dir data/raw
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np


# ── CSV loading ───────────────────────────────────────────────────────────────

def load_transactions(csv_path: Path, max_rows: int, scan_rows: int, target_fraud_frac: float) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, int
]:
    """
    Returns (src, dst, is_fraud_edge, amounts, num_account_ids).
    Uses 'balanced_edges' logic: scan scan_rows rows, subsample to
    max_rows edges with target_fraud_frac laundering edges.
    """
    rng = np.random.default_rng(7)
    fraud_rows, clean_rows = [], []
    acct_map: Dict[str, int] = {}

    def get_id(a: str) -> int:
        if a not in acct_map:
            acct_map[a] = len(acct_map)
        return acct_map[a]

    # Column layout: Timestamp(0), From Bank(1), Account(2), To Bank(3),
    # Account(4)[=dest], Amount Received(5), Receiving Currency(6),
    # Amount Paid(7), Payment Currency(8), Payment Format(9), Is Laundering(10)
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for i, row in enumerate(reader):
            if i >= scan_rows:
                break
            if len(row) < 11:
                continue
            src_acct = get_id(row[2])
            dst_acct = get_id(row[4])
            fraud = int(row[10]) if row[10].strip() else 0
            amt = float(row[7]) if row[7].strip() else 0.0
            entry = (src_acct, dst_acct, fraud, amt)
            if fraud:
                fraud_rows.append(entry)
            else:
                clean_rows.append(entry)

    n_fraud_target = int(max_rows * target_fraud_frac)
    n_clean_target = max_rows - n_fraud_target

    rng.shuffle(fraud_rows)
    rng.shuffle(clean_rows)
    selected = fraud_rows[:n_fraud_target] + clean_rows[:n_clean_target]
    rng.shuffle(selected)

    src_arr = np.array([r[0] for r in selected], dtype=np.int64)
    dst_arr = np.array([r[1] for r in selected], dtype=np.int64)
    fraud_arr = np.array([r[2] for r in selected], dtype=np.int64)
    amt_arr = np.array([r[3] for r in selected], dtype=np.float64)
    return src_arr, dst_arr, fraud_arr, amt_arr, len(acct_map)


# ── Node labels ───────────────────────────────────────────────────────────────

def derive_sar_nodes(src: np.ndarray, dst: np.ndarray, fraud: np.ndarray, n_nodes: int) -> np.ndarray:
    """A node is SAR if it is any endpoint of a fraudulent edge."""
    sar = np.zeros(n_nodes, dtype=np.int64)
    fraud_mask = fraud == 1
    sar[src[fraud_mask]] = 1
    sar[dst[fraud_mask]] = 1
    return sar


# ── Undirected adjacency ──────────────────────────────────────────────────────

def build_adj(src: np.ndarray, dst: np.ndarray, n_nodes: int) -> List[List[int]]:
    adj: List[List[int]] = [[] for _ in range(n_nodes)]
    for u, v in zip(src.tolist(), dst.tolist()):
        if u != v:
            adj[u].append(v)
            adj[v].append(u)
    return adj


# ── Graph stats ───────────────────────────────────────────────────────────────

def graph_stats(label: str, src: np.ndarray, dst: np.ndarray, fraud_e: np.ndarray,
                sar: np.ndarray, n_nodes: int) -> None:
    n_edges = len(src)
    n_sar = int(sar.sum())
    n_fraud_e = int(fraud_e.sum())
    out_deg = np.bincount(src, minlength=n_nodes)
    density = n_edges / max(1, n_nodes * (n_nodes - 1))
    avg_deg = 2 * n_edges / max(1, n_nodes)
    sar_out_deg = out_deg[sar == 1] if n_sar > 0 else np.array([0])
    clean_out_deg = out_deg[sar == 0]

    # Connected-component check (rough: count nodes with degree > 0)
    both_deg = np.bincount(np.concatenate([src, dst]), minlength=n_nodes)
    n_isolated = int((both_deg == 0).sum())

    print(f"\n{'─'*60}")
    print(f"  {label}")
    print(f"{'─'*60}")
    print(f"  nodes            : {n_nodes:>10,}")
    print(f"  edges            : {n_edges:>10,}")
    print(f"  isolated nodes   : {n_isolated:>10,}  ({100*n_isolated/max(1,n_nodes):.1f}%)")
    print(f"  density          : {density:>10.2e}")
    print(f"  avg undirected   : {avg_deg:>10.2f}  (edges×2 / nodes)")
    print(f"  SAR nodes        : {n_sar:>10,}  ({100*n_sar/max(1,n_nodes):.2f}%)")
    print(f"  fraud edges      : {n_fraud_e:>10,}  ({100*n_fraud_e/max(1,n_edges):.2f}%)")
    print(f"  SAR out-deg  mean: {sar_out_deg.mean():>10.1f}   max={sar_out_deg.max()}")
    print(f"  clean out-deg mean:{clean_out_deg.mean():>10.2f}  max={clean_out_deg.max()}")


# ── Snowball sampling ─────────────────────────────────────────────────────────

def snowball(
    src: np.ndarray, dst: np.ndarray, fraud_e: np.ndarray,
    sar: np.ndarray, n_nodes: int,
    *,
    max_nodes: int, top_k_seeds: int, wave_limit: int, seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, List[Dict]]:
    rng = np.random.default_rng(seed)
    out_deg = np.bincount(src, minlength=n_nodes)
    in_deg = np.bincount(dst, minlength=n_nodes)
    total_deg = out_deg + in_deg

    sar_idx = np.flatnonzero(sar == 1)
    if len(sar_idx) == 0:
        print("WARNING: no SAR nodes found — seeding on top-degree nodes instead")
        sar_idx = np.arange(n_nodes)

    k = min(top_k_seeds, len(sar_idx))
    ranked = sar_idx[np.argsort(total_deg[sar_idx])[::-1]]
    seeds = ranked[:k].tolist()

    adj = build_adj(src, dst, n_nodes)
    visited: Set[int] = set(seeds)
    frontier: List[int] = list(seeds)
    wave_stats = [{"wave": 0, "new_nodes": len(seeds), "total": len(seeds)}]

    for wn in range(1, wave_limit + 1):
        if len(visited) >= max_nodes:
            break
        next_f: List[int] = []
        for fi in rng.permutation(len(frontier)).tolist():
            u = frontier[fi]
            for ni in rng.permutation(len(adj[u])).tolist():
                v = adj[u][ni]
                if v not in visited:
                    visited.add(v)
                    next_f.append(v)
                    if len(visited) >= max_nodes:
                        break
            if len(visited) >= max_nodes:
                break
        wave_stats.append({"wave": wn, "new_nodes": len(next_f), "total": len(visited)})
        if not next_f:
            break
        frontier = next_f

    old_nodes = sorted(visited)
    remap = {old: i for i, old in enumerate(old_nodes)}
    sset = set(old_nodes)

    mask = np.array(
        [(int(s) in sset and int(d) in sset) for s, d in zip(src.tolist(), dst.tolist())],
        dtype=bool,
    )
    new_src = np.array([remap[int(s)] for s in src[mask]], dtype=np.int64)
    new_dst = np.array([remap[int(d)] for d in dst[mask]], dtype=np.int64)
    new_fe = fraud_e[mask]
    new_sar = sar[old_nodes]
    n_sub = len(old_nodes)
    return new_src, new_dst, new_fe, new_sar, n_sub, wave_stats


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=Path, default=Path("data/raw"))
    p.add_argument("--max_transactions", type=int, default=200_000,
                   help="Transactions to sample after scanning.")
    p.add_argument("--scan_rows", type=int, default=1_000_000,
                   help="CSV rows to scan when building the balanced slice.")
    p.add_argument("--target_fraud_frac", type=float, default=0.06,
                   help="Target fraction of fraud edges in the loaded slice.")
    p.add_argument("--max_nodes", type=int, default=1_200,
                   help="Node budget for GraphMaker (N² constraint; keep ≤1500).")
    p.add_argument("--top_k_seeds", type=int, default=20,
                   help="Top-k SAR hub seeds for snowball.")
    p.add_argument("--wave_limit", type=int, default=15,
                   help="Max snowball expansion waves.")
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args()

    csv_path = args.data_dir / "HI-Small_Trans.csv"
    if not csv_path.is_file():
        print(f"ERROR: {csv_path} not found", file=sys.stderr)
        return 1

    print(f"Loading up to {args.max_transactions:,} transactions "
          f"(scanning {args.scan_rows:,} rows, target fraud={args.target_fraud_frac:.0%}) …",
          flush=True)
    src, dst, fraud_e, _, n_nodes = load_transactions(
        csv_path, args.max_transactions, args.scan_rows, args.target_fraud_frac
    )
    sar = derive_sar_nodes(src, dst, fraud_e, n_nodes)

    # ── Full slice stats ──────────────────────────────────────────────────────
    graph_stats("FULL LOADED SLICE (before any subgraph sampling)",
                src, dst, fraud_e, sar, n_nodes)

    # ── Comparison: random induced ────────────────────────────────────────────
    rng = np.random.default_rng(args.seed)
    ri_node_list = sorted(rng.choice(n_nodes, size=min(args.max_nodes, n_nodes), replace=False).tolist())
    ri_nodes = set(ri_node_list)
    ri_mask = np.array([(int(s) in ri_nodes and int(d) in ri_nodes)
                         for s, d in zip(src.tolist(), dst.tolist())], dtype=bool)
    ri_remap = {old: i for i, old in enumerate(ri_node_list)}
    ri_src = np.array([ri_remap[int(s)] for s in src[ri_mask]], dtype=np.int64)
    ri_dst = np.array([ri_remap[int(d)] for d in dst[ri_mask]], dtype=np.int64)
    ri_fe = fraud_e[ri_mask]
    ri_sar = sar[ri_node_list]
    graph_stats(f"RANDOM INDUCED (max_nodes={args.max_nodes})",
                ri_src, ri_dst, ri_fe, ri_sar, len(ri_node_list))

    # ── Snowball stats ────────────────────────────────────────────────────────
    print(f"\nRunning snowball (top_k_seeds={args.top_k_seeds}, "
          f"wave_limit={args.wave_limit}, max_nodes={args.max_nodes}) …", flush=True)
    sb_src, sb_dst, sb_fe, sb_sar, sb_n, waves = snowball(
        src, dst, fraud_e, sar, n_nodes,
        max_nodes=args.max_nodes,
        top_k_seeds=args.top_k_seeds,
        wave_limit=args.wave_limit,
        seed=args.seed,
    )
    graph_stats(f"SNOWBALL SUBGRAPH (max_nodes={args.max_nodes})",
                sb_src, sb_dst, sb_fe, sb_sar, sb_n)

    print("\n  Wave-by-wave expansion:")
    for w in waves:
        print(f"    wave {w['wave']:2d}: +{w['new_nodes']:5d} nodes  →  {w['total']:5d} total")

    # ── Summary comparison table ──────────────────────────────────────────────
    def _fmt(s, d, fe, n):
        ne = len(s)
        density = ne / max(1, n * (n - 1))
        fraud_frac = fe.mean() if len(fe) > 0 else 0.0
        isolated = int((np.bincount(np.concatenate([s, d]), minlength=n) == 0).sum())
        return ne, density, fraud_frac, isolated

    ri_ne, ri_den, ri_ff, ri_iso = _fmt(ri_src, ri_dst, ri_fe, len(ri_nodes))
    sb_ne, sb_den, sb_ff, sb_iso = _fmt(sb_src, sb_dst, sb_fe, sb_n)

    print("\n" + "="*60)
    print("  SUMMARY (what GraphMaker will see)")
    print("="*60)
    print(f"  {'Metric':<25} {'Random induced':>15} {'Snowball':>15}")
    print(f"  {'-'*55}")
    print(f"  {'Nodes':<25} {len(ri_nodes):>15,} {sb_n:>15,}")
    print(f"  {'Edges':<25} {ri_ne:>15,} {sb_ne:>15,}")
    print(f"  {'Isolated nodes':<25} {ri_iso:>15,} {sb_iso:>15,}")
    print(f"  {'Density':<25} {ri_den:>15.2e} {sb_den:>15.2e}")
    print(f"  {'Fraud edge frac':<25} {ri_ff:>14.2%} {sb_ff:>14.2%}")
    print("="*60)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
