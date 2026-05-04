"""
Train GraphMaker-Async on an AMLWorld-derived DGL graph (patched upstream in third_party/GraphMaker).

Requires: ``pip install -e ".[graphmaker]"`` then ``python3 -m scripts.verify_graphmaker_env``.
DGL needs a **PyTorch version that matches a bundled GraphBolt** dylib (see pyproject ``graphmaker`` pins).

Fast conceptual train (matches pipeline defaults; GraphMaker is Θ(N²) per epoch):
  WANDB_MODE=disabled python3 -m scripts.graphmaker_train_amlworld --data_dir data/raw

Full YAML run (slow on CPU): set env ``GRAPHMAKER_USE_YAML_DEFAULTS=1`` before invoking.

Optional env (``train_amlworld_async.py``): ``GRAPHMAKER_NUM_EPOCHS``, ``GRAPHMAKER_PATIENT_EPOCHS``,
``GRAPHMAKER_BATCH_SIZE``, ``GRAPHMAKER_VAL_BATCH_SIZE``, ``GRAPHMAKER_VAL_EVERY_EPOCHS``.

Checkpoint: outputs/graphmaker/amlworld_cpts/Async_TX*_TE*.pth
Saved DGL graph (for sampling): outputs/graphmaker/amlworld_graph.bin
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    gm = root / "third_party" / "GraphMaker"
    if not gm.is_dir():
        print("Clone GraphMaker first: third_party/GraphMaker missing.", file=sys.stderr)
        return 1
    try:
        import dgl  # noqa: F401
    except Exception as e:
        print(
            "DGL is required for GraphMaker. Install a wheel matching your PyTorch build:\n"
            "  https://www.dgl.ai/pages/start.html\n"
            f"Import error: {e}",
            file=sys.stderr,
        )
        return 1

    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=Path, default=root / "data" / "raw")
    p.add_argument("--out_dir", type=Path, default=root / "outputs" / "graphmaker")
    p.add_argument(
        "--max_transactions",
        type=int,
        default=80_000,
        help="Smaller default keeps induced subgraph training closer to ~10 min CPU demos.",
    )
    p.add_argument(
        "--max_nodes",
        type=int,
        default=1_000,
        help="GraphMaker cost scales as N² over all node pairs; use ≤1500 for quick runs.",
    )
    p.add_argument(
        "--fraud_enriched",
        action="store_true",
        help="BFS-expand from node positives (undirected) then cap at max_nodes — higher fraud density for GraphMaker.",
    )
    p.add_argument("--neighbor_hops", type=int, default=2, help="With --fraud_enriched: BFS depth around positive nodes.")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument(
        "--slice_mode",
        type=str,
        default="prefix",
        choices=["prefix", "balanced_edges"],
        help="prefix=first N rows; balanced_edges=scan then subsample to target laundering rate (needs --max_transactions).",
    )
    p.add_argument(
        "--balance_scan_rows",
        type=int,
        default=2_000_000,
        help="With balanced_edges: max CSV rows to read before subsampling.",
    )
    p.add_argument(
        "--target_edge_pos_fraction",
        type=float,
        default=0.05,
        help="With balanced_edges: target fraction of laundering edges in the final slice.",
    )
    p.add_argument(
        "--no_stratify_edges",
        action="store_true",
        help="Disable sklearn stratified 60/20/20 edge splits when viable (use legacy split).",
    )
    p.add_argument(
        "--no_stratify_nodes",
        action="store_true",
        help="Disable sklearn stratified 60/20/20 node splits when viable.",
    )
    args = p.parse_args()

    os.environ.setdefault("WANDB_MODE", "disabled")
    if not os.environ.get("GRAPHMAKER_USE_YAML_DEFAULTS"):
        os.environ.setdefault("GRAPHMAKER_NUM_EPOCHS", "5")
        os.environ.setdefault("GRAPHMAKER_PATIENT_EPOCHS", "2")
        os.environ.setdefault("GRAPHMAKER_BATCH_SIZE", "65536")
        os.environ.setdefault("GRAPHMAKER_VAL_BATCH_SIZE", "131072")
        os.environ.setdefault("GRAPHMAKER_VAL_EVERY_EPOCHS", "3")
    # else: keep upstream YAML train.* (200 epochs, etc.)

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    graph_path = out_dir / "amlworld_graph.bin"
    # Absolute path: training subprocess uses cwd=third_party/GraphMaker.
    os.environ["AMLWORLD_DGL_GRAPH"] = str(graph_path.resolve())

    # Build graph from AMLWorld (project root on PYTHONPATH)
    sys.path.insert(0, str(root))
    from generators.graphmaker_aml.build import build_dgl_for_graphmaker, save_training_graph

    g = build_dgl_for_graphmaker(
        args.data_dir,
        max_transactions=args.max_transactions,
        max_nodes=args.max_nodes,
        seed=args.seed,
        fraud_enriched=bool(args.fraud_enriched),
        neighbor_hops=int(args.neighbor_hops),
        slice_mode=str(args.slice_mode),
        balance_scan_rows=int(args.balance_scan_rows),
        target_edge_pos_fraction=float(args.target_edge_pos_fraction),
        stratify_edges_if_possible=not bool(args.no_stratify_edges),
        stratify_nodes_if_possible=not bool(args.no_stratify_nodes),
    )
    save_training_graph(g, graph_path)
    print(f"Saved DGL graph for GraphMaker: {graph_path} ({g.num_nodes()} nodes)", flush=True)

    # Run training inside GraphMaker directory
    train_py = gm / "train_amlworld_async.py"
    cmd = [sys.executable, str(train_py), "-d", "amlworld"]
    env = os.environ.copy()
    env["WANDB_MODE"] = "disabled"
    env["AMLWORLD_DGL_GRAPH"] = str(graph_path.resolve())
    r = subprocess.run(cmd, cwd=str(gm), env=env)
    if r.returncode != 0:
        return r.returncode
    # Checkpoints are written under third_party/GraphMaker/amlworld_cpts/ (training cwd).
    src_cpts = gm / "amlworld_cpts"
    dst_cpts = out_dir / "amlworld_cpts"
    if src_cpts.is_dir():
        if dst_cpts.exists():
            shutil.rmtree(dst_cpts)
        shutil.copytree(src_cpts, dst_cpts)
        print(f"Copied checkpoints to {dst_cpts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
