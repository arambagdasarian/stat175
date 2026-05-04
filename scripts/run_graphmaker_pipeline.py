"""
End-to-end: train GraphMaker-Async → sample to PyG → transfer experiment.

**Why defaults are small:** upstream ``train_amlworld_async`` trains on **every** upper-triangular
node pair (Θ(N²) batches per epoch). This preset is for a **conceptual** transfer demo (~10 min
CPU wall including short GNN transfer), not publication-quality GraphMaker.

Example (still under ~10 min on a typical laptop):
  WANDB_MODE=disabled python3 -m scripts.run_graphmaker_pipeline --data_dir data/raw

Larger / slower: ``--max_nodes 2500 --graphmaker_epochs 25 --full_transfer`` (and avoid CPU).
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=Path, default=root / "data" / "raw")
    p.add_argument("--out_dir", type=Path, default=root / "outputs" / "graphmaker")
    p.add_argument(
        "--max_transactions",
        type=int,
        default=80_000,
        help="Transaction rows before induced subgraph; smaller = faster CSV + smaller graphs.",
    )
    p.add_argument(
        "--max_nodes",
        type=int,
        default=1_000,
        help="Induced subgraph cap. GraphMaker cost is Θ(N²) per epoch — keep ≤1500 for CPU demos.",
    )
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--slice_mode", type=str, default="prefix", choices=["prefix", "balanced_edges"])
    p.add_argument(
        "--balance_scan_rows",
        type=int,
        default=500_000,
        help="balanced_edges only: CSV rows to scan (lower = faster IO).",
    )
    p.add_argument("--target_edge_pos_fraction", type=float, default=0.05)
    p.add_argument("--no_stratify_edges", action="store_true")
    p.add_argument("--no_stratify_nodes", action="store_true")
    p.add_argument(
        "--syn_edge_fraud_rate",
        type=float,
        default=None,
        help="Forwarded to graphmaker_sample_to_pyg (optional override).",
    )
    p.add_argument(
        "--syn_edge_fraud_rate_floor",
        type=float,
        default=0.0,
        help="Forwarded to graphmaker_sample_to_pyg: minimum synthetic edge fraud Bernoulli rate.",
    )
    p.add_argument(
        "--experiment_out_dir",
        type=Path,
        default=None,
        help="Where run_experiment writes transfer_*.json (default: same as --out_dir).",
    )
    p.add_argument(
        "--min_synthetic_nodes",
        type=int,
        default=None,
        help="Min syn nodes for run_experiment (default: min(8192, --max_nodes); Async sample matches training |V|).",
    )
    p.add_argument(
        "--graphmaker_epochs",
        type=int,
        default=5,
        help="GRAPHMAKER_NUM_EPOCHS (YAML default 200 is very slow).",
    )
    p.add_argument(
        "--graphmaker_patience",
        type=int,
        default=2,
        help="GRAPHMAKER_PATIENT_EPOCHS early-stop window.",
    )
    p.add_argument(
        "--graphmaker_train_batch",
        type=int,
        default=65_536,
        help="Larger train minibatch => fewer steps per epoch (GRAPHMAKER_BATCH_SIZE).",
    )
    p.add_argument(
        "--graphmaker_val_batch",
        type=int,
        default=131_072,
        help="Larger val minibatch => faster validation (GRAPHMAKER_VAL_BATCH_SIZE).",
    )
    p.add_argument(
        "--graphmaker_val_every",
        type=int,
        default=3,
        help="Run validation every N epochs (GRAPHMAKER_VAL_EVERY_EPOCHS).",
    )
    p.add_argument(
        "--full_transfer",
        action="store_true",
        help="Use run_experiment's default GNN epoch counts; default is short epochs for a quick demo.",
    )
    p.add_argument(
        "--full_graphmaker_train",
        action="store_true",
        help="Use YAML training defaults (200 epochs, original batch sizes). Not recommended on CPU.",
    )
    args = p.parse_args()
    if args.experiment_out_dir is None:
        args.experiment_out_dir = args.out_dir
    if args.min_synthetic_nodes is None:
        args.min_synthetic_nodes = min(8192, int(args.max_nodes))
    if args.slice_mode == "balanced_edges" and int(args.max_transactions) <= 0:
        print("error: balanced_edges requires a positive --max_transactions.", file=sys.stderr)
        return 1

    py = sys.executable
    env = os.environ.copy()
    env.setdefault("WANDB_MODE", "disabled")
    if args.full_graphmaker_train:
        env["GRAPHMAKER_USE_YAML_DEFAULTS"] = "1"
        env.pop("GRAPHMAKER_NUM_EPOCHS", None)
        env.pop("GRAPHMAKER_PATIENT_EPOCHS", None)
        env.pop("GRAPHMAKER_BATCH_SIZE", None)
        env.pop("GRAPHMAKER_VAL_BATCH_SIZE", None)
        env.pop("GRAPHMAKER_VAL_EVERY_EPOCHS", None)
    else:
        env.pop("GRAPHMAKER_USE_YAML_DEFAULTS", None)
        env["GRAPHMAKER_NUM_EPOCHS"] = str(int(args.graphmaker_epochs))
        env["GRAPHMAKER_PATIENT_EPOCHS"] = str(int(args.graphmaker_patience))
        env["GRAPHMAKER_BATCH_SIZE"] = str(int(args.graphmaker_train_batch))
        env["GRAPHMAKER_VAL_BATCH_SIZE"] = str(int(args.graphmaker_val_batch))
        env["GRAPHMAKER_VAL_EVERY_EPOCHS"] = str(int(args.graphmaker_val_every))

    train_cmd = [
        py,
        "-m",
        "scripts.graphmaker_train_amlworld",
        "--data_dir",
        str(args.data_dir),
        "--out_dir",
        str(args.out_dir),
        "--max_transactions",
        str(args.max_transactions),
        "--max_nodes",
        str(args.max_nodes),
        "--seed",
        str(args.seed),
        "--slice_mode",
        str(args.slice_mode),
        "--balance_scan_rows",
        str(args.balance_scan_rows),
        "--target_edge_pos_fraction",
        str(args.target_edge_pos_fraction),
    ]
    if args.no_stratify_edges:
        train_cmd.append("--no_stratify_edges")
    if args.no_stratify_nodes:
        train_cmd.append("--no_stratify_nodes")
    r = subprocess.run(train_cmd, cwd=str(root), env=env)
    if r.returncode != 0:
        return r.returncode

    graph_bin = (args.out_dir / "amlworld_graph.bin").resolve()
    env["AMLWORLD_DGL_GRAPH"] = str(graph_bin)
    ckpt = args.out_dir / "amlworld_cpts" / "Async_TX6_TE9.pth"
    syn_pt = args.out_dir / "synthetic_from_graphmaker.pt"
    sample_cmd = [
        py,
        "-m",
        "scripts.graphmaker_sample_to_pyg",
        "--model_path",
        str(ckpt),
        "--out_pt",
        str(syn_pt),
        "--data_dir",
        str(args.data_dir),
        "--max_transactions",
        str(args.max_transactions),
        "--slice_mode",
        str(args.slice_mode),
        "--balance_scan_rows",
        str(args.balance_scan_rows),
        "--target_edge_pos_fraction",
        str(args.target_edge_pos_fraction),
        "--syn_edge_fraud_rate_floor",
        str(float(args.syn_edge_fraud_rate_floor)),
    ]
    if args.no_stratify_edges:
        sample_cmd.append("--no_stratify_edges")
    if args.no_stratify_nodes:
        sample_cmd.append("--no_stratify_nodes")
    if args.syn_edge_fraud_rate is not None:
        sample_cmd += ["--syn_edge_fraud_rate", str(float(args.syn_edge_fraud_rate))]
    r = subprocess.run(sample_cmd, cwd=str(root), env=env)
    if r.returncode != 0:
        return r.returncode

    exp_cmd = [
        py,
        "-m",
        "scripts.run_experiment",
        "--data_dir",
        str(args.data_dir),
        "--max_transactions",
        str(args.max_transactions),
        "--seed",
        str(args.seed),
        "--out_dir",
        str(args.experiment_out_dir),
        "--generator",
        "from_pt",
        "--synthetic_pt",
        str(syn_pt),
        "--min_synthetic_nodes",
        str(int(args.min_synthetic_nodes)),
        "--slice_mode",
        str(args.slice_mode),
        "--balance_scan_rows",
        str(args.balance_scan_rows),
        "--target_edge_pos_fraction",
        str(args.target_edge_pos_fraction),
    ]
    if args.no_stratify_edges:
        exp_cmd.append("--no_stratify_edges")
    if args.no_stratify_nodes:
        exp_cmd.append("--no_stratify_nodes")
    if not args.full_transfer:
        exp_cmd += [
            "--node_epochs",
            "6",
            "--edge_epochs",
            "5",
            "--syn_node_epochs",
            "6",
            "--syn_edge_epochs",
            "5",
        ]
    r = subprocess.run(exp_cmd, cwd=str(root), env=env)
    return r.returncode


if __name__ == "__main__":
    raise SystemExit(main())
