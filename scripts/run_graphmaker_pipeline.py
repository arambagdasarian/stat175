"""
End-to-end: train GraphMaker-Async → sample to PyG → utility transfer experiment.

Example:
  WANDB_MODE=disabled python3 -m scripts.run_graphmaker_pipeline \\
    --data_dir data/raw --max_transactions 300000 --max_nodes 3000
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
    p.add_argument("--max_transactions", type=int, default=300_000)
    p.add_argument("--max_nodes", type=int, default=3000)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument(
        "--experiment_out_dir",
        type=Path,
        default=root / "outputs",
        help="Where run_experiment writes transfer_*.json",
    )
    p.add_argument(
        "--graphmaker_epochs",
        type=int,
        default=None,
        help="If set, sets GRAPHMAKER_NUM_EPOCHS for training (else YAML default).",
    )
    p.add_argument(
        "--graphmaker_patience",
        type=int,
        default=None,
        help="If set, sets GRAPHMAKER_PATIENT_EPOCHS for training.",
    )
    args = p.parse_args()

    py = sys.executable
    env = os.environ.copy()
    env.setdefault("WANDB_MODE", "disabled")
    if args.graphmaker_epochs is not None:
        env["GRAPHMAKER_NUM_EPOCHS"] = str(args.graphmaker_epochs)
    if args.graphmaker_patience is not None:
        env["GRAPHMAKER_PATIENT_EPOCHS"] = str(args.graphmaker_patience)

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
    ]
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
    ]
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
    ]
    r = subprocess.run(exp_cmd, cwd=str(root), env=env)
    return r.returncode


if __name__ == "__main__":
    raise SystemExit(main())
