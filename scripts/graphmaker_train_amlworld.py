"""
Train GraphMaker-Async on an AMLWorld-derived DGL graph (patched upstream in third_party/GraphMaker).

Requires: pip install dgl  (see https://www.dgl.ai/pages/start.html)

Example:
  WANDB_MODE=disabled python3 -m scripts.graphmaker_train_amlworld \\
    --data_dir data/raw --out_dir outputs/graphmaker --max_nodes 3000

Optional env (upstream `train_amlworld_async.py`): GRAPHMAKER_NUM_EPOCHS, GRAPHMAKER_PATIENT_EPOCHS
override `configs/amlworld/train_Async.yaml` for shorter smoke runs.

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
    p.add_argument("--max_transactions", type=int, default=200_000)
    p.add_argument("--max_nodes", type=int, default=3000)
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args()

    os.environ.setdefault("WANDB_MODE", "disabled")

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
