"""
Load a trained GraphMaker-Async checkpoint on amlworld, draw one sample, export PyG Data.

Requires:
  - AMLWORLD_DGL_GRAPH pointing to the same amlworld_graph.bin used in training
  - pip install dgl

Example:
  export AMLWORLD_DGL_GRAPH=$(pwd)/outputs/graphmaker/amlworld_graph.bin
  python3 -m scripts.graphmaker_sample_to_pyg \\
    --model_path outputs/graphmaker/amlworld_cpts/Async_TX6_TE9.pth \\
    --out_pt outputs/graphmaker/synthetic_from_graphmaker.pt \\
    --data_dir data/raw --max_transactions 300000
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

import torch


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    gm = root / "third_party" / "GraphMaker"
    sys.path.insert(0, str(gm))

    p = argparse.ArgumentParser()
    p.add_argument("--model_path", type=Path, required=True)
    p.add_argument("--out_pt", type=Path, default=root / "outputs" / "graphmaker" / "synthetic_from_graphmaker.pt")
    p.add_argument("--data_dir", type=Path, default=root / "data" / "raw")
    p.add_argument("--max_transactions", type=int, default=300_000)
    args = p.parse_args()

    if not os.environ.get("AMLWORLD_DGL_GRAPH"):
        default_g = (root / "outputs" / "graphmaker" / "amlworld_graph.bin").resolve()
        if default_g.is_file():
            os.environ["AMLWORLD_DGL_GRAPH"] = str(default_g)
        else:
            print("Set AMLWORLD_DGL_GRAPH to amlworld_graph.bin from training.", file=sys.stderr)
            return 1

    # Project `data` package shadows GraphMaker's `data.py`; import project code first, then GraphMaker.
    sys.path.insert(0, str(root))
    from data.amlworld import load_amlworld_hi_small_pyg
    from generators.graphmaker_aml.sample_to_pyg import graphmaker_sample_to_pyg

    real, meta = load_amlworld_hi_small_pyg(
        args.data_dir, max_transactions=args.max_transactions, seed=7
    )
    edge_rate = float(meta.get("edge_label_pos_rate", real.y_edge.float().mean().item()))

    try:
        state_dict = torch.load(args.model_path, map_location="cpu", weights_only=False)
    except TypeError:
        state_dict = torch.load(args.model_path, map_location="cpu")

    train_yaml_data = state_dict["train_yaml_data"]
    model_name = train_yaml_data["meta_data"]["variant"]
    if model_name != "Async":
        print("This exporter only supports Async checkpoints for amlworld.", file=sys.stderr)
        return 1

    # Project package `data` is already imported; load GraphMaker's `data.py` under a distinct name.
    _gm_data_path = gm / "data.py"
    _spec = importlib.util.spec_from_file_location("graphmaker_upstream_data", _gm_data_path)
    _gm_data = importlib.util.module_from_spec(_spec)
    assert _spec.loader is not None
    _spec.loader.exec_module(_gm_data)
    load_dataset = _gm_data.load_dataset
    preprocess = _gm_data.preprocess

    from model import ModelAsync
    from setup_utils import set_seed

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    g_real = load_dataset("amlworld")
    X_one_hot_3d_real, Y_real, E_one_hot_real, X_marginal, Y_marginal, E_marginal, X_cond_Y_marginals = preprocess(
        g_real
    )

    X_marginal = X_marginal.to(device)
    Y_marginal = Y_marginal.to(device)
    E_marginal = E_marginal.to(device)
    X_cond_Y_marginals = X_cond_Y_marginals.to(device)
    num_nodes = Y_real.size(0)

    model = ModelAsync(
        X_marginal=X_marginal,
        Y_marginal=Y_marginal,
        E_marginal=E_marginal,
        mlp_X_config=train_yaml_data["mlp_X"],
        gnn_E_config=train_yaml_data["gnn_E"],
        num_nodes=num_nodes,
        **train_yaml_data["diffusion"],
    ).to(device)
    model.graph_encoder.pred_X.load_state_dict(state_dict["pred_X_state_dict"])
    model.graph_encoder.pred_E.load_state_dict(state_dict["pred_E_state_dict"])
    model.eval()

    set_seed()
    with torch.no_grad():
        X_0_one_hot, Y_0_one_hot, E_0 = model.sample(batch_size=8192, num_workers=0)

    syn = graphmaker_sample_to_pyg(
        X_0_one_hot.cpu(),
        Y_0_one_hot.cpu(),
        E_0.cpu(),
        ref_x_dim=int(real.x.size(1)),
        ref_edge_attr_dim=int(real.edge_attr.size(1)),
        edge_fraud_rate=min(edge_rate, 0.5),
        seed=7,
    )

    args.out_pt.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"data": syn, "meta": {"source": "graphmaker_async", "checkpoint": str(args.model_path)}}, args.out_pt)
    print(f"Wrote {args.out_pt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
