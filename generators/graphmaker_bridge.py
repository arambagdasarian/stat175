"""
Load a synthetic graph produced *outside* this repo (e.g. after sampling with GraphMaker)
into the same PyTorch Geometric schema as `load_amlworld_hi_small_pyg`.

GraphMaker (Li et al., TMLR 2024; arXiv:2310.13833) trains on **undirected** graphs with
**categorical** node attributes and no edge attributes in the paper's setup; AMLWorld is
**directed** and has continuous / one-hot features plus edge attributes. You must convert
GraphMaker samples to match our tensor shapes before training here — see README.

Official code: https://github.com/Graph-COM/GraphMaker
Paper: https://arxiv.org/abs/2310.13833
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
from torch_geometric.data import Data

from data.amlworld import _build_splits_binary


def _attach_masks_if_needed(
    syn: Data,
    *,
    seed: int,
    train_size: float = 0.6,
    val_size: float = 0.2,
) -> None:
    n = int(syn.num_nodes)
    if (
        not hasattr(syn, "node_train_mask")
        or syn.node_train_mask is None
        or not syn.node_train_mask.any()
    ):
        n_y = syn.y_node.cpu().numpy()
        n_tr, n_va, n_te = _build_splits_binary(n_y, train_size, val_size, seed)
        syn.node_train_mask = torch.zeros(n, dtype=torch.bool)
        syn.node_val_mask = torch.zeros(n, dtype=torch.bool)
        syn.node_test_mask = torch.zeros(n, dtype=torch.bool)
        syn.node_train_mask[n_tr] = True
        syn.node_val_mask[n_va] = True
        syn.node_test_mask[n_te] = True

    if (
        not hasattr(syn, "edge_train_idx")
        or syn.edge_train_idx is None
        or syn.edge_train_idx.numel() == 0
    ):
        e_y = syn.y_edge.cpu().numpy()
        e_tr, e_va, e_te = _build_splits_binary(e_y, train_size, val_size, seed + 1)
        syn.edge_train_idx = torch.tensor(e_tr, dtype=torch.long)
        syn.edge_val_idx = torch.tensor(e_va, dtype=torch.long)
        syn.edge_test_idx = torch.tensor(e_te, dtype=torch.long)


def load_synthetic_from_torch(
    path: Path,
    reference_real: Data,
    *,
    seed: int = 7,
    train_size: float = 0.6,
    val_size: float = 0.2,
) -> Tuple[Data, Dict[str, Any]]:
    """
    Load a file saved as `torch.save` containing either:
    - a `torch_geometric.data.Data` object, or
    - a dict with key `data` holding that object.

    Validates feature dimensions against `reference_real` (same AMLWorld preprocessing).
    If `edge_attr` is missing, fills zeros with shape (E, F_edge) matching the real graph.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)

    try:
        obj = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        obj = torch.load(path, map_location="cpu")
    if isinstance(obj, Data):
        syn = obj
    elif isinstance(obj, dict) and "data" in obj and isinstance(obj["data"], Data):
        syn = obj["data"]
    else:
        raise TypeError(
            "Expected torch.save file to contain a Data object or dict['data'] -> Data; "
            f"got {type(obj)}"
        )

    exp_x = int(reference_real.x.size(1))
    exp_e = int(reference_real.edge_attr.size(1))
    got_x = int(syn.x.size(1))
    if got_x > exp_x:
        raise ValueError(
            f"Synthetic x has dim {got_x} but real has {exp_x} (cannot truncate safely)."
        )
    if got_x < exp_x:
        pad = exp_x - got_x
        syn.x = torch.cat(
            [syn.x, torch.zeros((syn.num_nodes, pad), dtype=syn.x.dtype, device=syn.x.device)],
            dim=1,
        )

    e_cnt = int(syn.edge_index.size(1))
    if not hasattr(syn, "edge_attr") or syn.edge_attr is None:
        syn.edge_attr = torch.zeros((e_cnt, exp_e), dtype=torch.float32)
    elif syn.edge_attr.dim() != 2 or syn.edge_attr.size(1) != exp_e:
        raise ValueError(
            f"Synthetic edge_attr shape {tuple(syn.edge_attr.shape)}; "
            f"expected (*, {exp_e}) to match real AMLWorld edge features."
        )

    if syn.edge_attr.size(0) != e_cnt:
        raise ValueError("edge_attr rows must equal number of edges in edge_index.")

    _attach_masks_if_needed(syn, seed=seed, train_size=train_size, val_size=val_size)

    meta: Dict[str, Any] = {
        "generator": "external_torch_file",
        "path": str(path.resolve()),
        "note": "Typically GraphMaker or another generative model after conversion to PyG.",
        "references": {
            "graphmaker_paper": "https://arxiv.org/abs/2310.13833",
            "graphmaker_code": "https://github.com/Graph-COM/GraphMaker",
        },
    }
    if got_x < exp_x:
        meta["x_padded_trailing_columns"] = int(exp_x - got_x)
    return syn, meta
