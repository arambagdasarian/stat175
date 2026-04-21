from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Dict, Optional, Tuple

import torch
from torch_geometric.data import Data

from evaluation.similarity import fraud_pattern_report, structural_and_feature_report
from generators.degree_preserving import (
    DegreePreservingGeneratorConfig,
    generate_degree_preserving_synthetic,
)
from models.gnn import EdgeClassifier, GraphSAGEEncoder, NodeClassifier
from models.train_utils import Metrics, evaluate_edge, evaluate_node, train_edge_classifier, train_node_classifier


@dataclass(frozen=True)
class ModelConfig:
    hidden_dim: int = 128
    emb_dim: int = 128
    num_layers: int = 2
    dropout: float = 0.1
    # Real graph: moderate epochs (large graph, expensive).
    node_epochs: int = 12
    edge_epochs: int = 8
    syn_node_epochs: int = 12
    syn_edge_epochs: int = 6
    lr: float = 1e-3
    weight_decay: float = 1e-4
    seed: int = 7
    warm_start_from_real: bool = False


def _init_models(real_data, cfg: ModelConfig) -> Tuple[NodeClassifier, EdgeClassifier]:
    encoder = GraphSAGEEncoder(
        in_channels=int(real_data.x.size(1)),
        hidden_channels=cfg.hidden_dim,
        out_channels=cfg.emb_dim,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
    )
    node_model = NodeClassifier(encoder=encoder, emb_dim=cfg.emb_dim)
    edge_model = EdgeClassifier(
        emb_dim=cfg.emb_dim,
        edge_feat_dim=int(real_data.edge_attr.size(1)),
        hidden_dim=cfg.hidden_dim,
        dropout=cfg.dropout,
    )
    return node_model, edge_model


def run_transfer_experiment(
    real_data,
    *,
    model_cfg: Optional[ModelConfig] = None,
    gen_cfg: Optional[DegreePreservingGeneratorConfig] = None,
    synthetic_bundle: Optional[Tuple[Data, Dict[str, Any]]] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    Core experiment:
    - Train on real → evaluate on real
    - Train on synthetic → evaluate on real (same real test split)

    If `synthetic_bundle` is provided as `(syn_data, syn_meta)`, it is used instead of
    the built-in degree-preserving generator (e.g. GraphMaker output after conversion).
    """
    mcfg = model_cfg or ModelConfig()
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Real baseline ---
    node_model_real, edge_model_real = _init_models(real_data, mcfg)
    node_metrics_real = train_node_classifier(
        node_model_real,
        real_data,
        lr=mcfg.lr,
        weight_decay=mcfg.weight_decay,
        epochs=mcfg.node_epochs,
        device=device,
    )

    # Edge model uses the same encoder (inside node_model_real)
    edge_metrics_real = train_edge_classifier(
        node_model_real,
        edge_model_real,
        real_data,
        lr=mcfg.lr,
        weight_decay=mcfg.weight_decay,
        epochs=mcfg.edge_epochs,
        device=device,
    )

    # --- Synthetic graph ---
    if synthetic_bundle is not None:
        syn_data, syn_meta = synthetic_bundle
    else:
        syn_data, syn_meta = generate_degree_preserving_synthetic(real_data, config=gen_cfg)

    # --- Train on synthetic, evaluate on real ---
    node_model_syn, edge_model_syn = _init_models(syn_data, mcfg)
    if mcfg.warm_start_from_real:
        node_model_syn.load_state_dict(node_model_real.state_dict())
        edge_model_syn.load_state_dict(edge_model_real.state_dict())
    _ = train_node_classifier(
        node_model_syn,
        syn_data,
        lr=mcfg.lr,
        weight_decay=mcfg.weight_decay,
        epochs=mcfg.syn_node_epochs,
        device=device,
    )

    # Evaluate node model trained on synthetic on REAL test split
    sel = real_data.node_test_mask.to(device)
    node_transfer = evaluate_node(node_model_syn, real_data, sel, device)[0]

    # Edge transfer: train edge head on synthetic using synthetic embeddings, then evaluate on real
    edge_metrics_syn_train = train_edge_classifier(
        node_model_syn,
        edge_model_syn,
        syn_data,
        lr=mcfg.lr,
        weight_decay=mcfg.weight_decay,
        epochs=mcfg.syn_edge_epochs,
        device=device,
    )
    edge_model_syn.to(device)
    edge_model_syn.eval()
    # embeddings on real
    with torch.no_grad():
        _nl, h_real2 = node_model_syn(real_data.x.to(device), real_data.edge_index.to(device))
    edge_transfer = evaluate_edge(edge_model_syn, h_real2, real_data, real_data.edge_test_idx, device)

    # --- Reports ---
    similarity = structural_and_feature_report(real_data, syn_data)
    fraud_patterns = fraud_pattern_report(real_data, syn_data)

    def drop(real_m: Metrics, transfer_m: Metrics) -> Dict[str, Optional[float]]:
        def _d(a: float, b: float) -> Optional[float]:
            if not (math.isfinite(a) and math.isfinite(b)):
                return None
            return float(a - b)

        return {
            "roc_auc_drop": _d(real_m.roc_auc, transfer_m.roc_auc),
            "pr_auc_drop": _d(real_m.pr_auc, transfer_m.pr_auc),
            "f1_drop": _d(real_m.f1, transfer_m.f1),
        }

    result: Dict[str, Any] = {
        "device": str(device),
        "training": asdict(mcfg),
        "synthetic_meta": syn_meta,
        "node": {
            "real_train_eval": {k: v.as_dict() for k, v in node_metrics_real.items()},
            "transfer_eval_on_real_test": node_transfer.as_dict(),
            "drop_test": drop(node_metrics_real["test"], node_transfer),
        },
        "edge": {
            "real_train_eval": {k: v.as_dict() for k, v in edge_metrics_real.items()},
            "synthetic_train_eval": {k: v.as_dict() for k, v in edge_metrics_syn_train.items()},
            "transfer_eval_on_real_test": edge_transfer.as_dict(),
            "drop_test": drop(edge_metrics_real["test"], edge_transfer),
        },
        "similarity": similarity,
        "fraud_patterns": fraud_patterns,
    }
    return result

