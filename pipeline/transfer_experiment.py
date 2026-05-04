from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import random
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
from torch_geometric.data import Data

from evaluation.similarity import fraud_pattern_report, structural_and_feature_report
from generators.degree_preserving import (
    DegreePreservingGeneratorConfig,
    generate_degree_preserving_synthetic,
)
from generators.graphmaker_bridge import load_synthetic_from_torch
from models.dp_train import DPSGDConfig, train_edge_classifier_dp, train_node_classifier_dp
from models.gnn import EdgeClassifier, GraphSAGEEncoder, NodeClassifier
from models.torch_device import get_training_device
from models.train_utils import Metrics, evaluate_edge, evaluate_node, train_edge_classifier, train_node_classifier


@dataclass(frozen=True)
class ModelConfig:
    hidden_dim: int = 128
    emb_dim: int = 128
    num_layers: int = 2
    dropout: float = 0.1
    # Real graph: moderate epochs (large graph, expensive).
    node_epochs: int = 20
    edge_epochs: int = 15
    syn_node_epochs: int = 20
    syn_edge_epochs: int = 15
    lr: float = 1e-3
    weight_decay: float = 1e-4
    seed: int = 7
    warm_start_from_real: bool = False
    # Skip node/account classifier: only train encoder+edge head (jointly on edge loss).
    edges_only: bool = False
    # DP-SGD on random k-hop subgraphs (see models/dp_train.py).
    use_dp: bool = False
    dp_noise_multiplier: float = 0.7
    dp_max_grad_norm: float = 1.0
    dp_delta: float = 1e-5
    dp_batch_size: int = 512
    dp_steps_per_epoch: int = 80
    dp_num_hops: int = 2
    dp_pos_to_neg_ratio: int = 3


def _dp_cfg(mcfg: ModelConfig) -> DPSGDConfig:
    return DPSGDConfig(
        noise_multiplier=mcfg.dp_noise_multiplier,
        max_grad_norm=mcfg.dp_max_grad_norm,
        delta=mcfg.dp_delta,
        batch_size=mcfg.dp_batch_size,
        steps_per_epoch=mcfg.dp_steps_per_epoch,
        num_hops=mcfg.dp_num_hops,
        pos_to_neg_ratio=mcfg.dp_pos_to_neg_ratio,
    )


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
    synthetic_torch_path: Optional[Union[str, Path]] = None,
    min_synthetic_nodes: int = 0,
    device: Optional[torch.device] = None,
    checkpoint_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Core experiment:
    - Train on real → evaluate on real
    - Train on synthetic → evaluate on real (same real test split)

    If `synthetic_bundle` is provided as `(syn_data, syn_meta)`, it is used instead of
    the built-in degree-preserving generator (e.g. GraphMaker output after conversion).

    If `synthetic_torch_path` is set, the GraphMaker (or other) graph is loaded **after**
    the real-graph training phases, so PyTorch RNG state matches the degree-preserving
    run for the same seed (eager `torch.load` before training used to desync init).
    """
    mcfg = model_cfg or ModelConfig()
    device = device or get_training_device()
    random.seed(mcfg.seed)
    np.random.seed(mcfg.seed)
    torch.manual_seed(mcfg.seed)
    dpc = _dp_cfg(mcfg)

    def _dp_ckpt(tag: str) -> Dict[str, Any]:
        if checkpoint_dir is None:
            return {}
        return {"checkpoint_dir": checkpoint_dir, "checkpoint_tag": tag}

    # --- Real baseline ---
    node_model_real, edge_model_real = _init_models(real_data, mcfg)
    dp_accounting: Dict[str, Any] = {}
    node_metrics_real: Optional[Dict[str, Metrics]] = None

    if not mcfg.edges_only:
        if mcfg.use_dp:
            node_metrics_real, dp_accounting["node_real"] = train_node_classifier_dp(
                node_model_real,
                real_data,
                lr=mcfg.lr,
                weight_decay=mcfg.weight_decay,
                epochs=mcfg.node_epochs,
                device=device,
                dp=dpc,
                seed=mcfg.seed,
                **_dp_ckpt("node_real"),
            )
        else:
            node_metrics_real = train_node_classifier(
                node_model_real,
                real_data,
                lr=mcfg.lr,
                weight_decay=mcfg.weight_decay,
                epochs=mcfg.node_epochs,
                device=device,
            )

    if mcfg.use_dp:
        edge_metrics_real, dp_accounting["edge_real"] = train_edge_classifier_dp(
            node_model_real,
            edge_model_real,
            real_data,
            lr=mcfg.lr,
            weight_decay=mcfg.weight_decay,
            epochs=mcfg.edge_epochs,
            device=device,
            dp=dpc,
            seed=mcfg.seed,
            **_dp_ckpt("edge_real"),
        )
    else:
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
    elif synthetic_torch_path is not None:
        syn_data, syn_meta = load_synthetic_from_torch(
            Path(synthetic_torch_path), real_data, seed=mcfg.seed
        )
        if int(min_synthetic_nodes) > 0 and int(syn_data.num_nodes) < int(min_synthetic_nodes):
            raise ValueError(
                f"Synthetic graph too small: syn.num_nodes={int(syn_data.num_nodes)} < "
                f"min_synthetic_nodes={int(min_synthetic_nodes)}. "
                "Re-sample GraphMaker with a larger target node count (e.g. >= 50,000)."
            )
    else:
        syn_data, syn_meta = generate_degree_preserving_synthetic(real_data, config=gen_cfg)

    # --- Train on synthetic, evaluate on real ---
    node_model_syn, edge_model_syn = _init_models(syn_data, mcfg)
    if mcfg.warm_start_from_real:
        node_model_syn.load_state_dict(node_model_real.state_dict())
        edge_model_syn.load_state_dict(edge_model_real.state_dict())

    if not mcfg.edges_only:
        if mcfg.use_dp:
            _, dp_accounting["node_syn"] = train_node_classifier_dp(
                node_model_syn,
                syn_data,
                lr=mcfg.lr,
                weight_decay=mcfg.weight_decay,
                epochs=mcfg.syn_node_epochs,
                device=device,
                dp=dpc,
                seed=mcfg.seed,
                **_dp_ckpt("node_syn"),
            )
        else:
            _ = train_node_classifier(
                node_model_syn,
                syn_data,
                lr=mcfg.lr,
                weight_decay=mcfg.weight_decay,
                epochs=mcfg.syn_node_epochs,
                device=device,
            )

    if not mcfg.edges_only:
        sel = real_data.node_test_mask.to(device)
        node_transfer = evaluate_node(node_model_syn, real_data, sel, device)[0]
    else:
        node_transfer = None

    # Edge transfer: train edge head on synthetic using synthetic embeddings, then evaluate on real
    if mcfg.use_dp:
        edge_metrics_syn_train, dp_accounting["edge_syn"] = train_edge_classifier_dp(
            node_model_syn,
            edge_model_syn,
            syn_data,
            lr=mcfg.lr,
            weight_decay=mcfg.weight_decay,
            epochs=mcfg.syn_edge_epochs,
            device=device,
            dp=dpc,
            seed=mcfg.seed,
            **_dp_ckpt("edge_syn"),
        )
    else:
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

    if mcfg.edges_only:
        node_block: Dict[str, Any] = {
            "skipped": True,
            "reason": (
                "edges_only: GraphSAGE encoder and edge head are trained jointly on edge loss only; "
                "account-level node classifier was not trained or evaluated."
            ),
        }
    else:
        assert node_metrics_real is not None and node_transfer is not None
        node_block = {
            "real_train_eval": {k: v.as_dict() for k, v in node_metrics_real.items()},
            "transfer_eval_on_real_test": node_transfer.as_dict(),
            "drop_test": drop(node_metrics_real["test"], node_transfer),
        }

    result: Dict[str, Any] = {
        "device": str(device),
        "training": asdict(mcfg),
        "dp_accounting": dp_accounting if mcfg.use_dp else {},
        "checkpoint_dir": str(checkpoint_dir.resolve()) if checkpoint_dir else None,
        # Legacy key (same as checkpoint_dir); kept for older result consumers.
        "edge_checkpoint_dir": str(checkpoint_dir.resolve()) if checkpoint_dir else None,
        "synthetic_meta": syn_meta,
        "node": node_block,
        "edge": {
            "real_train_eval": {k: v.as_dict() for k, v in edge_metrics_real.items()},
            "synthetic_train_eval": {k: v.as_dict() for k, v in edge_metrics_syn_train.items()},
            "transfer_eval_on_real_test": edge_transfer.as_dict(),
            "drop_test": drop(edge_metrics_real["test"], edge_transfer),
        },
        "similarity": similarity,
        "fraud_patterns": fraud_patterns,
    }
    if mcfg.use_dp:
        if mcfg.edges_only:
            result["dp_composition_note"] = (
                "Each edge training phase (real edge, synthetic edge) uses its own RDPAccountant restart. "
                "Reported epsilon_rdp values are not automatically composed into a single joint (ε, δ) "
                "guarantee across phases."
            )
        else:
            result["dp_composition_note"] = (
                "Each training phase (real node, real edge, synthetic node, synthetic edge) uses its own "
                "RDPAccountant restart. Reported epsilon_rdp values are not automatically composed into a "
                "single joint (ε, δ) guarantee across phases."
            )
    return result

