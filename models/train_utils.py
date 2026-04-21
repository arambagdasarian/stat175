from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score


@dataclass
class Metrics:
    roc_auc: float
    pr_auc: float
    f1: float

    def as_dict(self) -> Dict[str, float]:
        return {"roc_auc": float(self.roc_auc), "pr_auc": float(self.pr_auc), "f1": float(self.f1)}


def _safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    # If only one class present, roc_auc_score errors. Return nan in that case.
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def compute_binary_metrics(
    y_true: np.ndarray, y_logits: np.ndarray, threshold: float = 0.5
) -> Metrics:
    y_logits = np.clip(np.asarray(y_logits, dtype=np.float64), -40.0, 40.0)
    y_prob = 1.0 / (1.0 + np.exp(-y_logits))
    roc = _safe_auc(y_true, y_prob)
    pr = float(average_precision_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else float("nan")
    y_pred = (y_prob >= threshold).astype(int)
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    return Metrics(roc_auc=roc, pr_auc=pr, f1=f1)


def make_pos_weight(y: torch.Tensor, *, max_ratio: float = 50_000.0) -> torch.Tensor:
    # BCEWithLogits pos_weight = (neg/pos). Clip extreme ratios to avoid destabilizing Adam on tiny folds.
    y = y.detach().flatten()
    pos = (y == 1).sum().item()
    neg = (y == 0).sum().item()
    if pos == 0:
        return torch.tensor(1.0, device=y.device)
    ratio = float(neg / pos)
    ratio = min(ratio, float(max_ratio))
    return torch.tensor(ratio, device=y.device)


@torch.no_grad()
def evaluate_node(
    model,
    data,
    mask: torch.Tensor,
    device: torch.device,
) -> Tuple[Metrics, torch.Tensor]:
    model.eval()
    logits, h = model(data.x.to(device), data.edge_index.to(device))
    y = data.y_node.to(device)
    sel = mask.to(device)
    m = compute_binary_metrics(
        y_true=y[sel].cpu().numpy(),
        y_logits=logits[sel].cpu().numpy(),
    )
    return m, h


@torch.no_grad()
def evaluate_edge(
    edge_model,
    h: torch.Tensor,
    data,
    edge_idx: torch.Tensor,
    device: torch.device,
) -> Metrics:
    edge_model.eval()
    if edge_idx.numel() == 0:
        return Metrics(roc_auc=float("nan"), pr_auc=float("nan"), f1=0.0)
    logits = edge_model(
        h=h,
        edge_index=data.edge_index.to(device),
        edge_attr=data.edge_attr.to(device),
        edge_idx=edge_idx.to(device),
    )
    y = data.y_edge.to(device)[edge_idx.to(device)]
    m = compute_binary_metrics(y_true=y.cpu().numpy(), y_logits=logits.cpu().numpy())
    return m


def train_node_classifier(
    model,
    data,
    *,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    epochs: int = 10,
    device: Optional[torch.device] = None,
) -> Dict[str, Metrics]:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_mask = data.node_train_mask.to(device)
    val_mask = data.node_val_mask.to(device)

    y = data.y_node.to(device).float()
    pos_weight = make_pos_weight(y[train_mask]).to(device)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_val = -float("inf")
    best_state = None

    for _ in range(epochs):
        model.train()
        opt.zero_grad(set_to_none=True)
        logits, _h = model(data.x.to(device), data.edge_index.to(device))
        loss = loss_fn(logits[train_mask], y[train_mask])
        loss.backward()
        opt.step()

        val_metrics, _ = evaluate_node(model, data, val_mask, device)
        score = val_metrics.pr_auc  # prioritize PR-AUC under imbalance
        if np.isfinite(score) and score > best_val:
            best_val = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    tr, _ = evaluate_node(model, data, train_mask, device)
    va, _ = evaluate_node(model, data, val_mask, device)
    te, _ = evaluate_node(model, data, data.node_test_mask, device)
    return {"train": tr, "val": va, "test": te}


def train_edge_classifier(
    encoder_model,
    edge_model,
    data,
    *,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    epochs: int = 5,
    device: Optional[torch.device] = None,
) -> Dict[str, Metrics]:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder_model.to(device)
    edge_model.to(device)

    opt = torch.optim.Adam(
        list(encoder_model.parameters()) + list(edge_model.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )

    train_idx = data.edge_train_idx.to(device)
    val_idx = data.edge_val_idx.to(device)

    y_train = data.y_edge.to(device)[train_idx].float()
    pos_weight = make_pos_weight(y_train).to(device)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_val = -float("inf")
    best_state = None

    for _ in range(epochs):
        encoder_model.train()
        edge_model.train()
        opt.zero_grad(set_to_none=True)

        # get node embeddings
        _node_logits, h = encoder_model(data.x.to(device), data.edge_index.to(device))
        # edge logits on train edges
        logits_e = edge_model(
            h=h,
            edge_index=data.edge_index.to(device),
            edge_attr=data.edge_attr.to(device),
            edge_idx=train_idx,
        )
        loss = loss_fn(logits_e, y_train)
        loss.backward()
        opt.step()

        # validate
        encoder_model.eval()
        edge_model.eval()
        with torch.no_grad():
            _nl, h_val = encoder_model(data.x.to(device), data.edge_index.to(device))
            val_metrics = evaluate_edge(edge_model, h_val, data, val_idx, device)
        score = val_metrics.pr_auc
        if np.isfinite(score) and score > best_val:
            best_val = score
            best_state = {
                "encoder": {k: v.detach().cpu().clone() for k, v in encoder_model.state_dict().items()},
                "edge": {k: v.detach().cpu().clone() for k, v in edge_model.state_dict().items()},
            }

    if best_state is not None:
        encoder_model.load_state_dict(best_state["encoder"])
        edge_model.load_state_dict(best_state["edge"])

    # final metrics
    encoder_model.eval()
    edge_model.eval()
    with torch.no_grad():
        _nl, h = encoder_model(data.x.to(device), data.edge_index.to(device))
    tr = evaluate_edge(edge_model, h, data, data.edge_train_idx, device)
    va = evaluate_edge(edge_model, h, data, data.edge_val_idx, device)
    te = evaluate_edge(edge_model, h, data, data.edge_test_idx, device)
    return {"train": tr, "val": va, "test": te}

