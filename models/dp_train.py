"""
DP-SGD style training for GraphSAGE + edge MLP on one large graph.

Full-graph transductive steps mix the entire graph, so per-example DP gradients are
not meaningful there. We optimize on random **k-hop subgraphs** around sampled
training nodes or around endpoints of sampled training edges: each step applies
global gradient clipping and Gaussian noise (Abadi et al. DP-SGD).

Privacy accounting: optional `opacus.accountants.RDPAccountant`. If `opacus` is not
installed, `epsilon_rdp` is omitted (noise/clip still apply).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph

from models.torch_device import get_training_device
from models.train_utils import Metrics, evaluate_edge, evaluate_node, make_pos_weight


def _try_rdp_accountant():
    try:
        from opacus.accountants import RDPAccountant

        return RDPAccountant
    except Exception:
        return None


@dataclass(frozen=True)
class DPSGDConfig:
    noise_multiplier: float = 1.0
    max_grad_norm: float = 1.0
    delta: float = 1e-5
    batch_size: int = 512
    steps_per_epoch: int = 40
    num_hops: int = 2
    # Balanced minibatches: pos:neg ratio for labeled items when both exist.
    # Example: 1 means ~1:1, 3 means ~3:1 positives.
    pos_to_neg_ratio: int = 1


def _take_counts(batch_size: int, pos_to_neg_ratio: int) -> Tuple[int, int]:
    r = max(1, int(pos_to_neg_ratio))
    n_pos_take = max(1, int(round(batch_size * (r / (r + 1)))))
    n_pos_take = min(n_pos_take, batch_size - 1)  # leave room for at least 1 negative if possible
    n_neg_take = max(1, batch_size - n_pos_take)
    return n_pos_take, n_neg_take


def _balanced_choice_from_two_pools(
    pos_pool: torch.Tensor,
    neg_pool: torch.Tensor,
    *,
    batch_size: int,
    pos_to_neg_ratio: int,
    rng: torch.Generator,
) -> torch.Tensor:
    """
    Return a shuffled batch of indices drawn from pos_pool and neg_pool.
    Pools are tensors of global indices on CPU.
    """
    pos_pool = pos_pool.view(-1)
    neg_pool = neg_pool.view(-1)
    n_pos = int(pos_pool.numel())
    n_neg = int(neg_pool.numel())
    if n_pos == 0 and n_neg == 0:
        return torch.tensor([], dtype=torch.long, device="cpu")
    if n_pos == 0:
        take = min(batch_size, n_neg)
        perm = torch.randperm(n_neg, generator=rng, device="cpu")[:take]
        return neg_pool[perm]
    if n_neg == 0:
        take = min(batch_size, n_pos)
        perm = torch.randperm(n_pos, generator=rng, device="cpu")[:take]
        return pos_pool[perm]

    n_pos_take, n_neg_take = _take_counts(batch_size, pos_to_neg_ratio)
    n_pos_take = min(n_pos_take, n_pos)
    n_neg_take = min(n_neg_take, n_neg)
    # If one side is too small, top up with the other.
    total = n_pos_take + n_neg_take
    if total < batch_size:
        rem = batch_size - total
        # Prefer topping up negatives if possible (avoid all-positive batches).
        add_neg = min(rem, n_neg - n_neg_take)
        n_neg_take += add_neg
        rem -= add_neg
        add_pos = min(rem, n_pos - n_pos_take)
        n_pos_take += add_pos

    p_ix = torch.randperm(n_pos, generator=rng, device="cpu")[:n_pos_take]
    n_ix = torch.randperm(n_neg, generator=rng, device="cpu")[:n_neg_take]
    mix = torch.cat([pos_pool[p_ix], neg_pool[n_ix]], dim=0)
    shuf = torch.randperm(mix.numel(), generator=rng, device="cpu")
    return mix[shuf]


def _balanced_edge_train_local_indices(
    train_idx: torch.Tensor,
    y_edge: torch.Tensor,
    batch_size: int,
    rng: torch.Generator,
    *,
    pos_to_neg_ratio: int = 1,
) -> torch.Tensor:
    """Positions into `train_idx` (~half pos / half neg edges per step) for DP edge training."""
    n_train = int(train_idx.numel())
    if n_train == 0:
        return torch.tensor([], dtype=torch.long, device="cpu")
    y_tr = y_edge[train_idx].float()
    pos_l = (y_tr > 0.5).nonzero(as_tuple=False).view(-1)
    neg_l = (y_tr <= 0.5).nonzero(as_tuple=False).view(-1)
    # These are "local positions" (into train_idx), so treat them as pools.
    return _balanced_choice_from_two_pools(
        pos_l.to("cpu"),
        neg_l.to("cpu"),
        batch_size=batch_size,
        pos_to_neg_ratio=pos_to_neg_ratio,
        rng=rng,
    )


def _save_node_checkpoint(
    path: Union[str, Path],
    model: nn.Module,
    *,
    epoch: int,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "epoch": int(epoch),
    }
    if extra:
        payload["extra"] = extra
    torch.save(payload, p)


def _save_edge_checkpoint(
    path: Union[str, Path],
    encoder_model: nn.Module,
    edge_model: nn.Module,
    *,
    epoch: int,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {
        "encoder_state_dict": encoder_model.state_dict(),
        "edge_state_dict": edge_model.state_dict(),
        "epoch": int(epoch),
    }
    if extra:
        payload["extra"] = extra
    torch.save(payload, p)


def _add_dp_noise_(params, noise_multiplier: float, max_grad_norm: float) -> None:
    with torch.no_grad():
        for p in params:
            if p.grad is None:
                continue
            p.grad.add_(torch.randn_like(p.grad) * noise_multiplier * max_grad_norm)


def train_node_classifier_dp(
    model: nn.Module,
    data: Data,
    *,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    epochs: int = 8,
    device: Optional[torch.device] = None,
    dp: DPSGDConfig = DPSGDConfig(),
    seed: int = 7,
    checkpoint_dir: Optional[Union[str, Path]] = None,
    checkpoint_tag: str = "node",
) -> Tuple[Dict[str, Metrics], Dict[str, Any]]:
    device = device or get_training_device()
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_mask = data.node_train_mask.to(device)
    val_mask = data.node_val_mask.to(device)
    y_full = data.y_node.to(device).float()
    pos_w = make_pos_weight(y_full[train_mask]).to(device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_w)

    # CPU generator: portable across CUDA/MPS (MPS does not always support torch.Generator on-device).
    rng = torch.Generator(device="cpu")
    rng.manual_seed(seed)

    # Cache CPU pools for faster balanced sampling.
    train_idx_cpu = train_mask.detach().to("cpu").nonzero(as_tuple=False).view(-1).to(torch.long)
    y_train_cpu = y_full.detach().to("cpu")[train_idx_cpu]
    pos_pool = train_idx_cpu[(y_train_cpu > 0.5).nonzero(as_tuple=False).view(-1)]
    neg_pool = train_idx_cpu[(y_train_cpu <= 0.5).nonzero(as_tuple=False).view(-1)]

    n_train = int(train_idx_cpu.numel())
    AccountantCls = _try_rdp_accountant()
    accountant = AccountantCls() if AccountantCls is not None else None
    sample_rate = min(1.0, float(dp.batch_size) / max(1, n_train))

    edge_index = data.edge_index.to(device)
    x_all = data.x.to(device)

    best_val = -float("inf")
    best_state = None
    total_steps = 0
    ckpt_root = Path(checkpoint_dir) if checkpoint_dir is not None else None

    for epoch in range(epochs):
        for _s in range(dp.steps_per_epoch):
            seeds_cpu = _balanced_choice_from_two_pools(
                pos_pool,
                neg_pool,
                batch_size=dp.batch_size,
                pos_to_neg_ratio=dp.pos_to_neg_ratio,
                rng=rng,
            )
            if seeds_cpu.numel() == 0:
                break
            seeds = seeds_cpu.to(device)
            subset, edge_sub, inv, _ = k_hop_subgraph(
                seeds,
                dp.num_hops,
                edge_index,
                num_nodes=int(data.num_nodes),
                relabel_nodes=True,
            )
            x_sub = x_all[subset]
            model.train()
            opt.zero_grad(set_to_none=True)
            logits, _ = model(x_sub, edge_sub)
            loss = loss_fn(logits[inv], y_full[seeds])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), dp.max_grad_norm)
            _add_dp_noise_(model.parameters(), dp.noise_multiplier, dp.max_grad_norm)
            opt.step()
            total_steps += 1
            if accountant is not None:
                accountant.step(noise_multiplier=dp.noise_multiplier, sample_rate=sample_rate)

        model.eval()
        with torch.no_grad():
            val_metrics, _ = evaluate_node(model, data, val_mask, device)
        score = val_metrics.pr_auc
        if np.isfinite(score) and score > best_val:
            best_val = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if ckpt_root is not None:
                _save_node_checkpoint(
                    ckpt_root / f"{checkpoint_tag}_best.pt",
                    model,
                    epoch=epoch + 1,
                    extra={"val_pr_auc": float(score), "total_steps": total_steps},
                )

        if ckpt_root is not None:
            _save_node_checkpoint(
                ckpt_root / f"{checkpoint_tag}_epoch_{epoch + 1:04d}.pt",
                model,
                epoch=epoch + 1,
                extra={"val_pr_auc": float(val_metrics.pr_auc), "total_steps": total_steps},
            )

    if best_state is not None:
        model.load_state_dict(best_state)

    tr, _ = evaluate_node(model, data, train_mask, device)
    va, _ = evaluate_node(model, data, val_mask, device)
    te, _ = evaluate_node(model, data, data.node_test_mask, device)

    eps: Optional[float] = None
    if accountant is not None:
        try:
            eps = float(accountant.get_epsilon(dp.delta))
        except Exception:
            eps = None

    meta = {
        "dp_sgd": True,
        "target": "node",
        "checkpoints_written": str(ckpt_root.resolve()) if ckpt_root is not None else None,
        "subgraph_k_hop": dp.num_hops,
        "noise_multiplier": dp.noise_multiplier,
        "max_grad_norm": dp.max_grad_norm,
        "delta": dp.delta,
        "batch_size": dp.batch_size,
        "steps_per_epoch": dp.steps_per_epoch,
        "epochs": epochs,
        "total_optimizer_steps": total_steps,
        "sample_rate_approx": sample_rate,
        "epsilon_rdp": eps,
        "accountant": "opacus.RDPAccountant" if accountant is not None else None,
        "pos_to_neg_ratio": int(dp.pos_to_neg_ratio),
        "balanced_minibatches": True,
        "note": (
            "DP-SGD on random k-hop subgraphs; RDP epsilon is a standard DP-SGD accounting "
            "heuristic for this stochastic subgraph scheme (not a formal graph DP guarantee). "
            "Training seeds oversample positives (pos:neg ratio configurable) when both classes exist."
        ),
    }
    return {"train": tr, "val": va, "test": te}, meta


def train_edge_classifier_dp(
    encoder_model: nn.Module,
    edge_model: nn.Module,
    data: Data,
    *,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    epochs: int = 6,
    device: Optional[torch.device] = None,
    dp: DPSGDConfig = DPSGDConfig(),
    seed: int = 7,
    checkpoint_dir: Optional[Union[str, Path]] = None,
    checkpoint_tag: str = "edge",
) -> Tuple[Dict[str, Metrics], Dict[str, Any]]:
    device = device or get_training_device()
    encoder_model.to(device)
    edge_model.to(device)
    params = list(encoder_model.parameters()) + list(edge_model.parameters())
    opt = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)

    train_idx = data.edge_train_idx.to(device)
    val_idx = data.edge_val_idx.to(device)
    y_edge = data.y_edge.to(device).float()
    pos_w = make_pos_weight(y_edge[train_idx]).to(device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_w)

    rng = torch.Generator(device="cpu")
    rng.manual_seed(seed + 1)

    n_train = int(train_idx.numel())
    AccountantCls = _try_rdp_accountant()
    accountant = AccountantCls() if AccountantCls is not None else None
    sample_rate = min(1.0, float(dp.batch_size) / max(1, n_train))

    edge_index = data.edge_index.to(device)
    x_all = data.x.to(device)
    edge_attr_all = data.edge_attr.to(device)

    # Cache CPU pools of local positions for faster balanced sampling.
    y_tr_cpu = y_edge.detach().to("cpu")[train_idx.detach().to("cpu")]
    pos_local_cpu = (y_tr_cpu > 0.5).nonzero(as_tuple=False).view(-1).to(torch.long)
    neg_local_cpu = (y_tr_cpu <= 0.5).nonzero(as_tuple=False).view(-1).to(torch.long)

    best_val = -float("inf")
    best_state = None
    total_steps = 0
    ckpt_root = Path(checkpoint_dir) if checkpoint_dir is not None else None

    for epoch in range(epochs):
        for _s in range(dp.steps_per_epoch):
            local_ix = _balanced_choice_from_two_pools(
                pos_local_cpu,
                neg_local_cpu,
                batch_size=dp.batch_size,
                pos_to_neg_ratio=dp.pos_to_neg_ratio,
                rng=rng,
            )
            if local_ix.numel() == 0:
                break
            e_batch = train_idx[local_ix.to(device)]
            src = edge_index[0, e_batch]
            dst = edge_index[1, e_batch]
            seeds = torch.cat([src, dst]).unique()

            subset, edge_sub, _, edge_mask = k_hop_subgraph(
                seeds,
                dp.num_hops,
                edge_index,
                num_nodes=int(data.num_nodes),
                relabel_nodes=True,
            )
            subset = subset.to(device)
            edge_sub = edge_sub.to(device)
            edge_mask = edge_mask.to(device)
            x_sub = x_all[subset]
            attr_sub = edge_attr_all[edge_mask]

            inv_map = torch.full((int(data.num_nodes),), -1, dtype=torch.long, device=device)
            inv_map[subset] = torch.arange(subset.numel(), device=device, dtype=torch.long)

            loc_map: Dict[Tuple[int, int], int] = {}
            for j in range(edge_sub.size(1)):
                loc_map[(int(edge_sub[0, j].item()), int(edge_sub[1, j].item()))] = j

            loc_cols: list[int] = []
            y_list: list[float] = []
            for i in range(e_batch.numel()):
                gi = int(e_batch[i].item())
                u = int(edge_index[0, gi].item())
                v = int(edge_index[1, gi].item())
                lu = int(inv_map[u].item())
                lv = int(inv_map[v].item())
                if lu < 0 or lv < 0:
                    continue
                j = loc_map.get((lu, lv))
                if j is None:
                    continue
                loc_cols.append(j)
                y_list.append(float(y_edge[gi].item()))

            if not loc_cols:
                continue

            edge_idx_local = torch.tensor(loc_cols, device=device, dtype=torch.long)
            y_b = torch.tensor(y_list, device=device, dtype=torch.float32)

            encoder_model.train()
            edge_model.train()
            opt.zero_grad(set_to_none=True)
            _nl, h = encoder_model(x_sub, edge_sub)
            logits_e = edge_model(
                h=h,
                edge_index=edge_sub,
                edge_attr=attr_sub,
                edge_idx=edge_idx_local,
            )
            loss = loss_fn(logits_e, y_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, dp.max_grad_norm)
            _add_dp_noise_(params, dp.noise_multiplier, dp.max_grad_norm)
            opt.step()
            total_steps += 1
            if accountant is not None:
                accountant.step(noise_multiplier=dp.noise_multiplier, sample_rate=sample_rate)

        encoder_model.eval()
        edge_model.eval()
        with torch.no_grad():
            _nl, h_val = encoder_model(x_all, edge_index)
            val_metrics = evaluate_edge(edge_model, h_val, data, val_idx, device)
        score = val_metrics.pr_auc
        if np.isfinite(score) and score > best_val:
            best_val = score
            best_state = {
                "encoder": {k: v.detach().cpu().clone() for k, v in encoder_model.state_dict().items()},
                "edge": {k: v.detach().cpu().clone() for k, v in edge_model.state_dict().items()},
            }
            if ckpt_root is not None:
                _save_edge_checkpoint(
                    ckpt_root / f"{checkpoint_tag}_best.pt",
                    encoder_model,
                    edge_model,
                    epoch=epoch + 1,
                    extra={"val_pr_auc": float(score), "total_steps": total_steps},
                )

        if ckpt_root is not None:
            _save_edge_checkpoint(
                ckpt_root / f"{checkpoint_tag}_epoch_{epoch + 1:04d}.pt",
                encoder_model,
                edge_model,
                epoch=epoch + 1,
                extra={"val_pr_auc": float(val_metrics.pr_auc), "total_steps": total_steps},
            )

    if best_state is not None:
        encoder_model.load_state_dict(best_state["encoder"])
        edge_model.load_state_dict(best_state["edge"])

    encoder_model.eval()
    edge_model.eval()
    with torch.no_grad():
        _nl, h = encoder_model(x_all, edge_index)
    tr = evaluate_edge(edge_model, h, data, data.edge_train_idx, device)
    va = evaluate_edge(edge_model, h, data, data.edge_val_idx, device)
    te = evaluate_edge(edge_model, h, data, data.edge_test_idx, device)

    eps: Optional[float] = None
    if accountant is not None:
        try:
            eps = float(accountant.get_epsilon(dp.delta))
        except Exception:
            eps = None

    meta = {
        "dp_sgd": True,
        "target": "edge",
        "checkpoints_written": str(ckpt_root.resolve()) if ckpt_root is not None else None,
        "subgraph_k_hop": dp.num_hops,
        "noise_multiplier": dp.noise_multiplier,
        "max_grad_norm": dp.max_grad_norm,
        "delta": dp.delta,
        "batch_size": dp.batch_size,
        "steps_per_epoch": dp.steps_per_epoch,
        "epochs": epochs,
        "total_optimizer_steps": total_steps,
        "sample_rate_approx": sample_rate,
        "epsilon_rdp": eps,
        "accountant": "opacus.RDPAccountant" if accountant is not None else None,
        "pos_to_neg_ratio": int(dp.pos_to_neg_ratio),
        "balanced_minibatches": True,
        "note": (
            "Encoder + edge head trained with DP-SGD on subgraphs around sampled train edges; "
            "full-graph eval uses trained weights (not a per-edge DP guarantee on inference). "
            "Each step oversamples positive training edges (pos:neg ratio configurable) when both classes exist."
        ),
    }
    return {"train": tr, "val": va, "test": te}, meta
