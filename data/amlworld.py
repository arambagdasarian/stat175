from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch_geometric.data import Data


def _make_ohe() -> OneHotEncoder:
    # sklearn compatibility: `sparse_output` is newer; older versions use `sparse`.
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


@dataclass(frozen=True)
class AMLWorldPaths:
    accounts_csv: Path
    transactions_csv: Path


def _normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.strip().str.lower()
    return df


def resolve_hi_small_paths(data_dir: Path) -> AMLWorldPaths:
    accounts = data_dir / "HI-Small_accounts.csv"
    trans = data_dir / "HI-Small_Trans.csv"
    if not accounts.exists():
        raise FileNotFoundError(f"Missing accounts file: {accounts}")
    if not trans.exists():
        raise FileNotFoundError(f"Missing transactions file: {trans}")
    return AMLWorldPaths(accounts_csv=accounts, transactions_csv=trans)


def _extract_entity_type(series: pd.Series) -> pd.Series:
    # e.g. "Sole Proprietorship #50438" -> "Sole Proprietorship"
    s = series.astype(str)
    return s.str.replace(r"\s+#\d+$", "", regex=True).str.strip()


def _parse_timestamp(ts: pd.Series) -> pd.Series:
    # AMLWorld HI-Small format observed in notebook: "%Y/%m/%d %H:%M"
    return pd.to_datetime(ts, format="%Y/%m/%d %H:%M", errors="coerce")


def _allocate_positives_train_val_test(
    n_pos: int,
    sizes: Tuple[int, int, int],
) -> Tuple[int, int, int]:
    """
    Spread rare positives so **test** (and val) are not empty when enough positives exist.

    Pure largest-remainder can assign [2,1,0] for three positives and 60/20/20 sizes, leaving
    **no** positives in test. Here we reserve a minimal mass so each phase of training has
    signal when possible, then distribute the remainder by largest remainder on split sizes.

    - ``n_pos == 1``: put the single positive in **test** (evaluation-focused; train has no positives).
    - ``n_pos == 2``: **train** and **val** get one each (validation-driven early stopping).
    - ``n_pos >= 3``: **train**, **val**, and **test** each get at least one; any **surplus**
      positives are assigned in **test → val → train** round-robin so test is not starved.
    """
    n_tr, n_va, n_te = sizes
    if n_pos <= 0:
        return 0, 0, 0
    if n_pos == 1:
        return 0, 0, min(1, n_te)
    if n_pos == 2:
        return min(1, n_tr), min(1, n_va), 0
    p_tr, p_va, p_te = 1, 1, 1
    rem = n_pos - 3
    if rem <= 0:
        return p_tr, p_va, p_te
    # Spread any surplus so **test** gets the next positive first (then val, then train).
    # Pure largest-remainder on 60/20/20 tends to pile extras on train and can leave test thin.
    extra_tr = extra_va = extra_te = 0
    prio = (2, 1, 0)  # indices into [p_tr, p_va, p_te] — test, val, train
    buckets = [0, 0, 0]
    for k in range(rem):
        buckets[prio[k % 3]] += 1
    extra_tr, extra_va, extra_te = buckets[0], buckets[1], buckets[2]
    return p_tr + extra_tr, p_va + extra_va, p_te + extra_te


def _split_label_counts(y: np.ndarray, train_idx: np.ndarray, val_idx: np.ndarray, test_idx: np.ndarray) -> List[Dict[str, int]]:
    y = np.asarray(y).reshape(-1)
    out: List[Dict[str, int]] = []
    for name, ix in (("train", train_idx), ("val", val_idx), ("test", test_idx)):
        yy = y[ix]
        out.append({"split": name, "n": int(ix.size), "positives": int(np.sum(yy == 1)), "negatives": int(np.sum(yy == 0))})
    return out


def _build_splits_binary(
    y: np.ndarray,
    train_size: float,
    val_size: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Train / val / test index arrays for binary labels.

    Stratified sklearn splits can assign **zero** positives to val or test on rare
    labels (only a handful of positives in the whole graph). That makes ROC on the
    test fold unstable (e.g. a single positive). Instead we:

    1. Take the **same fold sizes** as sklearn would for an unstratified split with
       the same `(train_size, val_size, seed)` (so overall 60/20/20 counts match the
       previous random baseline).
    2. Allocate **positive** indices with ``_allocate_positives_train_val_test`` so
       test (and val) receive positives when the global count allows (see that helper).
    3. Fill remaining slots with shuffled negatives.

    When `y` has fewer than two classes, we fall back to unstratified splits only.
    """
    y = np.asarray(y).reshape(-1)
    n = int(y.shape[0])
    rng = np.random.default_rng(int(seed))

    idx_all = np.arange(n, dtype=np.int64)
    if n == 0:
        return idx_all, idx_all, idx_all
    if n == 1:
        # sklearn cannot 60/20/20 a single sample; keep the only index in train.
        return idx_all, np.array([], dtype=np.int64), np.array([], dtype=np.int64)

    if len(np.unique(y)) < 2:
        tr, tmp = train_test_split(idx_all, train_size=train_size, random_state=seed, stratify=None)
        val_rel = val_size / (1.0 - train_size)
        va, te = train_test_split(tmp, train_size=val_rel, random_state=seed, stratify=None)
        return np.asarray(tr, dtype=np.int64), np.asarray(va, dtype=np.int64), np.asarray(te, dtype=np.int64)

    tr_ref, tmp_ref = train_test_split(idx_all, train_size=train_size, random_state=seed, stratify=None)
    n_tr, n_tmp = len(tr_ref), len(tmp_ref)
    val_rel = val_size / (1.0 - train_size)
    va_ref, te_ref = train_test_split(tmp_ref, train_size=val_rel, random_state=seed, stratify=None)
    n_va, n_te = len(va_ref), len(te_ref)

    pos = np.flatnonzero(y == 1).astype(np.int64, copy=False)
    neg = np.flatnonzero(y == 0).astype(np.int64, copy=False)
    rng.shuffle(pos)
    rng.shuffle(neg)

    n_pos = int(pos.size)
    n_neg = int(neg.size)
    p_tr, p_va, p_te = _allocate_positives_train_val_test(n_pos, (n_tr, n_va, n_te))

    need_neg_tr = n_tr - p_tr
    need_neg_va = n_va - p_va
    need_neg_te = n_te - p_te
    if need_neg_tr < 0 or need_neg_va < 0 or need_neg_te < 0 or (need_neg_tr + need_neg_va + need_neg_te) > n_neg:
        # Should not occur when n_pos + n_neg == n and allocation is consistent; fall back.
        tr, tmp = train_test_split(idx_all, train_size=train_size, random_state=seed, stratify=None)
        va, te = train_test_split(tmp, train_size=val_rel, random_state=seed, stratify=None)
        return np.asarray(tr, dtype=np.int64), np.asarray(va, dtype=np.int64), np.asarray(te, dtype=np.int64)

    train_idx = np.sort(np.concatenate([pos[:p_tr], neg[:need_neg_tr]]))
    val_idx = np.sort(
        np.concatenate([pos[p_tr : p_tr + p_va], neg[need_neg_tr : need_neg_tr + need_neg_va]])
    )
    test_idx = np.sort(
        np.concatenate(
            [
                pos[p_tr + p_va :],
                neg[need_neg_tr + need_neg_va : need_neg_tr + need_neg_va + need_neg_te],
            ]
        )
    )
    return train_idx, val_idx, test_idx


def load_amlworld_hi_small_pyg(
    data_dir: Path,
    *,
    max_transactions: Optional[int] = None,
    seed: int = 7,
    train_size: float = 0.6,
    val_size: float = 0.2,
    add_degree_features: bool = False,
) -> Tuple[Data, Dict[str, Any]]:
    """
    Build a PyTorch Geometric `Data` object from AMLWorld HI-Small.

    Node labels (binary):
    - `y_node`: derived SAR-account flag (any endpoint of laundering transaction)

    Edge labels (binary):
    - `y_edge`: `is laundering` (transaction fraud label)
    """
    paths = resolve_hi_small_paths(data_dir)
    trans = _normalize_cols(pd.read_csv(paths.transactions_csv))
    accts = _normalize_cols(pd.read_csv(paths.accounts_csv))

    # Columns confirmed in the user's notebook
    time_col = "timestamp"
    orig_col = "account"
    bene_col = "account.1"
    amt_col = "amount paid"
    label_col = "is laundering"
    payfmt_col = "payment format"
    bank_from_col = "from bank"
    bank_to_col = "to bank"

    if max_transactions is not None:
        trans = trans.iloc[:max_transactions].copy()

    trans[label_col] = trans[label_col].astype(int)
    trans[time_col] = _parse_timestamp(trans[time_col])

    # Node universe: from accounts table + any accounts referenced in transactions slice
    acct_id_col = "account number"
    acct_ids_from_accts = accts[acct_id_col].astype(str)
    acct_ids_from_trans = pd.concat([trans[orig_col].astype(str), trans[bene_col].astype(str)])
    all_acct_ids = pd.Index(acct_ids_from_accts).append(pd.Index(acct_ids_from_trans)).unique()
    node_id_to_idx = {aid: i for i, aid in enumerate(all_acct_ids)}

    # Derive SAR-account node label using laundering transactions in the selected slice
    launder = trans.loc[trans[label_col] == 1]
    sar_accts = set(launder[orig_col].astype(str)) | set(launder[bene_col].astype(str))
    y_node = np.array([1 if aid in sar_accts else 0 for aid in all_acct_ids], dtype=np.int64)

    # Node features
    # We use: bank_id (standardized) + entity_type (one-hot).
    # Note: accounts table may contain duplicates by account number (different banks).
    accts_dedup = accts.drop_duplicates(subset=[acct_id_col], keep="first").copy()
    accts_dedup["entity_type"] = _extract_entity_type(accts_dedup["entity name"])

    bank_id_map = dict(zip(accts_dedup[acct_id_col].astype(str), accts_dedup["bank id"]))
    ent_type_map = dict(zip(accts_dedup[acct_id_col].astype(str), accts_dedup["entity_type"]))

    bank_id = np.array([bank_id_map.get(aid, np.nan) for aid in all_acct_ids], dtype=np.float64)
    ent_type = np.array([ent_type_map.get(aid, "Unknown") for aid in all_acct_ids], dtype=object)

    # Impute missing bank id with median of observed
    med = np.nanmedian(bank_id) if np.isfinite(bank_id).any() else 0.0
    bank_id = np.where(np.isfinite(bank_id), bank_id, med).reshape(-1, 1)
    bank_id_scaled = StandardScaler().fit_transform(bank_id).astype(np.float32)

    ohe = _make_ohe()
    ent_ohe = ohe.fit_transform(ent_type.reshape(-1, 1)).astype(np.float32)

    x = np.concatenate([bank_id_scaled, ent_ohe], axis=1)

    # Edge index
    src = trans[orig_col].astype(str).map(node_id_to_idx).to_numpy(dtype=np.int64)
    dst = trans[bene_col].astype(str).map(node_id_to_idx).to_numpy(dtype=np.int64)
    edge_index = torch.tensor(np.stack([src, dst], axis=0), dtype=torch.long)

    # Edge features: log1p(amount) (standardized) + time (scaled) + payment format (one-hot)
    amt = trans[amt_col].astype(float).to_numpy()
    amt_log = np.log1p(np.clip(amt, a_min=0.0, a_max=None)).reshape(-1, 1)
    amt_scaled = StandardScaler().fit_transform(amt_log).astype(np.float32)

    ts = trans[time_col]
    t0 = ts.min()
    # seconds since first timestamp (missing -> 0)
    dt_sec = (ts - t0).dt.total_seconds().fillna(0.0).to_numpy().reshape(-1, 1)
    dt_scaled = StandardScaler().fit_transform(dt_sec).astype(np.float32)

    payfmt = trans[payfmt_col].astype(str).to_numpy().reshape(-1, 1)
    pay_ohe = _make_ohe()
    pay_feat = pay_ohe.fit_transform(payfmt).astype(np.float32)

    # Include from/to bank as numeric (scaled) to help edge model a bit
    bfrom = trans[bank_from_col].to_numpy(dtype=np.float64).reshape(-1, 1)
    bto = trans[bank_to_col].to_numpy(dtype=np.float64).reshape(-1, 1)
    b_scaled = StandardScaler().fit_transform(np.concatenate([bfrom, bto], axis=1)).astype(np.float32)

    edge_attr = np.concatenate([amt_scaled, dt_scaled, b_scaled, pay_feat], axis=1)
    edge_attr = torch.tensor(edge_attr, dtype=torch.float32)

    y_edge = torch.tensor(trans[label_col].to_numpy(dtype=np.int64), dtype=torch.long)

    if add_degree_features:
        # Optional: degree can be a strong signal but may overfit structural artifacts.
        num_nodes = len(all_acct_ids)
        deg_out = np.bincount(src, minlength=num_nodes).astype(np.float32).reshape(-1, 1)
        deg_in = np.bincount(dst, minlength=num_nodes).astype(np.float32).reshape(-1, 1)
        deg = np.concatenate([deg_in, deg_out], axis=1)
        deg_scaled = StandardScaler().fit_transform(deg).astype(np.float32)
        x = np.concatenate([x, deg_scaled], axis=1)

    data = Data(
        x=torch.tensor(x, dtype=torch.float32),
        edge_index=edge_index,
        edge_attr=edge_attr,
        y_node=torch.tensor(y_node, dtype=torch.long),
        y_edge=y_edge,
    )

    # Masks / splits
    node_train, node_val, node_test = _build_splits_binary(
        y_node, train_size=train_size, val_size=val_size, seed=seed
    )
    node_train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    node_val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    node_test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    node_train_mask[node_train] = True
    node_val_mask[node_val] = True
    node_test_mask[node_test] = True
    data.node_train_mask = node_train_mask
    data.node_val_mask = node_val_mask
    data.node_test_mask = node_test_mask

    e_y = y_edge.cpu().numpy()
    edge_train, edge_val, edge_test = _build_splits_binary(
        e_y, train_size=train_size, val_size=val_size, seed=seed
    )
    split_node_label_counts = _split_label_counts(y_node, node_train, node_val, node_test)
    split_edge_label_counts = _split_label_counts(e_y, edge_train, edge_val, edge_test)
    data.edge_train_idx = torch.tensor(edge_train, dtype=torch.long)
    data.edge_val_idx = torch.tensor(edge_val, dtype=torch.long)
    data.edge_test_idx = torch.tensor(edge_test, dtype=torch.long)

    meta: Dict[str, Any] = {
        # Keep output compact + JSON-serializable (don’t dump giant ID maps).
        "dataset": "amlworld_hi_small",
        "num_nodes": int(data.num_nodes),
        "num_edges": int(data.num_edges),
        "max_transactions_loaded": int(len(trans)),
        "node_label_pos_rate": float(y_node.mean()),
        "edge_label_pos_rate": float(float(data.y_edge.float().mean().item())),
        "node_feature_dim": int(data.x.size(1)),
        "edge_feature_dim": int(data.edge_attr.size(1)),
        "node_feature_entity_categories": [c.tolist() for c in ohe.categories_],
        "edge_feature_payfmt_categories": [c.tolist() for c in pay_ohe.categories_],
        "t0": str(t0) if t0 is not pd.NaT else None,
        "schema": {
            "node_label": "derived is_sar_account (any endpoint of laundering tx)",
            "edge_label": label_col,
            "edge_amount_col": amt_col,
            "edge_time_col": time_col,
            "edge_payfmt_col": payfmt_col,
            "edge_direction": f"{orig_col} -> {bene_col}",
        },
        "split_policy": "size_matched_unstratified_plus_min1_then_round_robin_surplus_positives",
        "split_node_label_counts": split_node_label_counts,
        "split_edge_label_counts": split_edge_label_counts,
    }
    return data, meta

