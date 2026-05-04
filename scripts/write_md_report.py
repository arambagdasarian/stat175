from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


def _f(x: Optional[float], digits: int = 3) -> str:
    if x is None:
        return "—"
    try:
        return f"{float(x):.{digits}f}"
    except Exception:
        return "—"


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--transfer_json", type=Path, required=True)
    p.add_argument("--pgb_json", type=Path, required=False, default=None)
    p.add_argument("--out_md", type=Path, required=True)
    args = p.parse_args(argv)

    tj = _read_json(args.transfer_json)
    dataset_meta = tj.get("dataset_meta", {})
    res = tj.get("result", {})

    node_block = res.get("node", {})
    if isinstance(node_block, dict) and node_block.get("skipped"):
        node_real_pr = None
        node_transfer_pr = None
    else:
        node_real_pr = (node_block.get("real_train_eval") or {}).get("test", {}).get("pr_auc")
        node_transfer_pr = (node_block.get("transfer_eval_on_real_test") or {}).get("pr_auc")
    edge_real_pr = res.get("edge", {}).get("real_train_eval", {}).get("test", {}).get("pr_auc")
    edge_transfer_pr = res.get("edge", {}).get("transfer_eval_on_real_test", {}).get("pr_auc")

    dp = res.get("dp_accounting", {})
    eps = {
        k: (dp.get(k, {}) or {}).get("epsilon_rdp")
        for k in ("node_real", "edge_real", "node_syn", "edge_syn")
    }

    lines: list[str] = []
    lines.append("# Run report\n")
    lines.append("## Transfer (TSTR) — PR-AUC\n")
    lines.append(f"- **transfer_json**: `{args.transfer_json}`")
    lines.append(f"- **device**: `{res.get('device')}`")
    lines.append(
        "- **slice**: "
        f"{int(dataset_meta.get('max_transactions_loaded', 0))} tx, "
        f"{int(dataset_meta.get('num_edges', 0))} edges, "
        f"{int(dataset_meta.get('num_nodes', 0))} nodes"
    )
    lines.append("")
    lines.append("| Task | Real → Real | Synthetic → Real |")
    lines.append("| ---- | -----------: | ----------------: |")
    lines.append(
        f"| Node (account fraud) | **{_f(node_real_pr, digits=4)}** | **{_f(node_transfer_pr, digits=4)}** |"
    )
    lines.append(
        f"| Edge (transaction fraud) | **{_f(edge_real_pr, digits=4)}** | **{_f(edge_transfer_pr, digits=4)}** |"
    )
    lines.append("")
    lines.append("## DP accounting (RDP ε)\n")
    lines.append("| Phase | ε (RDPAccountant) |")
    lines.append("| ----- | -----------------: |")
    for k in ("node_real", "edge_real", "node_syn", "edge_syn"):
        lines.append(f"| {k} | {_f(eps.get(k), digits=3)} |")
    lines.append("")

    if args.pgb_json is not None and args.pgb_json.is_file():
        pj = _read_json(args.pgb_json)
        runs = pj.get("runs", {})
        gm = runs.get("from_pt", {})
        pgb = gm.get("pgb_fifteen", {})
        q = pgb.get("queries", {})
        privacy = gm.get("empirical_privacy", {})
        lines.append("## PGB-style structural evaluation (GraphMaker)\n")
        lines.append(f"- **pgb_json**: `{args.pgb_json}`\n")
        lines.append("| Metric | Value |")
        lines.append("| ------ | ----: |")
        lines.append(f"| Mean scalar RE (capped) | {_f(pgb.get('summary_mean_RE_scalar_queries'))} |")
        lines.append(f"| Q6 KS statistic | {_f((q.get('Q6_degree_distribution', {}) or {}).get('KS_statistic'))} |")
        lines.append(f"| Q9 shortest-path L1 | {_f((q.get('Q9_distance_distribution', {}) or {}).get('L1_hist_diff'))} |")
        lines.append(f"| Q12 community NMI | {_f((q.get('Q12_community_detection', {}) or {}).get('NMI'))} |")
        lines.append(f"| Directed edge Jaccard | {_f((privacy.get('directed_edge_jaccard', {}) or {}).get('jaccard'))} |")
        lines.append(f"| Aligned node-feature mean L1 | {_f((privacy.get('aligned_node_feature_l1', {}) or {}).get('mean_l1'))} |")
        lines.append("")

    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text("\n".join(lines).rstrip() + "\n")
    print(f"Wrote {args.out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

