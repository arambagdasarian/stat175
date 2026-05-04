"""
Run PGB-style structural comparison (real vs synthetic) for degree-preserving and GraphMaker exports.
Writes JSON to outputs/ (default).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from data.amlworld import load_amlworld_hi_small_pyg, resolve_hi_small_paths
from evaluation.pgb_style import pgb_style_bundle
from generators.degree_preserving import DegreePreservingGeneratorConfig, generate_degree_preserving_synthetic
from generators.graphmaker_bridge import load_synthetic_from_torch
from pipeline.json_sanitize import sanitize_for_json


def _write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sanitize_for_json(obj), indent=2, sort_keys=False, allow_nan=False))


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description="PGB-style structural evaluation (Liu et al. arXiv:2408.02928).")
    p.add_argument("--data_dir", type=Path, default=Path("data/raw"))
    p.add_argument(
        "--max_transactions",
        type=int,
        default=300_000,
        help="Transaction rows (prefix). Use 0 or --all_transactions for full CSV.",
    )
    p.add_argument(
        "--all_transactions",
        action="store_true",
        help="Load entire HI-Small_Trans (same as --max_transactions 0).",
    )
    p.add_argument(
        "--pgb_max_edge_rows",
        type=int,
        default=600_000,
        help="Cap directed edges projected into each PGB NetworkX build (full graph safe).",
    )
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--out_dir", type=Path, default=Path("outputs"))
    p.add_argument(
        "--synthetic_pt",
        type=Path,
        default=Path("outputs/graphmaker/synthetic_from_graphmaker.pt"),
    )
    args = p.parse_args(argv)
    max_tx = None if (args.all_transactions or args.max_transactions <= 0) else int(args.max_transactions)

    _ = resolve_hi_small_paths(args.data_dir)
    real, meta = load_amlworld_hi_small_pyg(
        args.data_dir,
        max_transactions=max_tx,
        seed=args.seed,
        add_degree_features=False,
    )

    syn_dp, meta_dp = generate_degree_preserving_synthetic(
        real, config=DegreePreservingGeneratorConfig(seed=args.seed)
    )
    bundle_dp = pgb_style_bundle(
        real,
        syn_dp,
        syn_meta=meta_dp,
        seed=args.seed,
        max_directed_edges=args.pgb_max_edge_rows,
    )

    syn_path = args.synthetic_pt.expanduser()
    if not syn_path.is_file():
        print(f"Missing synthetic graph: {syn_path}", file=sys.stderr)
        return 1
    syn_gm, meta_gm = load_synthetic_from_torch(syn_path, real, seed=args.seed)
    bundle_gm = pgb_style_bundle(
        real,
        syn_gm,
        syn_meta=meta_gm,
        seed=args.seed,
        max_directed_edges=args.pgb_max_edge_rows,
    )

    m_tag = int(meta.get("max_transactions_loaded", max_tx or 0))
    payload: Dict[str, Any] = {
        "dataset_meta": meta,
        "settings": {
            "seed": args.seed,
            "max_transactions": m_tag,
            "pgb_max_edge_rows": args.pgb_max_edge_rows,
            "synthetic_pt": str(syn_path.resolve()),
        },
        "pgb_note": "Structural stats use undirected simple graphs on shared V*; see evaluation/pgb_style.py.",
        "privacy_note": "These scores are not ε-DP; they measure graph fidelity vs real (PGB-style queries).",
        "runs": {
            "degree_preserving": bundle_dp,
            "from_pt": bundle_gm,
        },
    }
    out = args.out_dir / f"pgb_style_n{real.num_nodes}_e{real.num_edges}_m{m_tag}.json"
    _write_json(out, payload)
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
