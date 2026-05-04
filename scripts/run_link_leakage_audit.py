"""
Run link-level leakage audit: predict real edges from synthetic graph signals.

  python3 -m scripts.run_link_leakage_audit --data_dir data/raw --max_transactions 80000 \\
    --synthetic_pt outputs/graphmaker/synthetic_from_graphmaker.pt

  python3 -m scripts.run_link_leakage_audit --generator degree_preserving ...

Requires aligned num_nodes / node features between real and synthetic Data.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import torch

from data.amlworld import load_amlworld_hi_small_pyg
from evaluation.link_leakage import LinkLeakageConfig, run_link_leakage_audit
from generators.degree_preserving import DegreePreservingGeneratorConfig, generate_degree_preserving_synthetic
from generators.graphmaker_bridge import load_synthetic_from_torch
from pipeline.json_sanitize import sanitize_for_json


def _load_syn(
    real,
    *,
    generator: str,
    synthetic_pt: Path,
    seed: int,
):
    if generator == "degree_preserving":
        return generate_degree_preserving_synthetic(real, config=DegreePreservingGeneratorConfig(seed=seed))[0]
    if generator == "from_pt":
        p = synthetic_pt.expanduser()
        if not p.is_file():
            print(f"Missing {p}", file=sys.stderr)
            sys.exit(1)
        return load_synthetic_from_torch(p, real, seed=seed)[0]
    raise SystemExit(f"Unknown --generator {generator}")


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=Path, default=Path("data/raw"))
    ap.add_argument("--max_transactions", type=int, default=80_000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--generator", choices=["from_pt", "degree_preserving"], default="from_pt")
    ap.add_argument("--synthetic_pt", type=Path, default=Path("outputs/graphmaker/synthetic_from_graphmaker.pt"))
    ap.add_argument("--n_positive", type=int, default=4000)
    ap.add_argument("--n_negative", type=int, default=4000)
    ap.add_argument("--out_json", type=Path, default=Path("outputs/link_leakage_audit.json"))
    args = ap.parse_args(argv)

    real, meta = load_amlworld_hi_small_pyg(args.data_dir, max_transactions=args.max_transactions, seed=args.seed)
    syn = _load_syn(real, generator=args.generator, synthetic_pt=args.synthetic_pt, seed=args.seed)

    try:
        result = run_link_leakage_audit(
            real,
            syn,
            cfg=LinkLeakageConfig(
                n_positive=args.n_positive,
                n_negative=args.n_negative,
                random_state=args.seed,
            ),
        )
    except ValueError as e:
        print(str(e), file=sys.stderr)
        return 1

    settings = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    payload = {"dataset_meta": meta, "settings": settings, "result": result}
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(sanitize_for_json(payload), indent=2, allow_nan=False))
    print(f"Wrote {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
