"""Print x / edge_attr dimensions for a given AMLWorld slice (for GraphMaker conversion)."""
from __future__ import annotations

import argparse
from pathlib import Path

from data.amlworld import load_amlworld_hi_small_pyg


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=Path, default=Path("data/raw"))
    p.add_argument("--max_transactions", type=int, default=300_000)
    args = p.parse_args()
    data, meta = load_amlworld_hi_small_pyg(
        args.data_dir, max_transactions=args.max_transactions, seed=7
    )
    print("num_nodes", data.num_nodes)
    print("num_edges", data.num_edges)
    print("x.shape", tuple(data.x.shape))
    print("edge_attr.shape", tuple(data.edge_attr.shape))
    print("meta keys:", list(meta.keys()) if isinstance(meta, dict) else meta)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
