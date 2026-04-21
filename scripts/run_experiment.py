from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from data.amlworld import load_amlworld_hi_small_pyg, resolve_hi_small_paths
from data.dummy import DummyConfig, make_dummy_pyg
from generators.degree_preserving import DegreePreservingGeneratorConfig
from generators.graphmaker_bridge import load_synthetic_from_torch
from pipeline.json_sanitize import sanitize_for_json
from pipeline.transfer_experiment import ModelConfig, run_transfer_experiment


def _maybe_kaggle_download(data_dir: Path) -> None:
    """
    Best-effort download using Kaggle CLI (requires `kaggle` configured).
    Mirrors the approach in your notebook.
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    # `python -m kaggle` is not available in recent kaggle releases; invoke CLI via `kaggle.cli`.
    cmd = [
        sys.executable,
        "-c",
        (
            "import sys; "
            "from kaggle.cli import main; "
            "sys.argv = ['kaggle'] + sys.argv[1:]; "
            "main()"
        ),
        "datasets",
        "download",
        "-d",
        "ealtman2019/ibm-transactions-for-anti-money-laundering-aml",
        "-p",
        str(data_dir),
        "--unzip",
    ]
    subprocess.run(cmd, check=False)


def _write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    clean = sanitize_for_json(obj)
    path.write_text(json.dumps(clean, indent=2, sort_keys=False, allow_nan=False))


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=Path, default=Path("data/raw"))
    p.add_argument("--variant", type=str, default="hi_small", choices=["hi_small"])
    p.add_argument("--max_transactions", type=int, default=300_000)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--out_dir", type=Path, default=Path("outputs"))
    p.add_argument(
        "--allow_dummy",
        action="store_true",
        help="If AMLWorld CSVs are missing, use a random dummy graph instead of exiting (smoke tests only).",
    )
    p.add_argument(
        "--generator",
        type=str,
        default="from_pt",
        choices=["from_pt", "degree_preserving"],
        help="Synthetic graph: GraphMaker export (from_pt, default) or built-in degree-preserving baseline.",
    )
    p.add_argument(
        "--synthetic_pt",
        type=Path,
        default=Path("outputs/graphmaker/synthetic_from_graphmaker.pt"),
        help="torch.save of PyG Data (or dict with 'data') when --generator from_pt.",
    )
    args = p.parse_args(argv)

    if args.variant != "hi_small":
        raise ValueError("Only hi_small is supported in v0.1")

    # Ensure dataset exists (or try to download). If Kaggle isn't configured, fall back to a dummy
    # graph so the rest of the pipeline can still be executed end-to-end.
    real = None
    meta: Dict[str, Any]
    if args.variant == "hi_small":
        try:
            _ = resolve_hi_small_paths(args.data_dir)
            real, meta = load_amlworld_hi_small_pyg(
                args.data_dir,
                max_transactions=args.max_transactions,
                seed=args.seed,
                add_degree_features=False,
            )
        except FileNotFoundError:
            print(
                f"AMLWorld files not found under {args.data_dir}/.\n"
                "Attempting Kaggle download (requires Kaggle CLI auth)...",
                file=sys.stderr,
            )
            _maybe_kaggle_download(args.data_dir)
            try:
                _ = resolve_hi_small_paths(args.data_dir)
                real, meta = load_amlworld_hi_small_pyg(
                    args.data_dir,
                    max_transactions=args.max_transactions,
                    seed=args.seed,
                    add_degree_features=False,
                )
            except Exception as e:
                if not args.allow_dummy:
                    raise
                print(
                    "Kaggle download unavailable (likely missing `~/.kaggle/kaggle.json`).\n"
                    "Falling back to a dummy graph so the pipeline still runs end-to-end.\n"
                    f"Root cause: {type(e).__name__}: {e}",
                    file=sys.stderr,
                )
                real, meta = make_dummy_pyg(DummyConfig(seed=args.seed))
    assert real is not None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    synthetic_bundle = None
    if args.generator == "from_pt":
        syn_path = args.synthetic_pt.expanduser()
        if not syn_path.is_file():
            print(
                "Missing GraphMaker synthetic graph:\n"
                f"  {syn_path.resolve()}\n"
                "Train and sample first, for example:\n"
                "  WANDB_MODE=disabled python3 -m scripts.graphmaker_train_amlworld "
                "--data_dir data/raw --out_dir outputs/graphmaker\n"
                "  export AMLWORLD_DGL_GRAPH=$(pwd)/outputs/graphmaker/amlworld_graph.bin\n"
                "  python3 -m scripts.graphmaker_sample_to_pyg "
                "--model_path outputs/graphmaker/amlworld_cpts/Async_TX6_TE9.pth "
                "--out_pt outputs/graphmaker/synthetic_from_graphmaker.pt\n"
                "Or use --generator degree_preserving for a quick baseline without GraphMaker.",
                file=sys.stderr,
            )
            return 1
        synthetic_bundle = load_synthetic_from_torch(syn_path, real, seed=args.seed)

    result = run_transfer_experiment(
        real,
        model_cfg=ModelConfig(seed=args.seed),
        gen_cfg=DegreePreservingGeneratorConfig(seed=args.seed),
        synthetic_bundle=synthetic_bundle,
        device=device,
    )

    payload = {"dataset_meta": meta, "result": result}
    tag = "graphmaker" if args.generator == "from_pt" else "degree_preserving"
    out_path = (
        args.out_dir
        / f"transfer_hi_small_n{real.num_nodes}_e{real.num_edges}_m{args.max_transactions}_{tag}.json"
    )
    _write_json(out_path, payload)
    print(f"Wrote results to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

