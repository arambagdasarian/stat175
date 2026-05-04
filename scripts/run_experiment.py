from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from data.amlworld import load_amlworld_hi_small_pyg, resolve_hi_small_paths
from models.torch_device import get_training_device
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
    p.add_argument(
        "--max_transactions",
        type=int,
        default=600_000,
        help="Rows from HI-Small_Trans: with --slice_mode prefix, first N rows; with balanced_edges, target edge count after rebalancing.",
    )
    p.add_argument(
        "--all_transactions",
        action="store_true",
        help="Load entire HI-Small_Trans (same as --max_transactions 0). Expect large RAM and slow training.",
    )
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--out_dir", type=Path, default=Path("outputs"))
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
    p.add_argument(
        "--use_dp",
        action="store_true",
        help="Train node/edge models with DP-SGD on random k-hop subgraphs (see models/dp_train.py).",
    )
    p.add_argument(
        "--edges_only",
        action="store_true",
        help="Skip account/node classifier: train encoder+edge head on edge loss only (real then synthetic).",
    )
    p.add_argument(
        "--checkpoint_dir",
        type=Path,
        default=None,
        help=(
            "With --use_dp only: save DP checkpoints under this directory each epoch "
            "(node_real_*, edge_real_*, node_syn_*, edge_syn_*) plus *_best.pt on val PR-AUC improvement."
        ),
    )
    p.add_argument("--dp_noise_multiplier", type=float, default=1.0)
    p.add_argument("--dp_max_grad_norm", type=float, default=1.0)
    p.add_argument("--dp_delta", type=float, default=1e-5)
    p.add_argument("--dp_batch_size", type=int, default=512)
    p.add_argument("--dp_steps_per_epoch", type=int, default=40)
    p.add_argument("--dp_num_hops", type=int, default=2)
    p.add_argument(
        "--dp_pos_to_neg_ratio",
        type=int,
        default=3,
        help="Balanced DP minibatches: desired labeled pos:neg ratio per step when both exist (e.g. 1 -> ~1:1, 3 -> ~3:1).",
    )
    p.add_argument("--node_epochs", type=int, default=None, help="Override ModelConfig.node_epochs.")
    p.add_argument("--edge_epochs", type=int, default=None, help="Override ModelConfig.edge_epochs.")
    p.add_argument("--syn_node_epochs", type=int, default=None, help="Override ModelConfig.syn_node_epochs.")
    p.add_argument("--syn_edge_epochs", type=int, default=None, help="Override ModelConfig.syn_edge_epochs.")
    p.add_argument(
        "--min_synthetic_nodes",
        type=int,
        default=50_000,
        help="When --generator from_pt: require synthetic graph to have at least this many nodes (re-sample GraphMaker if smaller).",
    )
    p.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Training device: auto picks CUDA, else Apple MPS (Metal), else CPU.",
    )
    p.add_argument(
        "--slice_mode",
        type=str,
        default="prefix",
        choices=["prefix", "balanced_edges"],
        help="balanced_edges oversamples positives in the transaction window for stabler PR-AUC/ROC (requires finite --max_transactions).",
    )
    p.add_argument(
        "--balance_scan_rows",
        type=int,
        default=2_000_000,
        help="balanced_edges: read at most this many CSV rows before subsampling to --max_transactions.",
    )
    p.add_argument(
        "--target_edge_pos_fraction",
        type=float,
        default=0.05,
        help="balanced_edges: target laundering edge fraction in the loaded slice (capped by positives in scan window).",
    )
    p.add_argument(
        "--no_stratify_edges",
        action="store_true",
        help="Do not use sklearn stratified edge train/val/test when viable.",
    )
    p.add_argument(
        "--no_stratify_nodes",
        action="store_true",
        help="Do not use sklearn stratified node train/val/test when viable.",
    )
    args = p.parse_args(argv)
    max_tx: Optional[int] = None if (args.all_transactions or args.max_transactions <= 0) else int(args.max_transactions)

    if args.variant != "hi_small":
        raise ValueError("Only hi_small is supported in v0.1")
    if args.slice_mode == "balanced_edges" and max_tx is None:
        print(
            "error: --slice_mode balanced_edges needs a finite --max_transactions (omit --all_transactions).",
            file=sys.stderr,
        )
        return 1

    load_kw: Dict[str, Any] = dict(
        max_transactions=max_tx,
        seed=args.seed,
        add_degree_features=False,
        slice_mode=args.slice_mode,
        balance_scan_rows=int(args.balance_scan_rows),
        target_edge_pos_fraction=float(args.target_edge_pos_fraction),
        stratify_edges_if_possible=not args.no_stratify_edges,
        stratify_nodes_if_possible=not args.no_stratify_nodes,
    )

    # Load AMLWorld CSVs from --data_dir, or try a one-shot Kaggle download if missing.
    try:
        _ = resolve_hi_small_paths(args.data_dir)
        real, meta = load_amlworld_hi_small_pyg(args.data_dir, **load_kw)
    except FileNotFoundError:
        print(
            f"AMLWorld files not found under {args.data_dir}/.\n"
            "Attempting Kaggle download (requires Kaggle CLI auth)...",
            file=sys.stderr,
        )
        _maybe_kaggle_download(args.data_dir)
        try:
            _ = resolve_hi_small_paths(args.data_dir)
            real, meta = load_amlworld_hi_small_pyg(args.data_dir, **load_kw)
        except Exception as e:
            raise RuntimeError(
                "Could not load AMLWorld HI-Small after attempting Kaggle download.\n"
                f"Put HI-Small_Trans.csv and HI-Small_accounts.csv under {args.data_dir.resolve()}/, "
                "or configure ~/.kaggle/kaggle.json for the CLI.\n"
                f"Original error: {type(e).__name__}: {e}"
            ) from e

    device = get_training_device(None if args.device == "auto" else args.device)
    synthetic_torch_path: Optional[Path] = None
    if args.generator == "from_pt":
        syn_path = args.synthetic_pt.expanduser()
        if not syn_path.is_file():
            print(
                "Missing GraphMaker synthetic graph:\n"
                f"  {syn_path.resolve()}\n"
                "Train and sample first, for example:\n"
                "  WANDB_MODE=disabled python3 -m scripts.graphmaker_train_amlworld "
                "--data_dir data/raw --out_dir outputs/graphmaker "
                "(optional: --slice_mode balanced_edges --target_edge_pos_fraction 0.06)\n"
                "  export AMLWORLD_DGL_GRAPH=$(pwd)/outputs/graphmaker/amlworld_graph.bin\n"
                "  python3 -m scripts.graphmaker_sample_to_pyg "
                "--model_path outputs/graphmaker/amlworld_cpts/Async_TX6_TE9.pth "
                "--out_pt outputs/graphmaker/synthetic_from_graphmaker.pt "
                "(optional: same --slice_mode/...; --syn_edge_fraud_rate_floor 0.08 for synthetic PR-AUC)\n"
                "Or use --generator degree_preserving for a quick baseline without GraphMaker.",
                file=sys.stderr,
            )
            return 1
        # Lazy-loaded inside run_transfer_experiment after real training so RNG matches
        # the degree-preserving run for the same seed (see pipeline/transfer_experiment.py).
        synthetic_torch_path = syn_path

    mc_kwargs: Dict[str, Any] = dict(
        seed=args.seed,
        edges_only=args.edges_only,
        use_dp=args.use_dp,
        dp_noise_multiplier=args.dp_noise_multiplier,
        dp_max_grad_norm=args.dp_max_grad_norm,
        dp_delta=args.dp_delta,
        dp_batch_size=args.dp_batch_size,
        dp_steps_per_epoch=args.dp_steps_per_epoch,
        dp_num_hops=args.dp_num_hops,
        dp_pos_to_neg_ratio=args.dp_pos_to_neg_ratio,
    )
    if args.node_epochs is not None:
        mc_kwargs["node_epochs"] = args.node_epochs
    if args.edge_epochs is not None:
        mc_kwargs["edge_epochs"] = args.edge_epochs
    if args.syn_node_epochs is not None:
        mc_kwargs["syn_node_epochs"] = args.syn_node_epochs
    if args.syn_edge_epochs is not None:
        mc_kwargs["syn_edge_epochs"] = args.syn_edge_epochs
    ckpt_dir = args.checkpoint_dir.expanduser() if args.checkpoint_dir is not None else None
    if ckpt_dir is not None:
        ckpt_dir.mkdir(parents=True, exist_ok=True)

    gen_cfg = None
    if args.generator == "degree_preserving":
        from generators.degree_preserving import DegreePreservingGeneratorConfig

        gen_cfg = DegreePreservingGeneratorConfig(seed=args.seed)

    result = run_transfer_experiment(
        real,
        model_cfg=ModelConfig(**mc_kwargs),
        gen_cfg=gen_cfg,
        synthetic_torch_path=synthetic_torch_path,
        min_synthetic_nodes=int(args.min_synthetic_nodes),
        device=device,
        checkpoint_dir=ckpt_dir,
    )

    payload = {"dataset_meta": meta, "result": result}
    tag = "graphmaker" if args.generator == "from_pt" else "degree_preserving"
    m_tag = int(meta.get("max_transactions_loaded", max_tx or 0))
    suffix = tag
    if args.edges_only:
        suffix = f"{suffix}_edgesonly"
    if args.use_dp:
        suffix = f"{suffix}_dp"
    out_path = args.out_dir / f"transfer_hi_small_n{real.num_nodes}_e{real.num_edges}_m{m_tag}_{suffix}.json"
    _write_json(out_path, payload)
    print(f"Wrote results to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

