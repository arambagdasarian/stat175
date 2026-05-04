"""
Exit 0 if DGL (with GraphBolt) and GraphMaker training deps import.

DGL 2.2.x on macOS ships ``libgraphbolt_pytorch_<torch>.dylib`` only for specific
PyTorch versions (e.g. 2.2.x–2.3.0). If ``import dgl`` fails, align torch/dgl::

  pip install -e ".[graphmaker]"
"""
from __future__ import annotations

import sys


def main() -> int:
    try:
        import torch
    except Exception as e:
        print(f"torch import failed: {e}", file=sys.stderr)
        return 1
    tv = torch.__version__.split("+", maxsplit=1)[0]
    print(f"torch {torch.__version__}")

    try:
        import dgl  # noqa: F401
    except Exception as e:
        print(
            f"dgl import failed ({type(e).__name__}: {e})\n"
            "On macOS, install a torch version that matches a bundled GraphBolt dylib, e.g.\n"
            "  pip install 'torch>=2.2.2,<=2.3.0' 'dgl>=2.2,<2.3' -f https://data.dgl.ai/wheels/repo.html\n"
            "  pip install 'torchdata>=0.7.1,<0.8'\n"
            "See pyproject optional dependency group ``graphmaker``.",
            file=sys.stderr,
        )
        return 1
    import dgl as _dgl

    print(f"dgl {_dgl.__version__}")

    try:
        import wandb  # noqa: F401
    except ImportError:
        print("wandb missing (GraphMaker train script imports it). pip install wandb", file=sys.stderr)
        return 1

    print(f"GraphBolt path OK for torch {tv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
