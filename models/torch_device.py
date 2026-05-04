"""
Best-effort accelerator selection for PyTorch training.

Order: CUDA (NVIDIA) → MPS (Apple Silicon, e.g. M4) → CPU.
"""
from __future__ import annotations

from typing import Optional

import torch


def get_training_device(preference: Optional[str] = None) -> torch.device:
    """
    preference:
      None / "auto" — pick best available
      "cpu" | "cuda" | "mps" — force that backend if available (else CPU for cuda/mps when missing)
    """
    pref = (preference or "auto").strip().lower()
    if pref == "cpu":
        return torch.device("cpu")
    if pref == "cuda":
        return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if pref == "mps":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    # auto
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
