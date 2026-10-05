"""Shared utilities: config loading, logging, IO helpers."""
from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def load_config(path: str | os.PathLike) -> dict[str, Any]:
    """Load YAML config file."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def ensure_dir(path: str | os.PathLike) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(obj: Any, path: str | os.PathLike) -> None:
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)


def load_json(path: str | os.PathLike) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def bin_value(value: float, bins: list[float], labels: list[str]) -> str:
    """Assign value to a labelled bin. bins = [b0, b1, b2, ...]; len(labels)=len(bins)-1."""
    for i in range(len(labels)):
        if bins[i] <= value < bins[i + 1]:
            return labels[i]
    return labels[-1]
