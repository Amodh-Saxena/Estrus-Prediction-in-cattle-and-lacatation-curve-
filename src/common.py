"""Common configuration, path resolution, dataset registry validation, and reproducibility seeds."""
from __future__ import annotations

from pathlib import Path
import random
from typing import Any
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_config(config_path: Path | None = None) -> dict[str, Any]:
    """Load configuration YAML from config.yaml."""
    cfg_p = config_path or (ROOT / "config.yaml")
    if not cfg_p.exists():
        return {}
    with open(cfg_p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_priors(priors_path: Path | None = None) -> dict[str, Any]:
    """Load literature priors from data/priors.yaml."""
    p_path = priors_path or (ROOT / "data" / "priors.yaml")
    if not p_path.exists():
        return {}
    with open(p_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def require(dataset: str, use: str, registry_path: Path | None = None) -> dict[str, Any]:
    """Enforce data governance: abort unless dataset is registered and approved for the requested use."""
    r_path = registry_path or (ROOT / "data" / "registry.yaml")
    if not r_path.exists():
        raise SystemExit(f"Data registry missing at {r_path}")
    with open(r_path, "r", encoding="utf-8") as f:
        reg = yaml.safe_load(f)
    datasets = reg.get("datasets", {})
    if dataset not in datasets:
        raise SystemExit(f"Dataset '{dataset}' is not registered in data/registry.yaml")
    allowed = datasets[dataset].get("allowed_use", [])
    if use not in allowed:
        raise SystemExit(f"Dataset '{dataset}' is not approved for use '{use}' (allowed: {allowed})")
    return datasets[dataset]


def seed_all(seed: int = 42) -> None:
    """Set seeds across Python random and numpy for deterministic pipelines."""
    random.seed(seed)
    np.random.seed(seed)

