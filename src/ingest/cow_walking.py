"""Ingest 10 Hz collar IMU clips from `db-cow-walking-main`."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from src.common import ROOT, require

COLLAR_ROOT = ROOT / "db-cow-walking-main"

CLASS_MAP_3 = {"Walking": "Walking", "Grazing": "Grazing", "Resting": "Resting"}
CLASS_MAP_4 = {
    "Walking": "Walking",
    "Grazing": "Grazing",
    "Resting": "Resting",
    "Standing": "Standing",
}


def load_real_clips(extended: bool = True) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Load and extract signal features for collar clips from `db-cow-walking-main`."""
    require("db_cow_walking", "behaviour_benchmark")
    from src.models.behaviour import extract_clip_features

    if not COLLAR_ROOT.exists():
        return pd.DataFrame(), np.empty(0), np.empty(0)

    rows: list[dict[str, float]] = []
    labels: list[str] = []
    cows: list[str] = []

    subdirs = [("Walking", "Walking"), ("Grazing", "Grazing"), ("Resting", "Resting")]
    if extended:
        subdirs.append(("Miscellaneous behaviors", "Standing"))

    for folder_name, target_class in subdirs:
        folder = COLLAR_ROOT / folder_name
        if not folder.exists():
            continue
        for csv_path in sorted(folder.glob("*.csv")):
            parts = csv_path.stem.split("_")
            if folder_name == "Miscellaneous behaviors":
                if len(parts) < 4 or parts[1] != "Standing":
                    continue
                cow_id = parts[2]
            else:
                cow_id = parts[2] if len(parts) >= 3 else "unknown"

            feat = extract_clip_features(csv_path)
            if feat:
                rows.append(feat)
                labels.append(target_class)
                cows.append(cow_id)

    if not rows:
        return pd.DataFrame(), np.empty(0), np.empty(0)

    x_df = pd.DataFrame(rows).fillna(0.0)
    return x_df, np.asarray(labels), np.asarray(cows)

