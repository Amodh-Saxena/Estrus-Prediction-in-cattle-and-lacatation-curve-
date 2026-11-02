"""Negative control and signal detection tests (FR-E3)."""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from sklearn.metrics import roc_auc_score

from src.features import add_features, FEATURES
from src.models.estrus import oof_scores
from tests.conftest import make_planted_herd_fixture


def test_signal_detection_and_shuffle_collapse() -> None:
    """Planted signal must yield high AUC; within-animal shuffle must collapse directly to chance (~0.50)."""
    herd = make_planted_herd_fixture(n_animals=30, n_days=50, seed=42)
    feat_df = add_features(herd)

    cfg = {"cv_folds": 4, "cv_repeats": 1, "seed": 42}
    scores_real = oof_scores(feat_df, FEATURES, "gbm", cfg=cfg)
    valid_mask = ~np.isnan(scores_real)
    auc_real = roc_auc_score(feat_df.loc[valid_mask, "estrus"], scores_real[valid_mask])
    assert auc_real > 0.85, f"Planted signal should be readily detected (got AUC={auc_real:.4f})"

    # Negative control: shuffle estrus labels strictly within each animal
    rng = np.random.default_rng(42)
    shuf_estrus = feat_df.groupby("animal_id")["estrus"].transform(lambda s: rng.permutation(s.values))
    feat_shuf = feat_df.copy()
    feat_shuf["estrus"] = shuf_estrus

    scores_shuf = oof_scores(feat_shuf, FEATURES, "gbm", cfg=cfg)
    auc_shuf = roc_auc_score(feat_shuf.loc[valid_mask, "estrus"], scores_shuf[valid_mask])
    assert abs(auc_shuf - 0.50) < 0.12, f"Shuffled control must collapse to chance (~0.50), got {auc_shuf:.4f}"
