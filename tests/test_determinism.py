"""Test pipeline determinism and seed repeatability (FR-E3)."""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.features import add_features, FEATURES
from src.models.estrus import oof_scores
from tests.conftest import make_planted_herd_fixture


def test_seed_determinism() -> None:
    """Identical seed must yield bitwise identical out-of-fold probability predictions."""
    herd = make_planted_herd_fixture(n_animals=15, n_days=35, seed=77)
    feat_df = add_features(herd)

    cfg = {"cv_folds": 3, "cv_repeats": 1, "seed": 42}
    scores_run1 = oof_scores(feat_df, FEATURES, "gbm", cfg=cfg)
    scores_run2 = oof_scores(feat_df, FEATURES, "gbm", cfg=cfg)

    np.testing.assert_allclose(scores_run1, scores_run2, atol=1e-12)
