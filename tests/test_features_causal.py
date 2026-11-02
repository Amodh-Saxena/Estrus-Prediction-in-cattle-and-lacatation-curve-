"""Test strict temporal causality of feature engineering (FR-B4)."""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from src.features import add_features, FEATURES
from tests.conftest import make_planted_herd_fixture


def test_causal_rolling_baseline_invariance() -> None:
    """Perturbing sensor values strictly after day t must leave day <= t features bitwise identical."""
    herd = make_planted_herd_fixture(n_animals=10, n_days=50, seed=42)
    cutoff_date = pd.Timestamp("2025-01-25")

    # Baseline computation on original time series
    feat_orig = add_features(herd)
    before_orig = feat_orig[feat_orig["date"] <= cutoff_date][FEATURES].copy()

    # Perturb all future sensor readings 10x after cutoff_date
    herd_perturbed = herd.copy()
    future_mask = herd_perturbed["date"] > cutoff_date
    herd_perturbed.loc[future_mask, "steps_sum"] *= 10.0
    herd_perturbed.loc[future_mask, "temp_mean"] += 5.0
    herd_perturbed.loc[future_mask, "temp_q90"] += 5.0

    feat_pert = add_features(herd_perturbed)
    before_pert = feat_pert[feat_pert["date"] <= cutoff_date][FEATURES].copy()

    assert_frame_equal(before_orig, before_pert, check_exact=False, atol=1e-10)


def test_today_excluded_from_own_baseline() -> None:
    """Altering only today's sensor reading must not alter today's baseline ratio denominator."""
    herd = make_planted_herd_fixture(n_animals=5, n_days=30, seed=123)
    target_idx = 15
    row_date = herd.iloc[target_idx]["date"]
    aid = herd.iloc[target_idx]["animal_id"]

    orig_feat = add_features(herd)
    orig_z_lag1 = orig_feat.loc[
        (orig_feat["animal_id"] == aid) & (orig_feat["date"] == row_date), "steps_sum_z_lag1"
    ].iloc[0]

    # Change today's reading
    herd_mod = herd.copy()
    herd_mod.loc[(herd_mod["animal_id"] == aid) & (herd_mod["date"] == row_date), "steps_sum"] *= 3.0
    mod_feat = add_features(herd_mod)
    mod_z_lag1 = mod_feat.loc[
        (mod_feat["animal_id"] == aid) & (mod_feat["date"] == row_date), "steps_sum_z_lag1"
    ].iloc[0]

    # Yesterday's z-score lag1 must be completely identical
    assert np.isclose(orig_z_lag1, mod_z_lag1, atol=1e-10)
