"""Pytest fixtures for CattleSense AI v2 testing.
Provides tiny planted-signal fixtures so unit and CI tests never require downloading 8 GB raw files.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def make_planted_herd_fixture(n_animals: int = 40, n_days: int = 60, seed: int = 42) -> pd.DataFrame:
    """Generate deterministic miniature herd with planted estrus surges every ~21 days."""
    rng = np.random.default_rng(seed)
    rows = []
    for a in range(n_animals):
        steps = rng.normal(120.0, 15.0, n_days).clip(20.0)
        temp = rng.normal(38.6, 0.15, n_days)
        estrus = np.zeros(n_days, dtype=int)
        for d in range(12, n_days - 6, 21):
            e = d + int(rng.integers(0, 3))
            estrus[e] = 1
            steps[e] *= 2.2
            temp[e] += 0.4
        rows.append(
            pd.DataFrame({
                "animal_id": f"cow_{a:02d}",
                "date": pd.date_range("2025-01-01", periods=n_days),
                "steps_sum": steps,
                "temp_mean": temp,
                "temp_q90": temp + 0.25,
                "estrus": estrus,
                "source": "fixture",
                "is_synthetic": 0,
            })
        )
    return pd.concat(rows, ignore_index=True)


@pytest.fixture
def herd_fixture() -> pd.DataFrame:
    return make_planted_herd_fixture()

