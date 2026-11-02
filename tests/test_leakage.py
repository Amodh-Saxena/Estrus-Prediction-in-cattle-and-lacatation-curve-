"""Test prevention of data leakage and provenance integrity (FR-A1, FR-A3)."""
from __future__ import annotations

import pandas as pd
import pytest

from src.audit import animal_overlap
from src.features import assert_real_only
from src.evaluate import split_animals


def test_animal_overlap_detection() -> None:
    """Disjoint sets must pass; shared animal IDs must raise AssertionError."""
    train_cows = {"cow_01", "cow_02", "cow_03"}
    test_cows = {"cow_04", "cow_05"}
    animal_overlap(train_cows, test_cows)

    with pytest.raises(AssertionError, match="LEAKAGE DETECTED"):
        animal_overlap({"cow_01", "cow_02"}, {"cow_02", "cow_03"})


def test_split_animals_disjointness() -> None:
    """Verify split_animals produces strictly zero-overlap sets."""
    all_cows = [f"cow_{i:02d}" for i in range(50)]
    dev, test = split_animals(all_cows, test_frac=0.20, seed=42)
    assert len(set(dev) & set(test)) == 0
    assert len(dev) + len(test) == 50


def test_assert_real_only_enforcement() -> None:
    """assert_real_only must reject synthetic data or fixture sources."""
    clean_df = pd.DataFrame({"animal_id": [1, 2], "is_synthetic": [0, 0], "source": ["real", "real"]})
    assert_real_only(clean_df)

    synthetic_df = pd.DataFrame({"animal_id": [1, 2], "is_synthetic": [1, 0], "source": ["real", "real"]})
    with pytest.raises(AssertionError, match="DATA INTEGRITY ERROR"):
        assert_real_only(synthetic_df)

    sim_source_df = pd.DataFrame({"animal_id": [1, 2], "is_synthetic": [0, 0], "source": ["simulated_herd", "real"]})
    with pytest.raises(AssertionError, match="DATA INTEGRITY ERROR"):
        assert_real_only(sim_source_df)

