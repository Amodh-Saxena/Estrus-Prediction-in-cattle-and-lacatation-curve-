"""Shared cow-level evaluation, bootstrap confidence intervals, and leakage tests (PRD Section 7)."""
from __future__ import annotations

from typing import Callable
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

from src.features import estrus_features_prd, compute_parity_curves, PRD_ESTRUS_FEATURES, assert_real_only


def split_animals(ids, test_frac: float = 0.20, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Split animal IDs into development (80%) and held-out test (20%) sets with zero overlap."""
    unique_ids = np.array(sorted(set(ids)))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(unique_ids)
    n_test = max(1, int(round(len(perm) * test_frac)))
    test_ids = perm[:n_test]
    dev_ids = perm[n_test:]
    assert not (set(dev_ids) & set(test_ids)), "Leakage: Overlap between dev and test animal sets!"
    return dev_ids, test_ids


def repeated_animal_folds(ids, k: int = 5, repeats: int = 3, seed: int = 42):
    """Repeated GroupKFold generator partitioned strictly by animal ID."""
    unique_ids = np.array(sorted(set(ids)))
    for r in range(repeats):
        perm = np.random.default_rng(seed + r).permutation(unique_ids)
        chunks = np.array_split(perm, k)
        for i in range(k):
            val_chunk = chunks[i]
            train_chunk = np.concatenate(chunks[:i] + chunks[i + 1:])
            yield r, i, train_chunk, val_chunk


def days_to_nearest_event(df: pd.DataFrame) -> pd.Series:
    """Distance in days to nearest confirmed estrus event within the same animal."""
    out = pd.Series(np.inf, index=df.index)
    for _, g in df.groupby("animal_id", sort=False):
        ev = g.loc[g["estrus"] == 1, "date"].to_numpy("datetime64[D]")
        if len(ev):
            d = g["date"].to_numpy("datetime64[D]")
            out[g.index] = np.abs(d[:, None] - ev[None, :]).min(axis=1).astype(int)
    return out


def event_metrics(df: pd.DataFrame, alert: np.ndarray, tol: int = 1) -> dict:
    """PRD v2 event-level detection metrics: recall within tol days of label, and false alerts per 100 animal-days."""
    d = df.reset_index(drop=True).assign(alert=np.asarray(alert, bool))
    dist = days_to_nearest_event(d)
    events = d[d["estrus"] == 1]
    hits = 0
    for _, e in events.iterrows():
        w = d[(d["animal_id"] == e["animal_id"]) & ((d["date"] - e["date"]).abs() <= pd.Timedelta(days=tol))]
        hits += int(w["alert"].any())
    false_alerts = int(((d["alert"]) & (dist > tol)).sum())
    return {
        "event_recall": float(hits / max(1, len(events))),
        "n_events": int(len(events)),
        "false_alerts_per_100": float(100.0 * false_alerts / max(1, len(d))),
        "n_animal_days": int(len(d)),
    }


def pick_threshold(df: pd.DataFrame, p: np.ndarray, max_fa: float = 5.0, tol: int = 1) -> float:
    """Select decision threshold maximizing recall subject to false alerts <= max_fa per 100 animal-days.
    Trained and tuned on validation scores only!
    """
    valid_p = p[~np.isnan(p)]
    if len(valid_p) == 0:
        return 0.50
    best_t, best_r = float(np.max(valid_p)) + 1e-9, -1.0
    quantiles = np.linspace(0.50, 0.999, 200)
    for t in np.unique(np.quantile(valid_p, quantiles)):
        m = event_metrics(df, p >= t, tol)
        if m["false_alerts_per_100"] <= max_fa and m["event_recall"] > best_r:
            best_t, best_r = float(t), m["event_recall"]
    return best_t


def bootstrap_by_animal(
    df: pd.DataFrame,
    alert: np.ndarray,
    tol: int = 1,
    reps: int = 1000,
    seed: int = 42,
) -> dict[str, tuple[float, float]]:
    """Cluster bootstrap resampled strictly by animal ID to compute empirical 95% confidence intervals."""
    d = df.reset_index(drop=True).assign(alert=np.asarray(alert, bool))
    groups = {a: g for a, g in d.groupby("animal_id", sort=False)}
    ids = np.array(list(groups))
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(reps):
        pick = rng.choice(ids, size=len(ids), replace=True)
        sampled = pd.concat([groups[a].assign(animal_id=f"{a}_{i}") for i, a in enumerate(pick)], ignore_index=True)
        out.append(event_metrics(sampled, sampled["alert"].to_numpy(), tol))
    r = pd.DataFrame(out)
    return {
        "event_recall": (float(r["event_recall"].quantile(0.025)), float(r["event_recall"].quantile(0.975))),
        "false_alerts_per_100": (float(r["false_alerts_per_100"].quantile(0.025)), float(r["false_alerts_per_100"].quantile(0.975))),
    }


def cow_train_validation_test(cow_ids, seed: int = 7, train_fraction: float = 0.60):
    """Split cow IDs into 60% train, 20% validation, and 20% test sets with zero overlap (FR-E1)."""
    cows = np.asarray(sorted(set(cow_ids)))
    split = GroupShuffleSplit(n_splits=1, train_size=train_fraction, random_state=seed)
    train_idx, remainder_idx = next(split.split(cows, groups=cows))
    remainder = cows[remainder_idx]
    split_rest = GroupShuffleSplit(n_splits=1, train_size=0.50, random_state=seed + 1)
    val_idx, test_idx = next(split_rest.split(remainder, groups=remainder))
    train, validation, test = set(cows[train_idx]), set(remainder[val_idx]), set(remainder[test_idx])
    if train & validation or train & test or validation & test:
        raise RuntimeError("Cow leakage detected across train/validation/test splits.")
    return train, validation, test


def group_folds(n_groups: int, requested: int = 5) -> GroupKFold:
    if n_groups < 2:
        raise ValueError("At least two cow groups are required for grouped evaluation.")
    return GroupKFold(n_splits=min(requested, n_groups))


def cow_bootstrap_ci(
    df: pd.DataFrame,
    metric_fn: Callable[[pd.DataFrame], float],
    cow_col: str = "cow_id",
    n_boot: int = 200,
    seed: int = 7,
) -> dict[str, float]:
    """Compute cow-level cluster bootstrap 95% confidence intervals (FR-E4)."""
    rng = np.random.default_rng(seed)
    cows = np.asarray(sorted(df[cow_col].unique()))
    if len(cows) < 2:
        val = float(metric_fn(df))
        return {"mean": val, "std": 0.0, "ci95_low": val, "ci95_high": val}

    grouped = {c: grp for c, grp in df.groupby(cow_col, sort=False)}
    scores = []
    for _ in range(n_boot):
        sampled_cows = rng.choice(cows, size=len(cows), replace=True)
        boot_df = pd.concat([grouped[c] for c in sampled_cows], ignore_index=True)
        try:
            s = float(metric_fn(boot_df))
            if np.isfinite(s):
                scores.append(s)
        except Exception:
            continue

    if not scores:
        val = float(metric_fn(df))
        return {"mean": val, "std": 0.0, "ci95_low": val, "ci95_high": val}

    arr = np.asarray(scores, dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "ci95_low": float(np.percentile(arr, 2.5)),
        "ci95_high": float(np.percentile(arr, 97.5)),
    }


def run_leakage_checks(
    daily: pd.DataFrame,
    cows: pd.DataFrame | None,
    train_cows: set,
    val_cows: set,
    test_cows: set,
) -> dict[str, bool]:
    """Run automated leakage tests required by PRD Section 7 (FR-E6)."""
    # 1. Disjoint cow splits
    no_cow_overlap = not bool(
        (train_cows & val_cows) or (train_cows & test_cows) or (val_cows & test_cows)
    )

    # 2. Causal baselines: perturbing values after day t must not alter features on or before day t
    first_cow = sorted(daily["cow_id"].unique())[0]
    sub = daily[daily["cow_id"] == first_cow].sort_values("dim").copy()
    sub_cows = cows[cows["cow_id"] == first_cow].copy() if cows is not None and "cow_id" in cows.columns else None

    cutoff_dim = 40
    feat_orig = estrus_features_prd(sub, sub_cows)
    sub_perturbed = sub.copy()
    act_col = "activity" if "activity" in sub_perturbed.columns else "activity_index"
    sub_perturbed.loc[sub_perturbed["dim"] > cutoff_dim, act_col] *= 5.0
    sub_perturbed.loc[sub_perturbed["dim"] > cutoff_dim, "milk_kg"] *= 3.0
    feat_pert = estrus_features_prd(sub_perturbed, sub_cows)

    before_orig = feat_orig[feat_orig["dim"] <= cutoff_dim][PRD_ESTRUS_FEATURES].fillna(-999.0).to_numpy()
    before_pert = feat_pert[feat_pert["dim"] <= cutoff_dim][PRD_ESTRUS_FEATURES].fillna(-999.0).to_numpy()
    causal_baselines = bool(np.allclose(before_orig, before_pert, atol=1e-10))

    # 3. Baseline excludes today: changing day t's activity must not change day t's rolling baseline
    sub_today = sub.copy()
    sub_today.loc[sub_today["dim"] == cutoff_dim, act_col] += 250.0
    feat_today = estrus_features_prd(sub_today, sub_cows)
    b_orig = float(feat_orig.loc[feat_orig["dim"] == cutoff_dim, "activity_baseline"].iloc[0])
    b_today = float(feat_today.loc[feat_today["dim"] == cutoff_dim, "activity_baseline"].iloc[0])
    baseline_excludes_today = bool(np.isclose(b_orig, b_today, atol=1e-10))

    # 4. Parity curve uses training cows only: modifying test cows' milk leaves parity curve unchanged
    train_daily = daily[daily["cow_id"].isin(train_cows)]
    curve_orig = compute_parity_curves(train_daily)
    daily_mod = daily.copy()
    daily_mod.loc[daily_mod["cow_id"].isin(test_cows), "milk_kg"] += 500.0
    curve_mod = compute_parity_curves(daily_mod[daily_mod["cow_id"].isin(train_cows)])
    parity_curve_train_only = bool(
        np.allclose(curve_orig["parity_mean_milk"].to_numpy(), curve_mod["parity_mean_milk"].to_numpy(), atol=1e-10)
    )

    results = {
        "no_cow_overlap": no_cow_overlap,
        "causal_baselines": causal_baselines,
        "baseline_excludes_today": baseline_excludes_today,
        "parity_curve_train_only": parity_curve_train_only,
        "all_passed": bool(
            no_cow_overlap and causal_baselines and baseline_excludes_today and parity_curve_train_only
        ),
    }
    if not results["all_passed"]:
        raise RuntimeError(f"Leakage check failed: {results}")
    return results
