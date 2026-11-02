"""Causal feature generation for Estrus, Conception, and Lactation models (PRD Section 5).

Supports both:
1. The PRD 300-cow schema (daily.csv, cows.csv, inseminations.csv) with 14-day causal baselines (FR-F1 to FR-F4).
2. The 10-cow synthetic SQLite & CSV schema (synthetic_daily_10cows.csv, synthetic_hourly_10cows.csv,
   synthetic_cattle.db, and out/estrus_predictions.csv) with 7-day causal baselines and hourly aggregations.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# 10-cow demo signals (core + walking/posture fraction signals)
SIGNALS_10COW = ["activity_index", "rumination_min", "distance_m", "head_temp_c"]
WALKING_POSTURE_SIGNALS_10COW = ["walking_frac", "lying_frac"]

# PRD v2 Real Estrus Biosignals (Electronic Ear-Tag Skin Temperature + Steps)
SIGNALS = ["steps_sum", "temp_mean", "temp_q90"]
MIN_SCALE = {"steps_sum": 1.0, "temp_mean": 0.05, "temp_q90": 0.05}

FEATURES = [c for s in SIGNALS for c in (f"{s}_ratio", f"{s}_z", f"{s}_z_lag1")] + [
    "steps_z_max3",
    "days_since_spike",
]


def assert_real_only(df: pd.DataFrame) -> None:
    """Enforce PRD v2 FR-A3: reject any synthetic data rows from reaching real-data evaluation."""
    if "is_synthetic" in df.columns:
        synth = (df["is_synthetic"] == 1).sum()
        assert synth == 0, f"DATA INTEGRITY ERROR: Found {synth} synthetic rows in real evaluation table!"
    if "source" in df.columns:
        bad_sources = set(df["source"].unique()) & {"simulated_herd", "fixture", "synthetic"}
        assert not bad_sources, f"DATA INTEGRITY ERROR: Found synthetic sources: {bad_sources}"


def to_daily_grid(df: pd.DataFrame) -> pd.DataFrame:
    """Reindex each animal's time series to a complete daily calendar grid so gaps do not distort baselines."""
    parts = []
    for a, g in df.groupby("animal_id", sort=False):
        d_min = pd.to_datetime(g["date"]).min()
        d_max = pd.to_datetime(g["date"]).max()
        full = pd.date_range(d_min, d_max, freq="D")
        sub = g.copy()
        sub["date"] = pd.to_datetime(sub["date"])
        reindexed = sub.set_index("date").reindex(full).rename_axis("date").assign(animal_id=a).reset_index()
        parts.append(reindexed)
    out = pd.concat(parts, ignore_index=True)
    if "estrus" in out.columns:
        out["estrus"] = out["estrus"].fillna(0).astype(int)
    return out


def _mad(w: np.ndarray) -> float:
    """Median Absolute Deviation."""
    med = np.median(w)
    return float(np.median(np.abs(w - med)))


def _days_since(flag: pd.Series) -> pd.Series:
    """Compute elapsed days since previous positive boolean flag."""
    out, last = [], None
    for i, f in enumerate(flag.to_numpy()):
        out.append(np.nan if last is None else float(i - last))
        if f:
            last = i
    return pd.Series(out, index=flag.index, dtype=float)


def add_features(
    df: pd.DataFrame,
    window: int = 14,
    min_periods: int = 5,
    alert_ratio: float = 1.5,
) -> pd.DataFrame:
    """PRD v2 strictly causal feature pipeline: rolling medians and MAD scaling ending at day t-1."""
    grid_df = to_daily_grid(df).sort_values(["animal_id", "date"]).reset_index(drop=True)
    g = grid_df.groupby("animal_id", sort=False)
    for s in SIGNALS:
        if s not in grid_df.columns:
            continue
        past = g[s].transform(lambda x: x.shift(1).rolling(window, min_periods=min_periods).median())
        mad = g[s].transform(
            lambda x: x.shift(1).rolling(window, min_periods=min_periods).apply(_mad, raw=True)
        )
        scale = np.maximum(1.4826 * mad, MIN_SCALE[s])
        grid_df[f"{s}_ratio"] = grid_df[s] / past.replace(0, np.nan)
        grid_df[f"{s}_z"] = (grid_df[s] - past) / scale
        grid_df[f"{s}_z_lag1"] = g[f"{s}_z"].shift(1)

    if "steps_sum_z" in grid_df.columns:
        grid_df["steps_z_max3"] = g["steps_sum_z"].transform(lambda x: x.rolling(3, min_periods=1).max())
    else:
        grid_df["steps_z_max3"] = 0.0

    if "steps_sum_ratio" in grid_df.columns:
        spike = (grid_df["steps_sum_ratio"] > alert_ratio).fillna(False)
        grid_df["days_since_spike"] = spike.groupby(grid_df["animal_id"]).transform(_days_since)
    else:
        grid_df["days_since_spike"] = np.nan

    return grid_df

# PRD 300-cow signals (5 signals -> 15 ratio/z/lag features + 5 context/rolling features = 20 features)
SIGNALS_PRD = ["activity", "rumination_min", "lying_min", "body_temp_c", "milk_kg"]

PRD_ESTRUS_FEATURES = [
    "activity_ratio",
    "rumination_min_ratio",
    "lying_min_ratio",
    "body_temp_c_ratio",
    "milk_kg_ratio",
    "activity_z",
    "rumination_min_z",
    "lying_min_z",
    "body_temp_c_z",
    "milk_kg_z",
    "activity_z_lag1",
    "rumination_min_z_lag1",
    "lying_min_z_lag1",
    "body_temp_c_z_lag1",
    "milk_kg_z_lag1",
    "act_z_roll3max",
    "days_since_alert",
    "dim",
    "parity",
    "pct_holstein",
]

DEMO_ESTRUS_FEATURES = [
    "activity_index_ratio",
    "rumination_min_ratio",
    "distance_m_ratio",
    "head_temp_c_ratio",
    "activity_index_z",
    "rumination_min_z",
    "distance_m_z",
    "head_temp_c_z",
    "activity_index_z_lag1",
    "rumination_min_z_lag1",
    "distance_m_z_lag1",
    "head_temp_c_z_lag1",
    "act_z_roll3max",
    "days_since_alert",
    "dim",
    "parity",
]

WALKING_PATTERN_FEATURES = [
    "walking_frac_ratio",
    "walking_frac_z",
    "walking_frac_z_lag1",
    "lying_frac_ratio",
    "lying_frac_z",
    "walking_to_lying_ratio",
]

HOURLY_ENRICHED_FEATURES = DEMO_ESTRUS_FEATURES + WALKING_PATTERN_FEATURES + [
    "hourly_act_max",
    "hourly_act_std",
    "hourly_mounting_sum",
    "hourly_proximity_max",
    "hourly_walking_max",
    "hourly_walking_to_lying_max",
    "hourly_grazing_min",
    "hourly_lying_min",
    "hourly_temp_max",
]

LACTATION_CORRECTION_FEATURES = [
    "parity",
    "pct_holstein",
    "early_mean",
    "early_slope",
    "scale",
    "dim",
    "scaled_forecast",
]


def _compute_days_since_alert(df: pd.DataFrame, ratio_col: str, threshold: float = 1.5) -> pd.Series:
    """Causally compute days since the last earlier day (< dim) with activity ratio > threshold."""
    result = pd.Series(np.nan, index=df.index, dtype=float)
    for _, indices in df.groupby("cow_id", sort=False).groups.items():
        last_alert = None
        vals = df.loc[indices, ratio_col].to_numpy()
        dims = df.loc[indices, "dim"].to_numpy()
        elapsed = []
        for dim, ratio in zip(dims, vals):
            elapsed.append(np.nan if last_alert is None else float(dim - last_alert))
            if np.isfinite(ratio) and ratio > threshold:
                last_alert = int(dim)
        result.loc[indices] = elapsed
    return result


def estrus_features_prd(
    daily: pd.DataFrame,
    cows: pd.DataFrame | None = None,
    window: int = 14,
    min_periods: int = 5,
    warmup_dim: int = 15,
    alert_ratio_threshold: float = 1.5,
) -> pd.DataFrame:
    """Compute the 20 causal estrus features defined in PRD Section 5 (FR-F1 to FR-F4)."""
    out = daily.sort_values(["cow_id", "dim"]).copy()
    if "pct_holstein" not in out.columns:
        if cows is not None and "pct_holstein" in cows.columns:
            out = out.merge(cows[["cow_id", "pct_holstein"]], on="cow_id", how="left")
        else:
            out["pct_holstein"] = 0.88

    grouped = out.groupby("cow_id", sort=False)
    for signal in SIGNALS_PRD:
        # FR-F1: Causal baseline is median of days t-window to t-1, min_periods=5. Today is never inside.
        baseline = grouped[signal].transform(
            lambda s: s.shift(1).rolling(window, min_periods=min_periods).median()
        )
        spread = (
            grouped[signal]
            .transform(lambda s: s.shift(1).rolling(window, min_periods=min_periods).std())
            .replace(0, np.nan)
        )
        out[f"{signal}_baseline"] = baseline
        out[f"{signal}_ratio"] = out[signal] / baseline
        out[f"{signal}_z"] = (out[signal] - baseline) / spread
        out[f"{signal}_z_lag1"] = grouped[f"{signal}_z"].shift(1)

    out["act_z_roll3max"] = out.groupby("cow_id", sort=False)["activity_z"].transform(
        lambda s: s.rolling(3, min_periods=1).max()
    )
    out["days_since_alert"] = _compute_days_since_alert(
        out, ratio_col="activity_ratio", threshold=alert_ratio_threshold
    )
    out["dim"] = out["dim"].astype(int)
    out["parity"] = out["parity"].astype(int)
    out["pct_holstein"] = out["pct_holstein"].astype(float)

    # FR-F2: Drop warm-up rows with DIM below 15
    return out.loc[out["dim"] >= warmup_dim].reset_index(drop=True)


def compute_collar_walking_features(imu_clips_df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-cow walking gait & behaviour distribution features from collar IMU clips."""
    if imu_clips_df is None or len(imu_clips_df) == 0 or "cow_id" not in imu_clips_df.columns:
        return pd.DataFrame()
    rows = []
    for cow_id, grp in imu_clips_df.groupby("cow_id", sort=False):
        n_total = max(1, len(grp))
        walk_grp = grp[grp["label"] == "Walking"]
        rows.append(
            {
                "cow_id": cow_id,
                "collar_walking_clip_frac": float((grp["label"] == "Walking").sum() / n_total),
                "collar_grazing_clip_frac": float((grp["label"] == "Grazing").sum() / n_total),
                "collar_resting_clip_frac": float((grp["label"] == "Resting").sum() / n_total),
                "collar_walk_acc_mag_mean": float(walk_grp["acc_mag_mean"].mean())
                if len(walk_grp) and "acc_mag_mean" in walk_grp
                else float(grp["acc_mag_mean"].mean()),
                "collar_walk_dom_freq_hz": float(walk_grp["dom_freq_hz"].mean())
                if len(walk_grp) and "dom_freq_hz" in walk_grp
                else float(grp["dom_freq_hz"].mean()),
                "collar_walk_gyro_std": float(walk_grp["gyro_std"].mean())
                if len(walk_grp) and "gyro_std" in walk_grp
                else float(grp["gyro_std"].mean()),
            }
        )
    return pd.DataFrame(rows)


def aggregate_hourly_features(hourly: pd.DataFrame) -> pd.DataFrame:
    """Aggregate synthetic_hourly_10cows.csv into daily peak/variability and walking-pattern features."""
    h_df = hourly.sort_values(["cow_id", "dim", "hour"]).copy()
    h_df["walking_to_lying"] = h_df["walking_frac"] / np.maximum(h_df["lying_frac"], 0.05)
    g = h_df.groupby(["cow_id", "dim"], as_index=False)
    agg = g.agg(
        hourly_act_max=("activity_index", "max"),
        hourly_act_std=("activity_index", "std"),
        hourly_mounting_sum=("mounting_events", "sum"),
        hourly_proximity_max=("neighbour_proximity", "max"),
        hourly_walking_max=("walking_frac", "max"),
        hourly_walking_to_lying_max=("walking_to_lying", "max"),
        hourly_grazing_min=("grazing_frac", "min"),
        hourly_lying_min=("lying_frac", "min"),
        hourly_temp_max=("head_temp_c", "max"),
        estrus_duration_h_obs=("heat", "sum"),
        days_to_next_heat_min=("days_to_next_heat", "min"),
    )
    return agg


def estrus_features(
    daily: pd.DataFrame,
    window: int = 7,
    hourly: pd.DataFrame | None = None,
    predictions_df: pd.DataFrame | None = None,
    imu_clips_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Causal feature generator supporting both the 10-cow schema and PRD 300-cow schema."""
    if set(SIGNALS_PRD).issubset(daily.columns) and "activity_index" not in daily.columns:
        return estrus_features_prd(daily, window=window if window >= 10 else 14)

    out = daily.sort_values(["cow_id", "dim"]).copy()
    if "parity" not in out.columns and hourly is not None and "parity" in hourly.columns:
        parity_map = hourly.groupby("cow_id")["parity"].first()
        out["parity"] = out["cow_id"].map(parity_map)
    elif "parity" not in out.columns:
        out["parity"] = 2

    grouped = out.groupby("cow_id", sort=False)
    all_signals = SIGNALS_10COW + [s for s in WALKING_POSTURE_SIGNALS_10COW if s in out.columns]
    for signal in all_signals:
        baseline = grouped[signal].transform(
            lambda s: s.shift(1).rolling(window, min_periods=3).median()
        )
        spread = (
            grouped[signal]
            .transform(lambda s: s.shift(1).rolling(window, min_periods=3).std())
            .replace(0, np.nan)
        )
        out[f"{signal}_baseline"] = baseline
        out[f"{signal}_ratio"] = out[signal] / baseline
        out[f"{signal}_z"] = (out[signal] - baseline) / spread
        out[f"{signal}_z_lag1"] = grouped[f"{signal}_z"].shift(1)

    if {"walking_frac", "lying_frac"}.issubset(out.columns):
        out["walking_to_lying_ratio"] = out["walking_frac"] / np.maximum(out["lying_frac"], 0.05)

    out["act_z_roll3max"] = out.groupby("cow_id", sort=False)["activity_index_z"].transform(
        lambda s: s.rolling(3, min_periods=1).max()
    )
    out["days_since_alert"] = _compute_days_since_alert(
        out, ratio_col="activity_index_ratio", threshold=1.5
    )
    out["parity"] = out["cow_id"].map(out.groupby("cow_id")["parity"].first()).fillna(2).astype(int)
    out["dim"] = out["dim"].astype(int)

    if hourly is not None:
        h_agg = aggregate_hourly_features(hourly)
        out = out.merge(h_agg, on=["cow_id", "dim"], how="left")

    if imu_clips_df is not None:
        collar_walk_df = compute_collar_walking_features(imu_clips_df)
        if len(collar_walk_df) > 0:
            out = out.merge(collar_walk_df, on="cow_id", how="left")

    if predictions_df is not None and {"cow_id", "dim", "p_estrus"}.issubset(predictions_df.columns):
        pred_sub = predictions_df[["cow_id", "dim", "p_estrus", "alert"]].copy()
        pred_sub["cow_id"] = pred_sub["cow_id"].astype(out["cow_id"].dtype, errors="ignore")
        if pred_sub["cow_id"].dtype == out["cow_id"].dtype:
            pred_sub = pred_sub.rename(
                columns={"p_estrus": "prev_stage_p_estrus", "alert": "prev_stage_alert"}
            )
            out = out.merge(pred_sub, on=["cow_id", "dim"], how="left")
            # Causal lagged prediction features from the prediction CSV
            out["prev_stage_p_estrus_lag1"] = out.groupby("cow_id", sort=False)["prev_stage_p_estrus"].shift(1)

    return out.loc[out["dim"] >= 15].reset_index(drop=True)


def build_conception_features(
    inseminations: pd.DataFrame,
    daily_features: pd.DataFrame | None = None,
    estrus_predictions: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build the 8 conception features per insemination (PRD Section 5, FR-F4).

    FR-F4: Heat-day strengths used by the conception model are computed from the same
    causal baseline ratios as the estrus features when daily_features is supplied.
    """
    df = inseminations.sort_values(["cow_id", "dim"]).copy()

    if daily_features is not None:
        act_col = "activity_ratio" if "activity_ratio" in daily_features.columns else "activity_index_ratio"
        rum_col = "rumination_min_ratio"
        if {act_col, rum_col}.issubset(daily_features.columns):
            ratios = daily_features[["cow_id", "dim", act_col, rum_col]].drop_duplicates(["cow_id", "dim"])
            df = df.merge(ratios, on=["cow_id", "dim"], how="left")
            if "estrus_activity_strength" not in df.columns:
                df["estrus_activity_strength"] = df[act_col]
            else:
                df["estrus_activity_strength"] = df["estrus_activity_strength"].combine_first(df[act_col])
            if "estrus_rumination_strength" not in df.columns:
                df["estrus_rumination_strength"] = df[rum_col]
            else:
                df["estrus_rumination_strength"] = df["estrus_rumination_strength"].combine_first(df[rum_col])
            df["estrus_activity_strength_causal"] = df[act_col].combine_first(df["estrus_activity_strength"])
            df["estrus_rumination_strength_causal"] = df[rum_col].combine_first(df["estrus_rumination_strength"])

    if estrus_predictions is not None and {"cow_id", "dim", "p_estrus"}.issubset(estrus_predictions.columns):
        preds = estrus_predictions[["cow_id", "dim", "p_estrus"]].drop_duplicates(["cow_id", "dim"])
        df = df.merge(preds, on=["cow_id", "dim"], how="left")
        df["p_estrus_feature"] = df["p_estrus"].fillna(0.5)

    return df


def compute_parity_curves(
    train_daily: pd.DataFrame,
    max_dim: int = 305,
) -> pd.DataFrame:
    """Compute mean milk by parity and DIM from training cows only (PRD Section 5 & FR-E6)."""
    curve = (
        train_daily.groupby(["parity", "dim"], as_index=False)["milk_kg"]
        .mean()
        .rename(columns={"milk_kg": "parity_mean_milk"})
    )
    # Ensure complete grid for parities 1, 2, 3 and DIM 1..max_dim
    grid = pd.MultiIndex.from_product([[1, 2, 3], range(1, max_dim + 1)], names=["parity", "dim"]).to_frame().reset_index(drop=True)
    merged = grid.merge(curve, on=["parity", "dim"], how="left")
    merged["parity_mean_milk"] = merged.groupby("parity")["parity_mean_milk"].transform(
        lambda s: s.interpolate(limit_direction="both")
    )
    overall_mean = train_daily.groupby("dim")["milk_kg"].mean()
    merged["parity_mean_milk"] = merged["parity_mean_milk"].fillna(merged["dim"].map(overall_mean)).fillna(30.0)
    return merged


def build_lactation_features(
    daily: pd.DataFrame,
    cows: pd.DataFrame | None,
    parity_curves: pd.DataFrame,
    early_start: int = 5,
    early_end: int = 60,
    pred_start: int = 61,
    pred_end: int = 305,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute early-window summaries (days 5..60) and prediction rows (days 61..305) per PRD Section 5."""
    df = daily.sort_values(["cow_id", "dim"]).copy()
    if "pct_holstein" not in df.columns:
        if cows is not None and "pct_holstein" in cows.columns:
            df = df.merge(cows[["cow_id", "pct_holstein"]], on="cow_id", how="left")
        else:
            df["pct_holstein"] = 0.88

    df = df.merge(parity_curves, on=["parity", "dim"], how="left")

    early = df[(df["dim"] >= early_start) & (df["dim"] <= early_end)].copy()
    first14 = early[early["dim"] <= early_start + 13].groupby("cow_id")["milk_kg"].mean()
    last14 = early[early["dim"] >= early_end - 13].groupby("cow_id")["milk_kg"].mean()
    early_mean = early.groupby("cow_id")["milk_kg"].mean()
    early_total = df[df["dim"] <= early_end].groupby("cow_id")["milk_kg"].sum()

    # Scale s = sum(y * f) / sum(f^2) over days 5 to 60
    early["yf"] = early["milk_kg"] * early["parity_mean_milk"]
    early["f2"] = early["parity_mean_milk"] ** 2
    sums = early.groupby("cow_id")[["yf", "f2", "parity", "pct_holstein"]].agg(
        {"yf": "sum", "f2": "sum", "parity": "first", "pct_holstein": "first"}
    )
    sums["scale"] = sums["yf"] / sums["f2"].replace(0, np.nan)
    sums["scale"] = sums["scale"].fillna(1.0)
    sums["early_mean"] = sums.index.map(early_mean)
    f14 = sums.index.map(first14)
    l14 = sums.index.map(last14)
    sums["early_slope"] = (l14 / pd.Series(f14, index=sums.index).replace(0, np.nan)).fillna(1.0)
    sums["observed_1_to_60_kg"] = sums.index.map(early_total).fillna(0.0)
    cow_summary = sums.reset_index()[
        ["cow_id", "parity", "pct_holstein", "scale", "early_mean", "early_slope", "observed_1_to_60_kg"]
    ]

    pred_df = df[(df["dim"] >= pred_start) & (df["dim"] <= pred_end)][
        ["cow_id", "dim", "parity", "milk_kg", "parity_mean_milk"]
    ].copy()
    pred_df = pred_df.merge(
        cow_summary[["cow_id", "pct_holstein", "scale", "early_mean", "early_slope", "observed_1_to_60_kg"]],
        on="cow_id",
        how="inner",
    )
    pred_df["scaled_forecast"] = pred_df["scale"] * pred_df["parity_mean_milk"]
    pred_df["correction_target"] = pred_df["milk_kg"] / pred_df["scaled_forecast"].replace(0, np.nan)
    return pred_df.reset_index(drop=True), cow_summary
