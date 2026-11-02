"""305-day lactation curve forecasting model and baselines (PRD Section 6.4).

Forecasts daily milk yield for DIM 61..305 from early-lactation observations (DIM 5..60),
parity, and % Holstein using:
1. Primary model: Scaled parity-mean curve (training cows only, FR-E6) + HistGradientBoostingRegressor correction.
2. Baselines: Unscaled parity-mean curve and free Wood curve nonlinear least-squares fit on days 5..60.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupKFold

from src.features import compute_parity_curves, build_lactation_features, LACTATION_CORRECTION_FEATURES
from src.evaluate import cow_bootstrap_ci

REQUIRED_COLUMNS = {"cow_id", "dim", "parity", "milk_kg"}


def wood_fn(dim: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    return a * np.power(dim, b) * np.exp(-c * dim)


def new_correction_model(seed: int = 7, hgb_cfg: dict | None = None) -> HistGradientBoostingRegressor:
    cfg = hgb_cfg or {}
    return HistGradientBoostingRegressor(
        max_depth=int(cfg.get("max_depth", 4)),
        learning_rate=float(cfg.get("learning_rate", 0.05)),
        max_iter=int(cfg.get("max_iter", 200)),
        l2_regularization=float(cfg.get("l2_regularization", 2.0)),
        random_state=int(cfg.get("random_state", seed)),
    )


def _fit_wood_baseline_per_cow(
    daily_df: pd.DataFrame,
    cow_ids: set,
    early_start: int = 5,
    early_end: int = 60,
    pred_start: int = 61,
    pred_end: int = 305,
) -> pd.DataFrame:
    """Fit free Wood curve y = a * t^b * exp(-c * t) on days 5..60 per cow (PRD Section 6.4 baseline)."""
    sub = daily_df[daily_df["cow_id"].isin(cow_ids)].sort_values(["cow_id", "dim"])
    early = sub[(sub["dim"] >= early_start) & (sub["dim"] <= early_end)]
    pred_rows = sub[(sub["dim"] >= pred_start) & (sub["dim"] <= pred_end)][["cow_id", "dim", "milk_kg"]].copy()

    params_by_cow: dict = {}
    bounds = ([5.0, 0.02, 0.0], [60.0, 0.60, 0.01])
    p0 = [20.0, 0.20, 0.003]

    for cow_id, grp in early.groupby("cow_id", sort=False):
        t = grp["dim"].to_numpy(dtype=float)
        y = grp["milk_kg"].to_numpy(dtype=float)
        try:
            popt, _ = curve_fit(wood_fn, t, y, p0=p0, bounds=bounds, maxfev=2000)
            params_by_cow[cow_id] = popt
        except Exception:
            params_by_cow[cow_id] = ( float(np.mean(y)), 0.05, 0.002 )

    preds = []
    for cow_id, grp in pred_rows.groupby("cow_id", sort=False):
        a, b, c = params_by_cow[cow_id]
        t = grp["dim"].to_numpy(dtype=float)
        g = grp.copy()
        g["wood_pred"] = np.clip(wood_fn(t, a, b, c), 1.0, 100.0)
        preds.append(g)
    return pd.concat(preds, ignore_index=True) if preds else pred_rows.assign(wood_pred=30.0)


def _score_lactation_predictions(df: pd.DataFrame, pred_col: str, actual_col: str = "milk_kg") -> dict[str, float]:
    y_true = df[actual_col].to_numpy(dtype=float)
    y_pred = df[pred_col].to_numpy(dtype=float)
    mae = float(mean_absolute_error(y_true, y_pred))
    mape = float(np.mean(np.abs(y_true - y_pred) / np.maximum(y_true, 1e-6)) * 100.0)

    by_cow = df.groupby("cow_id", sort=False)[[actual_col, pred_col]].sum()
    rem_err = np.abs(by_cow[pred_col] - by_cow[actual_col]) / np.maximum(by_cow[actual_col], 1e-6) * 100.0
    return {
        "mae_kg": mae,
        "mape_pct": mape,
        "remaining_yield_error_pct": float(np.mean(rem_err)),
    }


def fit(
    daily: pd.DataFrame,
    cows: pd.DataFrame | None = None,
    train_cows: set | None = None,
    val_cows: set | None = None,
    test_cows: set | None = None,
    seed: int = 7,
    hgb_cfg: dict | None = None,
) -> tuple[dict, dict]:
    """Train and evaluate the PRD Section 6.4 Lactation forecasting model and baselines."""
    missing = sorted(REQUIRED_COLUMNS - set(daily.columns))
    if missing:
        raise ValueError(
            "Cannot train lactation model: daily milk-yield records are unavailable; "
            f"missing required columns: {', '.join(missing)}"
        )

    df = daily.sort_values(["cow_id", "dim"]).copy()
    if "pct_holstein" not in df.columns:
        if cows is not None and "pct_holstein" in cows.columns:
            df = df.merge(cows[["cow_id", "pct_holstein"]], on="cow_id", how="left")
        else:
            df["pct_holstein"] = 0.88

    all_cows = sorted(df["cow_id"].unique())
    if not train_cows or not test_cows:
        n_tr = int(0.6 * len(all_cows))
        n_va = int(0.2 * len(all_cows))
        train_cows = set(all_cows[:n_tr])
        val_cows = set(all_cows[n_tr : n_tr + n_va])
        test_cows = set(all_cows[n_tr + n_va :])

    # FR-E6: Build parity-mean curve strictly from training cows only
    tr_daily = df[df["cow_id"].isin(train_cows)]
    parity_curves = compute_parity_curves(tr_daily)

    tr_pred_df, _ = build_lactation_features(tr_daily, cows, parity_curves)
    va_pred_df, _ = build_lactation_features(df[df["cow_id"].isin(val_cows)], cows, parity_curves)
    te_pred_df, te_summary = build_lactation_features(df[df["cow_id"].isin(test_cows)], cows, parity_curves)

    # Fit correction model on training cows
    corr_model = new_correction_model(seed=seed, hgb_cfg=hgb_cfg)
    corr_model.fit(
        tr_pred_df[LACTATION_CORRECTION_FEATURES],
        tr_pred_df["correction_target"].to_numpy(dtype=float),
    )

    te_corr = corr_model.predict(te_pred_df[LACTATION_CORRECTION_FEATURES])
    te_pred_df["milk_kg_pred"] = te_pred_df["scaled_forecast"] * te_corr

    primary_metrics = _score_lactation_predictions(te_pred_df, "milk_kg_pred")
    scaled_only_metrics = _score_lactation_predictions(te_pred_df, "scaled_forecast")
    parity_curve_metrics = _score_lactation_predictions(te_pred_df, "parity_mean_milk")

    wood_df = _fit_wood_baseline_per_cow(df, test_cows)
    wood_metrics = _score_lactation_predictions(wood_df, "wood_pred")

    mae_reduction_pct = float(
        100.0 * (parity_curve_metrics["mae_kg"] - primary_metrics["mae_kg"]) / max(parity_curve_metrics["mae_kg"], 1e-6)
    )

    # 5-fold GroupKFold by cow (FR-E1, FR-E4)
    gkf = GroupKFold(n_splits=min(5, len(all_cows)))
    cow_arr = np.asarray(all_cows)
    cv_maes, cv_rem_errs = [], []
    for f_tr_idx, f_te_idx in gkf.split(cow_arr, groups=cow_arr):
        f_tr_cows = set(cow_arr[f_tr_idx])
        f_te_cows = set(cow_arr[f_te_idx])
        f_curves = compute_parity_curves(df[df["cow_id"].isin(f_tr_cows)])
        f_tr_df, _ = build_lactation_features(df[df["cow_id"].isin(f_tr_cows)], cows, f_curves)
        f_te_df, _ = build_lactation_features(df[df["cow_id"].isin(f_te_cows)], cows, f_curves)
        m_cv = new_correction_model(seed=seed, hgb_cfg=hgb_cfg)
        m_cv.fit(f_tr_df[LACTATION_CORRECTION_FEATURES], f_tr_df["correction_target"].to_numpy(dtype=float))
        f_te_df["milk_kg_pred"] = f_te_df["scaled_forecast"] * m_cv.predict(f_te_df[LACTATION_CORRECTION_FEATURES])
        sc = _score_lactation_predictions(f_te_df, "milk_kg_pred")
        cv_maes.append(sc["mae_kg"])
        cv_rem_errs.append(sc["remaining_yield_error_pct"])

    # Cow-level bootstrap 95% CI on test cows (FR-E4)
    boot_mae = cow_bootstrap_ci(
        te_pred_df,
        lambda d: mean_absolute_error(d["milk_kg"], d["milk_kg_pred"]),
        n_boot=200,
        seed=seed,
    )
    boot_rem = cow_bootstrap_ci(
        te_pred_df,
        lambda d: _score_lactation_predictions(d, "milk_kg_pred")["remaining_yield_error_pct"],
        n_boot=200,
        seed=seed,
    )

    metrics = {
        "status": "trained",
        "source": "synthetic PRD 300-cow daily.csv + cows.csv",
        "n_cows": int(len(all_cows)),
        "test_cows": int(len(test_cows)),
        "test_rows_dim_61_305": int(len(te_pred_df)),
        **primary_metrics,
        "mae_reduction_vs_parity_curve_pct": mae_reduction_pct,
        "cv_5fold": {
            "mae_kg_mean": float(np.mean(cv_maes)),
            "mae_kg_std": float(np.std(cv_maes)),
            "remaining_yield_error_pct_mean": float(np.mean(cv_rem_errs)),
            "remaining_yield_error_pct_std": float(np.std(cv_rem_errs)),
        },
        "bootstrap_95ci": {
            "mae_kg": boot_mae,
            "remaining_yield_error_pct": boot_rem,
        },
        "baselines": {
            "parity_mean_curve": parity_curve_metrics,
            "free_wood_fit_5_60": wood_metrics,
            "scaled_parity_curve_uncorrected": scaled_only_metrics,
        },
        "acceptance": {
            "remaining_yield_error_target_pct": 5.0,
            "mae_reduction_target_pct": 30.0,
            "acceptance_met": bool(
                primary_metrics["remaining_yield_error_pct"] <= 5.0 and mae_reduction_pct >= 30.0
            ),
        },
        "is_synthetic": 1,
    }

    bundle = {
        "model": corr_model,
        "parity_curves": parity_curves,
        "features": LACTATION_CORRECTION_FEATURES,
        "model_version": "lactation-scaled_parity_hgb-1.0.0",
        "training_scope": "synthetic-300cow-prd",
    }
    return bundle, metrics


def predict_lactation(daily: pd.DataFrame, cows: pd.DataFrame | None, bundle: dict) -> pd.DataFrame:
    """Predict daily milk for DIM 61..305 and total 305-day lactation yield per cow (FR-P3)."""
    parity_curves = bundle["parity_curves"]
    feat_cols = bundle["features"]
    pred_df, cow_summary = build_lactation_features(daily, cows, parity_curves)
    corr = bundle["model"].predict(pred_df[feat_cols])
    pred_df["milk_kg_pred"] = np.round(pred_df["scaled_forecast"] * corr, 3)

    rem_totals = pred_df.groupby("cow_id")["milk_kg_pred"].sum()
    early_totals = cow_summary.set_index("cow_id")["observed_1_to_60_kg"]
    total_305 = (early_totals + rem_totals).round(2)

    pred_df["total_305_pred"] = pred_df["cow_id"].map(total_305)
    pred_df["model_version"] = bundle["model_version"]
    pred_df["is_synthetic"] = 1
    return pred_df[["cow_id", "dim", "milk_kg_pred", "total_305_pred", "model_version", "is_synthetic"]]
