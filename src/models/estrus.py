"""Per-cow-day estrus detector and baselines (PRD Section 6.2).

Trains and evaluates:
1. Primary model: HistGradientBoostingClassifier (depth 5, lr 0.06, 300 iters, L2 1.0, early stopping 15%, seed 7).
2. Baselines: Balanced Logistic Regression and tuned Activity-Ratio Rule.
3. Supports both the 300-cow PRD dataset (20 features) and the 10-cow dataset (daily + hourly + prediction CSVs).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, mean_absolute_error
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.features import PRD_ESTRUS_FEATURES, DEMO_ESTRUS_FEATURES, HOURLY_ENRICHED_FEATURES
from src.evaluate import cow_bootstrap_ci

# Backward-compatible export for 10-cow feature list
FEATURES = DEMO_ESTRUS_FEATURES


def new_model(seed: int = 7, hgb_cfg: dict | None = None):
    """Create the PRD Section 6.2 HistGradientBoostingClassifier."""
    cfg = hgb_cfg or {}
    return HistGradientBoostingClassifier(
        max_depth=int(cfg.get("max_depth", 5)),
        learning_rate=float(cfg.get("learning_rate", 0.06)),
        max_iter=int(cfg.get("max_iter", 300)),
        l2_regularization=float(cfg.get("l2_regularization", 1.0)),
        validation_fraction=float(cfg.get("validation_fraction", 0.15)),
        early_stopping=bool(cfg.get("early_stopping", True)),
        random_state=int(cfg.get("random_state", seed)),
    )


def new_logistic_baseline(seed: int = 7):
    """Create the PRD Section 6.2 balanced logistic regression baseline (FR-F3: median imputation + scaling)."""
    return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(class_weight="balanced", max_iter=500, random_state=seed),
    )


def choose_threshold(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    grid_start: float = 0.05,
    grid_stop: float = 0.96,
    grid_step: float = 0.01,
) -> float:
    thresholds = np.arange(grid_start, grid_stop, grid_step)
    scores = [f1_score(y_true, probabilities >= t, zero_division=0) for t in thresholds]
    return float(np.round(thresholds[int(np.argmax(scores))], 4))


def choose_rule_threshold(
    y_true: np.ndarray,
    activity_ratio: np.ndarray,
    grid_start: float = 1.10,
    grid_stop: float = 2.00,
    grid_step: float = 0.01,
) -> float:
    ratio = np.nan_to_num(activity_ratio, nan=0.0)
    thresholds = np.arange(grid_start, grid_stop, grid_step)
    scores = [f1_score(y_true, ratio >= t, zero_division=0) for t in thresholds]
    return float(np.round(thresholds[int(np.argmax(scores))], 4))


def score(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    estrus_duration_h: np.ndarray | None = None,
) -> dict[str, float]:
    y_true_arr = np.asarray(y_true, dtype=int)
    prob_arr = np.asarray(probabilities, dtype=float)
    predicted = prob_arr >= threshold
    false_positives = int(np.sum((predicted == 1) & (y_true_arr == 0)))
    n_cow_days = max(1, len(y_true_arr))
    out = {
        "pr_auc": float(average_precision_score(y_true_arr, prob_arr)),
        "precision": float(precision_score(y_true_arr, predicted, zero_division=0)),
        "recall": float(recall_score(y_true_arr, predicted, zero_division=0)),
        "f1": float(f1_score(y_true_arr, predicted, zero_division=0)),
        "false_alerts_per_100_cow_days": float(100.0 * false_positives / n_cow_days),
    }
    # FR-E5: Report recall for silent heats separately from normal heats
    if estrus_duration_h is not None:
        dur = np.asarray(estrus_duration_h, dtype=float)
        # In the simulator, silent heats have shrunk duration/deviation; identify silent vs normal
        silent_mask = (y_true_arr == 1) & (dur > 0) & (dur <= 7.5)
        normal_mask = (y_true_arr == 1) & (~silent_mask)
        if np.any(normal_mask):
            out["normal_heat_recall"] = float(np.mean(predicted[normal_mask]))
        if np.any(silent_mask):
            out["silent_heat_recall"] = float(np.mean(predicted[silent_mask]))
    return out


def fit_and_evaluate_prd(
    features_df: pd.DataFrame,
    train_cows: set,
    val_cows: set,
    test_cows: set,
    inseminations_df: pd.DataFrame | None = None,
    seed: int = 7,
    hgb_cfg: dict | None = None,
) -> tuple[dict, dict, pd.DataFrame]:
    """Train and evaluate the PRD Section 6.2 Estrus Detector, baselines, and 5-fold GroupKFold."""
    label_col = "estrus_day" if "estrus_day" in features_df.columns else "heat"
    feat_cols = PRD_ESTRUS_FEATURES if set(PRD_ESTRUS_FEATURES).issubset(features_df.columns) else DEMO_ESTRUS_FEATURES
    ratio_col = "activity_ratio" if "activity_ratio" in features_df.columns else "activity_index_ratio"

    tr = features_df[features_df["cow_id"].isin(train_cows)].copy()
    va = features_df[features_df["cow_id"].isin(val_cows)].copy()
    te = features_df[features_df["cow_id"].isin(test_cows)].copy()

    y_tr = tr[label_col].astype(int).to_numpy()
    y_va = va[label_col].astype(int).to_numpy()
    y_te = te[label_col].astype(int).to_numpy()

    # 1. Primary HistGradientBoostingClassifier
    model = new_model(seed=seed, hgb_cfg=hgb_cfg)
    model.fit(tr[feat_cols], y_tr)
    va_prob = model.predict_proba(va[feat_cols])[:, 1]
    threshold = choose_threshold(y_va, va_prob)
    te_prob = model.predict_proba(te[feat_cols])[:, 1]

    dur_te = te["estrus_duration_h"].to_numpy() if "estrus_duration_h" in te.columns else None
    primary_metrics = score(y_te, te_prob, threshold, estrus_duration_h=dur_te)

    # Silent vs normal heat recall on test set using activity_ratio or inseminations silent_heat flag
    if label_col == "estrus_day" and ratio_col in te.columns:
        heat_te = y_te == 1
        silent_te = heat_te & (te[ratio_col].fillna(1.0).to_numpy() < 1.32)
        normal_te = heat_te & (~silent_te)
        pred_te = te_prob >= threshold
        primary_metrics["normal_heat_recall"] = float(np.mean(pred_te[normal_te])) if np.any(normal_te) else 1.0
        primary_metrics["silent_heat_recall"] = float(np.mean(pred_te[silent_te])) if np.any(silent_te) else 0.0

    # 2. Balanced Logistic Regression baseline
    lr_model = new_logistic_baseline(seed=seed)
    lr_model.fit(tr[feat_cols], y_tr)
    lr_va_prob = lr_model.predict_proba(va[feat_cols])[:, 1]
    lr_thresh = choose_threshold(y_va, lr_va_prob)
    lr_te_prob = lr_model.predict_proba(te[feat_cols])[:, 1]
    lr_metrics = score(y_te, lr_te_prob, lr_thresh)
    lr_metrics["validation_threshold"] = lr_thresh

    # 3. Activity Ratio Rule baseline (tuned on validation cows only, FR-E2)
    rule_thresh = choose_rule_threshold(y_va, va[ratio_col].to_numpy())
    rule_te_pred = (te[ratio_col].fillna(0.0).to_numpy() >= rule_thresh).astype(float)
    rule_metrics = {
        "rule_threshold": rule_thresh,
        "precision": float(precision_score(y_te, rule_te_pred >= 0.5, zero_division=0)),
        "recall": float(recall_score(y_te, rule_te_pred >= 0.5, zero_division=0)),
        "f1": float(f1_score(y_te, rule_te_pred >= 0.5, zero_division=0)),
        "false_alerts_per_100_cow_days": float(
            100.0 * np.sum((rule_te_pred >= 0.5) & (y_te == 0)) / max(1, len(y_te))
        ),
    }

    # 4. 5-fold GroupKFold across all cows + Out-of-Fold predictions for downstream models
    gkf = GroupKFold(n_splits=min(5, features_df["cow_id"].nunique()))
    oof_probs = np.zeros(len(features_df), dtype=float)
    cv_f1s, cv_pr_aucs = [], []
    all_X = features_df[feat_cols]
    all_y = features_df[label_col].astype(int).to_numpy()
    all_groups = features_df["cow_id"].to_numpy()

    for f_tr, f_te in gkf.split(all_X, all_y, groups=all_groups):
        m_cv = new_model(seed=seed, hgb_cfg=hgb_cfg)
        m_cv.fit(all_X.iloc[f_tr], all_y[f_tr])
        p_cv = m_cv.predict_proba(all_X.iloc[f_te])[:, 1]
        oof_probs[f_te] = p_cv
        cv_f1s.append(float(f1_score(all_y[f_te], p_cv >= threshold, zero_division=0)))
        cv_pr_aucs.append(float(average_precision_score(all_y[f_te], p_cv)))

    # 5. Cow-level bootstrap 95% CI on test cows (FR-E4)
    te_eval_df = pd.DataFrame({"cow_id": te["cow_id"].to_numpy(), "y": y_te, "p": te_prob})
    f1_boot = cow_bootstrap_ci(
        te_eval_df,
        lambda d: f1_score(d["y"], d["p"] >= threshold, zero_division=0),
        n_boot=200,
        seed=seed,
    )
    pr_auc_boot = cow_bootstrap_ci(
        te_eval_df,
        lambda d: average_precision_score(d["y"], d["p"]),
        n_boot=200,
        seed=seed,
    )

    primary_metrics.update(
        {
            "source": "synthetic PRD 300-cow daily.csv + cows.csv",
            "n_cows": int(features_df["cow_id"].nunique()),
            "validation_threshold": threshold,
            "test_rows": int(len(te)),
            "cv_5fold": {
                "f1_mean": float(np.mean(cv_f1s)),
                "f1_std": float(np.std(cv_f1s)),
                "pr_auc_mean": float(np.mean(cv_pr_aucs)),
                "pr_auc_std": float(np.std(cv_pr_aucs)),
            },
            "bootstrap_95ci": {
                "f1": f1_boot,
                "pr_auc": pr_auc_boot,
            },
            "baselines": {
                "balanced_logistic_regression": lr_metrics,
                "activity_ratio_rule": rule_metrics,
            },
            "acceptance": {
                "f1_target": 0.85,
                "pr_auc_target": 0.90,
                "beats_activity_rule": bool(primary_metrics["f1"] > rule_metrics["f1"]),
                "acceptance_met": bool(
                    primary_metrics["f1"] >= 0.85
                    and primary_metrics["pr_auc"] >= 0.90
                    and primary_metrics["f1"] > rule_metrics["f1"]
                ),
            },
            "is_synthetic": 1,
        }
    )

    oof_df = pd.DataFrame(
        {
            "cow_id": features_df["cow_id"].to_numpy(),
            "dim": features_df["dim"].to_numpy(),
            "p_estrus": oof_probs,
            "alert": (oof_probs >= threshold).astype(int),
            "threshold": threshold,
            "model_version": "estrus-hist_gradient_boosting-1.0.0",
            "is_synthetic": 1,
        }
    )

    bundle = {
        "model": model,
        "logistic_baseline": lr_model,
        "features": feat_cols,
        "threshold": threshold,
        "rule_threshold": rule_thresh,
        "model_version": "estrus-hist_gradient_boosting-1.0.0",
        "training_scope": "synthetic-300cow-prd",
    }
    return bundle, primary_metrics, oof_df


def fit_and_evaluate_10cow_hybrid(
    features_10: pd.DataFrame,
    train_cows: set,
    val_cows: set,
    test_cows: set,
    seed: int = 7,
) -> tuple[dict, dict]:
    """Train and evaluate the 10-cow model using synthetic_daily_10cows.csv,
    synthetic_hourly_10cows.csv, and out/estrus_predictions.csv."""
    base_cols = [c for c in HOURLY_ENRICHED_FEATURES if c in features_10.columns]
    for col in (
        "collar_walking_clip_frac",
        "collar_grazing_clip_frac",
        "collar_resting_clip_frac",
        "collar_walk_acc_mag_mean",
        "collar_walk_dom_freq_hz",
        "collar_walk_gyro_std",
        "prev_stage_p_estrus_lag1",
    ):
        if col in features_10.columns:
            base_cols.append(col)

    tr = features_10[features_10["cow_id"].isin(train_cows)]
    va = features_10[features_10["cow_id"].isin(val_cows)]
    te = features_10[features_10["cow_id"].isin(test_cows)]

    # Standard 16-feature daily model for synthetic_cattle.db compatibility
    daily_model = make_pipeline(
        SimpleImputer(strategy="median", add_indicator=True),
        new_model(seed=seed),
    )
    daily_model.fit(tr[DEMO_ESTRUS_FEATURES], tr["heat"].astype(int))
    va_prob_d = daily_model.predict_proba(va[DEMO_ESTRUS_FEATURES])[:, 1]
    thresh_d = choose_threshold(va["heat"].astype(int).to_numpy(), va_prob_d)
    te_prob_d = daily_model.predict_proba(te[DEMO_ESTRUS_FEATURES])[:, 1]
    daily_metrics = score(te["heat"].astype(int).to_numpy(), te_prob_d, thresh_d)

    # Enriched hourly+daily+prediction CSV model + days_to_next_heat regressor
    enriched_model = make_pipeline(
        SimpleImputer(strategy="median", add_indicator=True),
        new_model(seed=seed),
    )
    enriched_model.fit(tr[base_cols], tr["heat"].astype(int))
    va_prob_e = enriched_model.predict_proba(va[base_cols])[:, 1]
    thresh_e = choose_threshold(va["heat"].astype(int).to_numpy(), va_prob_e)
    te_prob_e = enriched_model.predict_proba(te[base_cols])[:, 1]
    enriched_metrics = score(te["heat"].astype(int).to_numpy(), te_prob_e, thresh_e)

    next_heat_mae = None
    next_heat_reg = None
    if "days_to_next_heat_min" in features_10.columns:
        tr_nh = tr.dropna(subset=["days_to_next_heat_min"])
        te_nh = te.dropna(subset=["days_to_next_heat_min"])
        if len(tr_nh) > 10 and len(te_nh) > 5:
            next_heat_reg = make_pipeline(
                SimpleImputer(strategy="median"),
                HistGradientBoostingRegressor(max_depth=4, learning_rate=0.05, max_iter=150, random_state=seed),
            )
            next_heat_reg.fit(tr_nh[base_cols], tr_nh["days_to_next_heat_min"])
            nh_pred = next_heat_reg.predict(te_nh[base_cols])
            next_heat_mae = float(mean_absolute_error(te_nh["days_to_next_heat_min"], nh_pred))

    metrics = {
        "source": "synthetic_daily_10cows.csv + synthetic_hourly_10cows.csv + data/synthetic_cattle.db + out/estrus_predictions.csv",
        "n_cows": int(features_10["cow_id"].nunique()),
        "split_cows": {
            "train": sorted(train_cows),
            "validation": sorted(val_cows),
            "test": sorted(test_cows),
        },
        "daily_model": {
            **daily_metrics,
            "validation_threshold": thresh_d,
            "test_rows": int(len(te)),
        },
        "hourly_and_prediction_enriched_model": {
            **enriched_metrics,
            "validation_threshold": thresh_e,
            "features_used": base_cols,
            "days_to_next_heat_mae_days": next_heat_mae,
        },
        "is_synthetic": 1,
    }
    bundle = {
        "model": daily_model,
        "enriched_model": enriched_model,
        "next_heat_regressor": next_heat_reg,
        "features": DEMO_ESTRUS_FEATURES,
        "enriched_features": base_cols,
        "threshold": thresh_d,
        "enriched_threshold": thresh_e,
        "model_version": "estrus-hist_gradient_boosting-0.1.0",
        "training_scope": "synthetic-10cow-daily-hourly-pred",
    }
    return bundle, metrics
