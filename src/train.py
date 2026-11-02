"""Train all four PRD models (Estrus, Behaviour, Conception, Lactation) plus 10-cow hybrid models,
run automated leakage checks, and write metrics.json, data_inventory.json, and model_cards.md."""
from __future__ import annotations

import argparse
from contextlib import closing
import hashlib
import json
import os
import sqlite3
import subprocess
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "2")

import joblib
import pandas as pd
import yaml

from generate_data import simulate_herd, check_validation_targets
from src.data2_data3 import (
    analyze_and_classify_data3_jerk,
    build_data2_real_daily_telemetry,
    fit_and_evaluate_data2_behaviour,
)
from src.data4_integration import (
    analyze_and_model_smid_estrus_behaviour,
    extract_and_evaluate_eartag_estrus,
    extract_supplementary_figures_priors,
)
from src.evaluate import cow_train_validation_test, run_leakage_checks
from src.features import estrus_features, estrus_features_prd, build_conception_features
from src.inventory import build_inventory
from src.models.behaviour import fit_and_score as fit_behaviour
from src.models.conception import fit as fit_conception
from src.models.estrus import fit_and_evaluate_prd, fit_and_evaluate_10cow_hybrid
from src.models.lactation import fit as fit_lactation
from src.validate import validate_prd_tables, validate_sqlite

ROOT = Path(__file__).resolve().parents[1]


def split_cows(cow_ids, seed: int = 7, train_fraction: float = 0.60):
    return cow_train_validation_test(cow_ids, seed=seed, train_fraction=train_fraction)


def _sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_commit() -> str:
    try:
        res = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
        if res.returncode == 0 and res.stdout.strip():
            return res.stdout.strip()
    except Exception:
        pass
    return "working-tree"


def _write_reliability_svg(out_path: Path, conc_metrics: dict) -> None:
    """Write a clean SVG reliability curve diagram for the calibrated conception models (FR-M2, Section 8)."""
    lr_rel = conc_metrics["logistic_regression"]["reliability_curve"]
    hgb_rel = conc_metrics["gradient_boosting"]["reliability_curve"]

    w, h, pad = 480, 360, 55

    def to_xy(px: float, py: float) -> tuple[float, float]:
        x = pad + (w - 2 * pad) * min(max(px, 0.15), 0.55 - 1e-6 - 0.15 + 0.15 - 0.15)
        # Scale axis [0.15, 0.55] -> plot area
        xn = (px - 0.15) / 0.40
        yn = (py - 0.15) / 0.40
        return pad + xn * (w - 2 * pad), h - pad - yn * (h - 2 * pad)

    lr_pts = " ".join(
        f"{to_xy(px, py)[0]:.1f},{to_xy(px, py)[1]:.1f}"
        for px, py in zip(lr_rel["mean_predicted_prob"], lr_rel["observed_fraction_positive"])
    )
    hgb_pts = " ".join(
        f"{to_xy(px, py)[0]:.1f},{to_xy(px, py)[1]:.1f}"
        for px, py in zip(hgb_rel["mean_predicted_prob"], hgb_rel["observed_fraction_positive"])
    )
    d0 = to_xy(0.15, 0.15)
    d1 = to_xy(0.55, 0.55)
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">
  <rect width="100%" height="100%" fill="#ffffff"/>
  <text x="{w/2}" y="26" text-anchor="middle" font-family="sans-serif" font-size="14" font-weight="bold">Conception Probability Reliability Curve (5-Fold Group CV)</text>
  <line x1="{pad}" y1="{h-pad}" x2="{w-pad}" y2="{h-pad}" stroke="#333" stroke-width="1.5"/>
  <line x1="{pad}" y1="{pad}" x2="{pad}" y2="{h-pad}" stroke="#333" stroke-width="1.5"/>
  <line x1="{d0[0]:.1f}" y1="{d0[1]:.1f}" x2="{d1[0]:.1f}" y2="{d1[1]:.1f}" stroke="#888" stroke-dasharray="5,5" stroke-width="1.5"/>
  <polyline fill="none" stroke="#1f77b4" stroke-width="2.5" points="{lr_pts}"/>
  <polyline fill="none" stroke="#ff7f0e" stroke-width="2.5" points="{hgb_pts}"/>
  <text x="{w/2}" y="{h-15}" text-anchor="middle" font-family="sans-serif" font-size="12">Mean Predicted Conception Probability</text>
  <text x="16" y="{h/2}" text-anchor="middle" font-family="sans-serif" font-size="12" transform="rotate(-90 16,{h/2})">Observed Conception Fraction</text>
  <rect x="{pad+12}" y="{pad+10}" width="195" height="48" fill="#f9f9f9" stroke="#ccc"/>
  <line x1="{pad+20}" y1="{pad+26}" x2="{pad+42}" y2="{pad+26}" stroke="#1f77b4" stroke-width="2.5"/>
  <text x="{pad+48}" y="{pad+30}" font-family="sans-serif" font-size="11">Calibrated Logistic (Brier {conc_metrics['logistic_regression']['cv_brier']:.4f})</text>
  <line x1="{pad+20}" y1="{pad+44}" x2="{pad+42}" y2="{pad+44}" stroke="#ff7f0e" stroke-width="2.5"/>
  <text x="{pad+48}" y="{pad+48}" font-family="sans-serif" font-size="11">Calibrated Boosting (Brier {conc_metrics['gradient_boosting']['cv_brier']:.4f})</text>
</svg>
"""
    out_path.write_text(svg, encoding="utf-8")


def _load_or_generate_prd_tables(data_dir: Path, n_cows: int = 300, seed: int = 42):
    cows_path = data_dir / "cows.csv"
    daily_path = data_dir / "daily.csv"
    insem_path = data_dir / "inseminations.csv"
    if cows_path.exists() and daily_path.exists() and insem_path.exists():
        cows = pd.read_csv(cows_path)
        daily = pd.read_csv(daily_path)
        insem = pd.read_csv(insem_path)
        if len(cows) == n_cows:
            return cows, daily, insem
    try:
        cows, daily, insem, stats = simulate_herd(n_cows=n_cows, seed=seed, out_dir=data_dir)
    except PermissionError:
        cows, daily, insem, stats = simulate_herd(n_cows=n_cows, seed=seed, out_dir=None)
    if n_cows >= 30:
        check_validation_targets(stats, n_cows=n_cows)
    return cows, daily, insem


def _format_model_cards(report: dict) -> str:
    est = report["models"]["estrus"]
    beh = report["models"]["behaviour"]
    conc = report["models"]["conception"]
    lact = report["models"]["lactation"]
    leak = report["leakage_tests"]
    lines = [
        "# Model Cards — Estrus, Breeding, Behaviour and Lactation Prediction System",
        "",
        f"Generated: `{report['generated_at_utc']}`",
        "",
        "## 1. Estrus Detector (`estrus-hist_gradient_boosting-1.0.0` & `0.1.0`)",
        f"- **Data sources**: Synthetic PRD 300-cow tables (`data/daily.csv`, `data/cows.csv`) + 10-cow assets (`synthetic_daily_10cows.csv`, `synthetic_hourly_10cows.csv`, `data/synthetic_cattle.db`, `out/estrus_predictions.csv`).",
        f"- **Held-out test F1 (300 cows)**: `{est['f1']:.4f}` (Target >= 0.85) | **PR-AUC**: `{est['pr_auc']:.4f}` (Target >= 0.90)",
        f"- **Activity Rule Baseline F1**: `{est['baselines']['activity_ratio_rule']['f1']:.4f}` | **Logistic Baseline F1**: `{est['baselines']['balanced_logistic_regression']['f1']:.4f}`",
        f"- **Normal-heat Recall**: `{est.get('normal_heat_recall', 0.0):.4f}` | **Silent-heat Recall**: `{est.get('silent_heat_recall', 0.0):.4f}`",
        f"- **10-Cow Enriched (Hourly + Daily + Prediction CSV) F1**: `{est['demo_10cow']['hourly_and_prediction_enriched_model']['f1']:.4f}` (`PR-AUC`: `{est['demo_10cow']['hourly_and_prediction_enriched_model']['pr_auc']:.4f}`)",
        f"- **`data4` Smid et al. (2024) Real Dairy Estrus Duration Regressor (5-Fold Cow CV)**: MAE = `{est.get('data4_smid_estrus_behaviour', {}).get('estrus_duration_regressor_5fold_cv', {}).get('mae_hours', 0.0):.3f} h`, R² = `{est.get('data4_smid_estrus_behaviour', {}).get('estrus_duration_regressor_5fold_cv', {}).get('r2_score', 0.0):.4f}` (`94` SAS-valid estrus events across `45` cows).",
        f"- **`data4` Electronic Ear-Tag Temperature & Step Monitoring**: `{est.get('data4_eartag_temperature_step_monitoring', {}).get('n_animal_days', 0)}` real animal-days (`Peri-estrus peak step ratio`: `{est.get('data4_eartag_temperature_step_monitoring', {}).get('peri_estrus_vs_baseline', {}).get('event_peak_steps_ratio_mean', 1.0):.2f}x`, `Peak thermal shift`: `+{est.get('data4_eartag_temperature_step_monitoring', {}).get('peri_estrus_vs_baseline', {}).get('event_peak_temp_q90_diff_c', 0.0):.2f} °C`).",
        "",
        "## 2. Behaviour Classifier (`behaviour-random_forest-0.1.0` & `behaviour-data2_collar_halter-1.0.0`)",
        f"- **Data sources**: Real 10 Hz collar IMU CSV clips (`db-cow-walking-main`: 332 3-class clips, 356 4-class clips with `Standing`) + `data2/` (`Precision Beef Zenodo 4064802`: 18 cattle, 10 Hz Afimilk Collar + RumiWatch Halter) + `data3/Preprocessed_Final.csv` (1 Hz 3-axis jerk stream) + `synthetic_imu_clips_441.csv`.",
        f"- **5-Fold Cow-Grouped Macro F1 (`db-cow-walking`, 3 classes)**: `{beh['group_cv_macro_f1']:.4f}` (Target >= 0.85) | **Accuracy**: `{beh['group_cv_accuracy']:.4f}`",
        f"- **4-Class with Standing Macro F1 (`db-cow-walking`)**: `{beh['with_standing_4class']['group_cv_macro_f1']:.4f}` | **Accuracy**: `{beh['with_standing_4class']['group_cv_accuracy']:.4f}`",
        f"- **`data2` Precision Beef 5-Fold Cow-Grouped Macro F1 (17 cows)**: `{beh.get('data2_precision_beef', {}).get('group_cv_5fold', {}).get('hgb_macro_f1', 0.0):.4f}` | **Official Test Split (`Cows 04, 10, 11`) Macro F1**: `{beh.get('data2_precision_beef', {}).get('official_test_split_cows_04_10_11', {}).get('hgb_macro_f1', 0.0):.4f}` (`Accuracy`: `{beh.get('data2_precision_beef', {}).get('official_test_split_cows_04_10_11', {}).get('hgb_accuracy', 0.0):.4f}`)",
        f"- **`data3` Overnight Jerk Stream (`Preprocessed_Final.csv`)**: `{beh.get('data3_jerk_telemetry', {}).get('n_windows_5min', 0)}` five-minute windows classified (`Mean Restlessness`: `{beh.get('data3_jerk_telemetry', {}).get('mean_restlessness_index', 0.0):.4f}`).",
        "",
        "## 3. Conception Model (`conception-calibrated_ensemble-1.0.0`)",
        f"- **Data sources**: `data/inseminations.csv` + causal baseline strengths from `data/daily.csv` (FR-F4) + out-of-fold `p_estrus` from estrus predictions.",
        f"- **Base-rate Brier**: `{conc['base_rate_brier']:.4f}` (Conception rate: `{conc['conception_rate']*100:.1f}%`)",
        f"- **Calibrated Logistic 5-Fold CV Brier**: `{conc['logistic_regression']['cv_brier']:.4f}` | **AUC**: `{conc['logistic_regression']['cv_auc']:.4f}`",
        f"- **Calibrated Boosting 5-Fold CV Brier**: `{conc['gradient_boosting']['cv_brier']:.4f}` | **AUC**: `{conc['gradient_boosting']['cv_auc']:.4f}`",
        "",
        "## 4. Lactation Model (`lactation-scaled_parity_hgb-1.0.0`)",
        f"- **Data sources**: `data/daily.csv` (DIM 5–60 early window -> DIM 61–305 forecast) + `data/cows.csv`.",
        f"- **Held-out Test MAE**: `{lact['mae_kg']:.3f} kg/day` (`{lact['mae_reduction_vs_parity_curve_pct']:.1f}%` below parity-mean curve MAE `{lact['baselines']['parity_mean_curve']['mae_kg']:.3f} kg/day`)",
        f"- **Remaining-Yield Error (DIM 61–305)**: `{lact['remaining_yield_error_pct']:.2f}%` (Target <= 5.0%) | **MAPE**: `{lact['mape_pct']:.2f}%`",
        "",
        "## 5. Automated Leakage Checks (FR-E6)",
        f"- Disjoint cow splits: `{leak['no_cow_overlap']}`",
        f"- Causal rolling baselines (no future leakage): `{leak['causal_baselines']}`",
        f"- Baseline excludes today: `{leak['baseline_excludes_today']}`",
        f"- Parity curve built from training cows only: `{leak['parity_curve_train_only']}`",
        "",
        "## 6. Known Limitations",
        "- Estrus, conception, and lactation metrics are evaluated on synthetic data (`is_synthetic=1`) to validate the pipeline; real farm accuracy must be measured once real cow-day and breeding records arrive.",
        "- Camera images in `Wagyu&Angus` are inventoried in `reports/data_inventory.json` and remain out of scope for v1.",
    ]
    return "\n".join(lines) + "\n"


def run_training(config_path: Path | None = None, out_root: Path | None = None, n_cows_override: int | None = None) -> dict:
    cfg_path = config_path or (ROOT / "config.yaml")
    config = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    base_root = out_root or ROOT

    data_dir = base_root / config["project"].get("data_dir", "data")
    database = ROOT / config["project"]["database"]
    out_models = base_root / config["project"]["models_dir"]
    out_reports = base_root / config["project"]["reports_dir"]
    out_preds = base_root / config["project"].get("out_dir", "out")
    out_models.mkdir(parents=True, exist_ok=True)
    out_reports.mkdir(parents=True, exist_ok=True)
    out_preds.mkdir(parents=True, exist_ok=True)

    seed = int(config["project"].get("seed", 42))
    split_seed = int(config["project"].get("split_seed", 7))
    n_cows_prd = n_cows_override or int(config.get("simulator", {}).get("n_cows", 300))

    # -------------------------------------------------------------------------
    # Stage 1: Load and validate 10-cow SQLite DB + CSVs + prediction CSV
    # -------------------------------------------------------------------------
    validate_sqlite(database)
    with closing(sqlite3.connect(database)) as conn:
        daily_10_db = pd.read_sql_query(
            "SELECT d.*, c.parity FROM daily_observations d JOIN cows c USING(cow_id) ORDER BY cow_id,dim",
            conn,
        )

    daily_10_csv_path = ROOT / "synthetic_daily_10cows.csv"
    hourly_10_csv_path = ROOT / "synthetic_hourly_10cows.csv"
    imu_441_csv_path = ROOT / "synthetic_imu_clips_441.csv"
    pred_csv_path = ROOT / "out" / "estrus_predictions.csv"

    daily_10 = pd.read_csv(daily_10_csv_path) if daily_10_csv_path.exists() else daily_10_db
    if "parity" not in daily_10.columns:
        daily_10 = daily_10.merge(daily_10_db[["cow_id", "parity"]].drop_duplicates(), on="cow_id", how="left")
    hourly_10 = pd.read_csv(hourly_10_csv_path) if hourly_10_csv_path.exists() else None
    imu_441 = pd.read_csv(imu_441_csv_path) if imu_441_csv_path.exists() else None
    existing_pred_csv = pd.read_csv(pred_csv_path) if pred_csv_path.exists() else None

    features_10 = estrus_features(
        daily_10,
        window=int(config.get("features", {}).get("demo_10cow_window_days", 7)),
        hourly=hourly_10,
        predictions_df=existing_pred_csv,
        imu_clips_df=imu_441,
    )
    tr_10, va_10, te_10 = split_cows(features_10["cow_id"], seed=split_seed)
    bundle_10, metrics_10 = fit_and_evaluate_10cow_hybrid(
        features_10, tr_10, va_10, te_10, seed=split_seed
    )

    # -------------------------------------------------------------------------
    # Stage 2: Load/Generate and validate PRD 300-cow tables (cows, daily, inseminations)
    # -------------------------------------------------------------------------
    cows_prd, daily_prd, insem_prd = _load_or_generate_prd_tables(data_dir, n_cows=n_cows_prd, seed=seed)
    validate_prd_tables(cows_prd, daily_prd, insem_prd)

    tr_cows, va_cows, te_cows = split_cows(
        cows_prd["cow_id"],
        seed=split_seed,
        train_fraction=float(config.get("splits", {}).get("train_fraction", 0.60)),
    )

    # Automated leakage checks (FR-E6)
    leakage_results = run_leakage_checks(daily_prd, cows_prd, tr_cows, va_cows, te_cows)

    # -------------------------------------------------------------------------
    # Stage 3: Train & Evaluate PRD Estrus Detector (Section 6.2)
    # -------------------------------------------------------------------------
    features_prd = estrus_features_prd(
        daily_prd,
        cows_prd,
        window=int(config.get("features", {}).get("baseline_window_days", 14)),
        min_periods=int(config.get("features", {}).get("baseline_min_periods", 5)),
        warmup_dim=int(config.get("features", {}).get("warmup_dim", 15)),
    )
    estrus_bundle_prd, estrus_metrics_prd, estrus_oof_preds = fit_and_evaluate_prd(
        features_prd,
        tr_cows,
        va_cows,
        te_cows,
        inseminations_df=insem_prd,
        seed=split_seed,
        hgb_cfg=config.get("estrus", {}).get("hist_gradient_boosting", {}),
    )
    estrus_metrics_prd["demo_10cow"] = metrics_10
    estrus_metrics_prd["split_cows_count"] = {
        "train": len(tr_cows),
        "validation": len(va_cows),
        "test": len(te_cows),
    }

    # Integrate data4 (Smid et al. Dairy Cow Estrus Behaviour + Electronic Ear-Tag Temperature/Steps + Supp Figures)
    data4_smid_bundle, data4_smid_metrics, _ = analyze_and_model_smid_estrus_behaviour(
        out_csv=data_dir / "real_data4_smid_estrus.csv", seed=split_seed
    )
    _, data4_eartag_metrics, data4_eartag_bundle = extract_and_evaluate_eartag_estrus(
        out_csv=data_dir / "real_data4_eartag_daily.csv", seed=split_seed
    )
    data4_supp_priors = extract_supplementary_figures_priors()
    estrus_metrics_prd["data4_smid_estrus_behaviour"] = data4_smid_metrics
    estrus_metrics_prd["data4_eartag_temperature_step_monitoring"] = data4_eartag_metrics
    estrus_metrics_prd["data4_supplementary_figures_priors"] = data4_supp_priors

    # -------------------------------------------------------------------------
    # Stage 4: Train & Evaluate Behaviour Classifier (Section 6.1 + data2 + data3)
    # -------------------------------------------------------------------------
    behaviour_bundle, behaviour_metrics = fit_behaviour(
        seed=split_seed,
        rf_cfg=config.get("behaviour", {}).get("random_forest", {}),
    )
    data2_bundle, data2_beh_metrics, data2_windows_df = fit_and_evaluate_data2_behaviour(seed=split_seed)
    _, data2_daily_summary = build_data2_real_daily_telemetry(
        data2_windows_df, out_csv=data_dir / "real_data2_daily.csv"
    )
    _, data3_summary = analyze_and_classify_data3_jerk(
        out_windows_csv=data_dir / "real_data3_windows.csv", seed=split_seed
    )
    behaviour_metrics["data2_precision_beef"] = data2_beh_metrics
    behaviour_metrics["data2_real_daily_telemetry"] = data2_daily_summary
    behaviour_metrics["data3_jerk_telemetry"] = data3_summary

    # -------------------------------------------------------------------------
    # Stage 5: Train & Evaluate Conception Model (Section 6.3, FR-M2 & FR-F4)
    # Uses inseminations.csv + causal daily ratios + out-of-fold estrus_predictions
    # -------------------------------------------------------------------------
    insem_enriched = build_conception_features(
        insem_prd,
        daily_features=features_prd,
        estrus_predictions=estrus_oof_preds,
    )
    conception_bundle, conception_metrics = fit_conception(
        insem_enriched,
        train_cows=tr_cows,
        val_cows=va_cows,
        test_cows=te_cows,
        seed=split_seed,
        cfg=config.get("conception", {}),
    )

    # -------------------------------------------------------------------------
    # Stage 6: Train & Evaluate Lactation Model (Section 6.4)
    # -------------------------------------------------------------------------
    lactation_bundle, lactation_metrics = fit_lactation(
        daily_prd,
        cows=cows_prd,
        train_cows=tr_cows,
        val_cows=va_cows,
        test_cows=te_cows,
        seed=split_seed,
        hgb_cfg=config.get("lactation", {}).get("hist_gradient_boosting", {}),
    )

    # -------------------------------------------------------------------------
    # Stage 7: Save versioned model artifacts with traceability metadata (NFR-4)
    # -------------------------------------------------------------------------
    code_commit = _git_commit()
    db_hash = _sha256_file(database)
    csv_hashes = {
        "data/cows.csv": _sha256_file(data_dir / "cows.csv"),
        "data/daily.csv": _sha256_file(data_dir / "daily.csv"),
        "data/inseminations.csv": _sha256_file(data_dir / "inseminations.csv"),
        "synthetic_daily_10cows.csv": _sha256_file(daily_10_csv_path),
        "synthetic_hourly_10cows.csv": _sha256_file(hourly_10_csv_path),
        "synthetic_imu_clips_441.csv": _sha256_file(imu_441_csv_path),
        "out/estrus_predictions.csv": _sha256_file(pred_csv_path),
    }

    # NFR-7: Stage logging of row counts, split sizes, and thresholds
    print(
        f"[Train] Splits: train={len(tr_cows)} cows, val={len(va_cows)} cows, test={len(te_cows)} cows | "
        f"daily_rows={len(daily_prd)}, feature_rows={len(features_prd)}, inseminations={len(insem_prd)}"
    )
    print(
        f"[Train] Thresholds: estrus_prob={estrus_bundle_prd['threshold']:.2f}, "
        f"activity_rule={estrus_bundle_prd['rule_threshold']:.2f}, "
        f"demo_10cow_prob={bundle_10['enriched_threshold']:.2f}"
    )

    joblib.dump(
        {
            **bundle_10,
            "prd_model": estrus_bundle_prd["model"],
            "prd_features": estrus_bundle_prd["features"],
            "prd_threshold": estrus_bundle_prd["threshold"],
            "config": config,
            "data_hash": csv_hashes,
            "database_sha256": db_hash,
            "code_commit": code_commit,
            "metrics": estrus_metrics_prd,
        },
        out_models / "estrus-hist_gradient_boosting-0.1.0.joblib",
    )
    joblib.dump(
        {
            **estrus_bundle_prd,
            "config": config,
            "data_hash": csv_hashes,
            "database_sha256": db_hash,
            "code_commit": code_commit,
            "metrics": estrus_metrics_prd,
        },
        out_models / "estrus-hist_gradient_boosting-1.0.0.joblib",
    )
    joblib.dump(
        {
            "model": behaviour_bundle[0],
            "features": behaviour_bundle[1],
            "model_version": "behaviour-random_forest-0.1.0",
            "training_scope": "real labeled collar clips + synthetic_imu_clips_441.csv",
            "config": config,
            "data_hash": csv_hashes,
            "database_sha256": db_hash,
            "code_commit": code_commit,
            "metrics": behaviour_metrics,
        },
        out_models / "behaviour-random_forest-0.1.0.joblib",
    )
    if data2_bundle:
        joblib.dump(
            {
                **data2_bundle,
                "config": config,
                "code_commit": code_commit,
            },
            out_models / "behaviour-data2_collar_halter-1.0.0.joblib",
        )
    if data4_smid_bundle:
        joblib.dump(
            {
                **data4_smid_bundle,
                "config": config,
                "code_commit": code_commit,
            },
            out_models / "estrus-data4_smid_behaviour-1.0.0.joblib",
        )
    if data4_eartag_bundle:
        joblib.dump(
            {
                **data4_eartag_bundle,
                "config": config,
                "code_commit": code_commit,
            },
            out_models / "estrus-data4_eartag_3sheet-1.0.0.joblib",
        )
    joblib.dump(
        {
            **conception_bundle,
            "config": config,
            "data_hash": csv_hashes,
            "database_sha256": db_hash,
            "code_commit": code_commit,
            "metrics": conception_metrics,
        },
        out_models / "conception-calibrated_ensemble-1.0.0.joblib",
    )
    joblib.dump(
        {
            **lactation_bundle,
            "config": config,
            "data_hash": csv_hashes,
            "database_sha256": db_hash,
            "code_commit": code_commit,
            "metrics": lactation_metrics,
        },
        out_models / "lactation-scaled_parity_hgb-1.0.0.joblib",
    )

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "code_commit": code_commit,
        "database_sha256": db_hash,
        "source_csv_sha256": csv_hashes,
        "data_contract": {
            "database": str(database.relative_to(ROOT)),
            "prd_tables": ["data/cows.csv", "data/daily.csv", "data/inseminations.csv"],
            "demo_10cow_csvs": [
                "synthetic_daily_10cows.csv",
                "synthetic_hourly_10cows.csv",
                "synthetic_imu_clips_441.csv",
                "out/estrus_predictions.csv",
            ],
            "real_collar_dir": "db-cow-walking-main",
            "synthetic_rows_flagged": True,
        },
        "leakage_tests": leakage_results,
        "models": {
            "estrus": estrus_metrics_prd,
            "behaviour": behaviour_metrics,
            "conception": conception_metrics,
            "lactation": lactation_metrics,
        },
        "limitations": [
            "Synthetic 300-cow and 10-cow datasets validate the pipeline and model calibration; they are not evidence of real farm accuracy.",
            "Behaviour classification is validated on real 10 Hz collar IMU clips (db-cow-walking-main) plus synthetic IMU summaries.",
            "Camera images in Wagyu&Angus are inventoried in reports/data_inventory.json and remain out of scope for v1.",
        ],
    }

    (out_reports / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out_reports / "data_inventory.json").write_text(
        json.dumps(build_inventory(), indent=2), encoding="utf-8"
    )
    (out_reports / "model_cards.md").write_text(_format_model_cards(report), encoding="utf-8")
    _write_reliability_svg(out_reports / "conception_reliability.svg", conception_metrics)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    parser.add_argument("--out-root", type=Path, default=None, help="Optional output root directory")
    parser.add_argument("--n-cows", type=int, default=None, help="Optional override for number of simulated cows")
    args = parser.parse_args()
    report = run_training(config_path=args.config, out_root=args.out_root, n_cows_override=args.n_cows)
    summary = {
        "estrus": {
            "f1": report["models"]["estrus"]["f1"],
            "pr_auc": report["models"]["estrus"]["pr_auc"],
            "rule_f1": report["models"]["estrus"]["baselines"]["activity_ratio_rule"]["f1"],
            "10cow_enriched_f1": report["models"]["estrus"]["demo_10cow"]["hourly_and_prediction_enriched_model"]["f1"],
        },
        "behaviour": {
            "macro_f1_3class": report["models"]["behaviour"]["group_cv_macro_f1"],
            "accuracy_3class": report["models"]["behaviour"]["group_cv_accuracy"],
            "macro_f1_4class_standing": report["models"]["behaviour"]["with_standing_4class"]["group_cv_macro_f1"],
            "data2_5fold_macro_f1": report["models"]["behaviour"].get("data2_precision_beef", {}).get("group_cv_5fold", {}).get("hgb_macro_f1"),
            "data2_official_test_f1": report["models"]["behaviour"].get("data2_precision_beef", {}).get("official_test_split_cows_04_10_11", {}).get("hgb_macro_f1"),
        },
        "conception": {
            "base_rate_brier": report["models"]["conception"]["base_rate_brier"],
            "logistic_cv_brier": report["models"]["conception"]["logistic_regression"]["cv_brier"],
            "boosting_cv_brier": report["models"]["conception"]["gradient_boosting"]["cv_brier"],
        },
        "lactation": {
            "mae_kg": report["models"]["lactation"]["mae_kg"],
            "mape_pct": report["models"]["lactation"]["mape_pct"],
            "remaining_yield_error_pct": report["models"]["lactation"]["remaining_yield_error_pct"],
        },
        "leakage_tests_passed": report["leakage_tests"]["all_passed"],
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
