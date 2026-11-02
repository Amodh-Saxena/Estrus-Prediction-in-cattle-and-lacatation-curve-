"""Batch scoring for Estrus, Conception, Lactation, Behaviour, and Breeding Window Schedule (PRD Section 8, FR-P3)."""
from __future__ import annotations

import argparse
from contextlib import closing
import os
import sqlite3
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "2")

import joblib
import numpy as np
import pandas as pd

from generate_data import simulate_herd
from src.features import estrus_features, estrus_features_prd, build_conception_features
from src.models.behaviour import load_real_clips
from src.models.lactation import predict_lactation

ROOT = Path(__file__).resolve().parents[1]


def run_predictions(
    daily_source: Path | None = None,
    out_dir: Path | None = None,
    models_dir: Path | None = None,
    data_dir: Path | None = None,
) -> dict[str, Path]:
    daily_path = daily_source or (ROOT / "data" / "synthetic_cattle.db")
    out_path = out_dir or (ROOT / "out")
    m_dir = models_dir or (ROOT / "models")
    d_dir = data_dir or (ROOT / "data")
    out_path.mkdir(parents=True, exist_ok=True)

    written: dict[str, Path] = {}

    # 1. Score the requested --daily input (SQLite 10-cow DB or CSV)
    if daily_path.suffix.lower() in {".db", ".sqlite", ".sqlite3"}:
        with closing(sqlite3.connect(daily_path)) as conn:
            daily_input = pd.read_sql_query(
                "SELECT d.*, c.parity FROM daily_observations d JOIN cows c USING(cow_id) ORDER BY cow_id,dim",
                conn,
            )
    else:
        daily_input = pd.read_csv(daily_path)

    hourly_csv = ROOT / "synthetic_hourly_10cows.csv"
    hourly_df = pd.read_csv(hourly_csv) if hourly_csv.exists() else None
    imu_csv = ROOT / "synthetic_imu_clips_441.csv"
    imu_df = pd.read_csv(imu_csv) if imu_csv.exists() else None

    if "activity_index" in daily_input.columns:
        features_10 = estrus_features(daily_input, hourly=hourly_df, imu_clips_df=imu_df)
        bundle_10 = joblib.load(m_dir / "estrus-hist_gradient_boosting-0.1.0.joblib")
        if "enriched_model" in bundle_10 and set(bundle_10.get("enriched_features", [])).issubset(features_10.columns):
            probs_10 = bundle_10["enriched_model"].predict_proba(features_10[bundle_10["enriched_features"]])[:, 1]
            thresh_10 = bundle_10["enriched_threshold"]
        else:
            probs_10 = bundle_10["model"].predict_proba(features_10[bundle_10["features"]])[:, 1]
            thresh_10 = bundle_10["threshold"]

        estrus_10_out = pd.DataFrame(
            {
                "cow_id": features_10["cow_id"],
                "dim": features_10["dim"],
                "p_estrus": probs_10,
                "alert": (probs_10 >= thresh_10).astype(int),
                "threshold": thresh_10,
                "model_version": bundle_10["model_version"],
                "is_synthetic": 1,
            }
        )
        estrus_10_out.to_csv(out_path / "estrus_predictions.csv", index=False)
        written["estrus_predictions"] = out_path / "estrus_predictions.csv"

    # 2. Load or generate PRD 300-cow tables to produce full FR-P3 outputs for all models
    cows_csv = d_dir / "cows.csv"
    daily_csv = d_dir / "daily.csv"
    insem_csv = d_dir / "inseminations.csv"
    if cows_csv.exists() and daily_csv.exists() and insem_csv.exists():
        cows_prd = pd.read_csv(cows_csv)
        daily_prd = pd.read_csv(daily_csv)
        insem_prd = pd.read_csv(insem_csv)
    else:
        cows_prd, daily_prd, insem_prd, _ = simulate_herd(n_cows=300, seed=42, out_dir=None)

    estrus_prd_model_path = m_dir / "estrus-hist_gradient_boosting-1.0.0.joblib"
    features_prd = None
    estrus_prd_out = None
    if estrus_prd_model_path.exists():
        bundle_prd = joblib.load(estrus_prd_model_path)
        features_prd = estrus_features_prd(daily_prd, cows_prd)
        probs_prd = bundle_prd["model"].predict_proba(features_prd[bundle_prd["features"]])[:, 1]
        thresh_prd = bundle_prd["threshold"]
        estrus_prd_out = pd.DataFrame(
            {
                "cow_id": features_prd["cow_id"],
                "dim": features_prd["dim"],
                "p_estrus": np.round(probs_prd, 6),
                "alert": (probs_prd >= thresh_prd).astype(int),
                "threshold": thresh_prd,
                "model_version": bundle_prd["model_version"],
                "is_synthetic": 1,
            }
        )
        estrus_prd_out.to_csv(out_path / "estrus_predictions_300cows.csv", index=False)
        written["estrus_predictions_300cows"] = out_path / "estrus_predictions_300cows.csv"
        if "estrus_predictions" not in written:
            estrus_prd_out.to_csv(out_path / "estrus_predictions.csv", index=False)
            written["estrus_predictions"] = out_path / "estrus_predictions.csv"

    # 3. Conception predictions (FR-P3: cow_id, dim, semen_type, p_conception, model_version)
    conc_model_path = m_dir / "conception-calibrated_ensemble-1.0.0.joblib"
    if conc_model_path.exists():
        conc_bundle = joblib.load(conc_model_path)
        insem_feat = build_conception_features(
            insem_prd,
            daily_features=features_prd,
            estrus_predictions=estrus_prd_out,
        )
        p_lr = conc_bundle["model"].predict_proba(insem_feat[conc_bundle["features"]])[:, 1]
        p_hgb = conc_bundle["boosting_model"].predict_proba(insem_feat[conc_bundle["features"]])[:, 1]
        p_conc = np.round(0.5 * p_lr + 0.5 * p_hgb, 4)
        conc_out = pd.DataFrame(
            {
                "cow_id": insem_feat["cow_id"],
                "dim": insem_feat["dim"],
                "semen_type": insem_feat["semen_type"],
                "p_conception": p_conc,
                "model_version": conc_bundle["model_version"],
                "is_synthetic": 1,
            }
        )
        conc_out.to_csv(out_path / "conception_predictions.csv", index=False)
        written["conception_predictions"] = out_path / "conception_predictions.csv"

    # 4. Lactation predictions (FR-P3: cow_id, dim, milk_kg_pred for days 61..305, total_305_pred)
    lact_model_path = m_dir / "lactation-scaled_parity_hgb-1.0.0.joblib"
    if lact_model_path.exists():
        lact_bundle = joblib.load(lact_model_path)
        lact_out = predict_lactation(daily_prd, cows_prd, lact_bundle)
        lact_out.to_csv(out_path / "lactation_predictions.csv", index=False)
        written["lactation_predictions"] = out_path / "lactation_predictions.csv"

    # 5. Behaviour predictions on real collar clips
    beh_model_path = m_dir / "behaviour-random_forest-0.1.0.joblib"
    if beh_model_path.exists():
        beh_bundle = joblib.load(beh_model_path)
        x_clips, y_clips, g_clips, clip_ids = load_real_clips(include_standing=False)
        pred_labels = beh_bundle["model"].predict(x_clips[beh_bundle["features"]])
        beh_out = pd.DataFrame(
            {
                "clip_id": clip_ids,
                "cow_id": g_clips,
                "true_label": y_clips,
                "predicted_label": pred_labels,
                "model_version": beh_bundle["model_version"],
                "is_synthetic": 0,
            }
        )
        beh_out.to_csv(out_path / "behaviour_predictions.csv", index=False)
        written["behaviour_predictions"] = out_path / "behaviour_predictions.csv"

    # 6. Breeding-window scheduler summary combining Estrus, Conception, and Lactation outputs
    if {"estrus_predictions_300cows", "conception_predictions", "lactation_predictions"}.issubset(written):
        alerts = estrus_prd_out[estrus_prd_out["alert"] == 1].copy()
        lact_summary = lact_out[["cow_id", "total_305_pred"]].drop_duplicates("cow_id")
        sched = alerts.merge(
            conc_out[["cow_id", "dim", "semen_type", "p_conception"]],
            on=["cow_id", "dim"],
            how="left",
        ).merge(lact_summary, on="cow_id", how="left")
        sched["best_insemination_window"] = sched["dim"].apply(lambda d: f"DIM {d} (12-18h post-onset)")
        sched.to_csv(out_path / "breeding_schedule.csv", index=False)
        written["breeding_schedule"] = out_path / "breeding_schedule.csv"

    for name, path in written.items():
        print(f"Wrote {name}: {path}")
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--daily", type=Path, default=ROOT / "data" / "synthetic_cattle.db")
    parser.add_argument("--out", type=Path, default=ROOT / "out")
    parser.add_argument("--models", type=Path, default=ROOT / "models")
    parser.add_argument("--data", type=Path, default=ROOT / "data")
    args = parser.parse_args()
    run_predictions(daily_source=args.daily, out_dir=args.out, models_dir=args.models, data_dir=args.data)


if __name__ == "__main__":
    main()
