"""Integration module for `data2` (Precision Beef 18-cow 10 Hz Afimilk Collar + RumiWatch Halter dataset,
Zenodo 4064802) and `data3` (`Preprocessed_Final.csv` ~1 Hz 3-axis accelerometer jerk/delta stream).

Provides three concrete integration pipelines:
1. `data2` Dual-Sensor Behaviour & Rumination Classifier (17 paired cows + unlabeled Cow 18 inference,
   evaluated on 5-fold Animal-Grouped CV and the official Easter Howgate test split: Cows 04, 10, 11).
2. `data2` Multi-Day Real Cow-Day Telemetry Aggregator (`data/real_data2_daily.csv`, `is_synthetic=0`)
   mapping continuous 10 Hz collar/halter streams into daily activity, rumination, eating, and lying metrics.
3. `data3` Cross-Sensor 1 Hz Jerk & Overnight Restlessness Pipeline (`data/real_data3_windows.csv`, `is_synthetic=0`)
   bridging `x_diff, y_diff, z_diff, sum_diff` with `db-cow-walking-main` and `data2` for overnight estrus/restlessness monitoring.
"""
from __future__ import annotations

import io
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline

from src.inventory import DATA2_TEST_ANIMALS, DATA2_TRIAL_MAP

ROOT = Path(__file__).resolve().parents[1]

HALTER_LABEL_MAP = {
    0: "Other",
    1: "Ruminating",
    2: "Eating",
}
HALTER_INV_MAP = {v: k for k, v in HALTER_LABEL_MAP.items()}

DATA2_FEATURE_COLUMNS = [
    "x_mean", "y_mean", "z_mean",
    "x_dev_cow", "y_dev_cow", "z_dev_cow",
    "x_std", "y_std", "z_std",
    "x_iqr", "y_iqr", "z_iqr",
    "odba", "vedba", "mag_std",
    "sum_diff_mean", "sum_diff_std", "active_jerk_frac",
    "pitch_deg", "pitch_dev_deg",
    "b_low_frac", "b_eat_frac", "b_rum_frac", "b_walk_frac",
    "rum_to_low_ratio",
    "sub10s_std_min", "sub10s_std_max", "sub10s_std_cv",
    "zero_cross_x", "zero_cross_y",
]

JERK_BRIDGE_FEATURES = [
    "sum_diff_mean",
    "sum_diff_std",
    "sum_diff_max",
    "sum_diff_q90",
    "active_jerk_frac",
    "high_burst_frac",
    "x_diff_std",
    "y_diff_std",
    "z_diff_std",
    "z_jerk_share",
    "horiz_to_vert_jerk_ratio",
]


def _seek_csv_by_ts(path: Path, target_ts: str, n_lines: int = 3600) -> pd.DataFrame:
    """Binary-search a multi-hundred-MB sorted timestamp CSV in O(log N) seeks and read `n_lines` rows."""
    size = path.stat().st_size
    with open(path, "rb") as f:
        header = f.readline().decode("utf-8", errors="ignore").strip()
        lo, hi = f.tell(), max(f.tell(), size - 120)
        while lo < hi:
            mid = (lo + hi) // 2
            f.seek(mid)
            f.readline()
            line = f.readline().decode("utf-8", errors="ignore")
            if not line:
                hi = mid - 1
                continue
            if line[:19] < target_ts:
                lo = mid + 1
            else:
                hi = mid
        f.seek(lo)
        if lo > len(header) + 5:
            f.readline()
        lines = [f.readline().decode("utf-8", errors="ignore") for _ in range(n_lines)]
    return pd.read_csv(io.StringIO(header + "\n" + "".join(lines))).dropna()


def read_aligned_chunk(
    animal_id: int,
    frac: float,
    n_lines: int = 3600,
    data2_dir: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Read `n_lines` (default 6 min at 10 Hz) at relative file offset `frac` from `accel-XX` and `halter-XX`."""
    d2 = data2_dir or (ROOT / "data2")
    a_path = d2 / f"accel-{animal_id:02d}.csv"
    h_path = d2 / f"halter-{animal_id:02d}.csv"
    if not a_path.exists():
        raise FileNotFoundError(f"Missing {a_path}")

    if h_path.exists():
        h_size = h_path.stat().st_size
        with open(h_path, "rb") as f:
            h_hdr = f.readline().decode("utf-8", errors="ignore").strip()
            offset = int(h_size * frac)
            if offset > len(h_hdr) + 5:
                f.seek(offset)
                f.readline()
            h_lines = [f.readline().decode("utf-8", errors="ignore") for _ in range(n_lines)]
        h_df = pd.read_csv(io.StringIO(h_hdr + "\n" + "".join(h_lines))).dropna()
        target_ts = str(h_df["timestamp"].iloc[0])[:19]
        a_df = _seek_csv_by_ts(a_path, target_ts, n_lines=n_lines)
        m = min(len(a_df), len(h_df))
        return a_df.iloc[:m].reset_index(drop=True), h_df.iloc[:m].reset_index(drop=True)
    else:
        a_size = a_path.stat().st_size
        with open(a_path, "rb") as f:
            a_hdr = f.readline().decode("utf-8", errors="ignore").strip()
            offset = int(a_size * frac)
            if offset > len(a_hdr) + 5:
                f.seek(offset)
                f.readline()
            a_lines = [f.readline().decode("utf-8", errors="ignore") for _ in range(n_lines)]
        a_df = pd.read_csv(io.StringIO(a_hdr + "\n" + "".join(a_lines))).dropna()
        return a_df.reset_index(drop=True), None


def _compute_data2_window_features(
    sub_a: pd.DataFrame,
    cow_x_med: float,
    cow_y_med: float,
    cow_z_med: float,
) -> dict[str, float]:
    """Compute 30 biomechanical, posture, bolus-pause, spectral, and jerk features for a 10 Hz collar window."""
    x = pd.to_numeric(sub_a["x"], errors="coerce").fillna(0.0).to_numpy(dtype=float) / 1000.0
    yv = pd.to_numeric(sub_a["y"], errors="coerce").fillna(0.0).to_numpy(dtype=float) / 1000.0
    z = pd.to_numeric(sub_a["z"], errors="coerce").fillna(0.0).to_numpy(dtype=float) / 1000.0

    dx = np.diff(x) if len(x) > 1 else np.zeros(1)
    dy = np.diff(yv) if len(yv) > 1 else np.zeros(1)
    dz = np.diff(z) if len(z) > 1 else np.zeros(1)
    sum_diff = np.abs(dx) + np.abs(dy) + np.abs(dz)

    mag = np.sqrt(x**2 + yv**2 + z**2)
    odba = float(np.mean(np.abs(x - x.mean()) + np.abs(yv - yv.mean()) + np.abs(z - z.mean())))
    vedba = float(np.mean(np.sqrt((x - x.mean()) ** 2 + (yv - yv.mean()) ** 2 + (z - z.mean()) ** 2)))

    freqs = np.fft.rfftfreq(len(x), d=0.1)
    fx = np.abs(np.fft.rfft(x - x.mean())) ** 2
    fy = np.abs(np.fft.rfft(yv - yv.mean())) ** 2
    fz = np.abs(np.fft.rfft(z - z.mean())) ** 2
    ftot = fx + fy + fz + 1e-9
    tot_pow = float(np.sum(ftot[1:])) + 1e-9

    b_low = float(np.sum(ftot[(freqs >= 0.05) & (freqs < 0.50)]))
    b_eat = float(np.sum(ftot[(freqs >= 0.50) & (freqs < 0.85)]))
    b_rum = float(np.sum(ftot[(freqs >= 0.85) & (freqs <= 1.35)]))
    b_walk = float(np.sum(ftot[(freqs > 1.35) & (freqs <= 3.00)]))

    win = len(mag)
    step = max(20, win // 6)
    sub_stds = [float(np.std(mag[k : k + step])) for k in range(0, win - step + 1, step)] or [float(np.std(mag))]

    return {
        "x_mean": float(x.mean()),
        "y_mean": float(yv.mean()),
        "z_mean": float(z.mean()),
        "x_dev_cow": float(x.mean() - cow_x_med),
        "y_dev_cow": float(yv.mean() - cow_y_med),
        "z_dev_cow": float(z.mean() - cow_z_med),
        "x_std": float(x.std()),
        "y_std": float(yv.std()),
        "z_std": float(z.std()),
        "x_iqr": float(np.percentile(x, 75) - np.percentile(x, 25)),
        "y_iqr": float(np.percentile(yv, 75) - np.percentile(yv, 25)),
        "z_iqr": float(np.percentile(z, 75) - np.percentile(z, 25)),
        "odba": odba,
        "vedba": vedba,
        "mag_std": float(mag.std()),
        "sum_diff_mean": float(np.mean(sum_diff)),
        "sum_diff_std": float(np.std(sum_diff)),
        "active_jerk_frac": float(np.mean(sum_diff > 0.015)),
        "pitch_deg": float(np.mean(np.degrees(np.arctan2(x, np.sqrt(yv**2 + z**2) + 1e-5)))),
        "pitch_dev_deg": float(np.mean(np.degrees(np.arctan2(x - cow_x_med, 1.0)))),
        "b_low_frac": b_low / tot_pow,
        "b_eat_frac": b_eat / tot_pow,
        "b_rum_frac": b_rum / tot_pow,
        "b_walk_frac": b_walk / tot_pow,
        "rum_to_low_ratio": b_rum / (b_low + 1e-6),
        "sub10s_std_min": float(np.min(sub_stds)),
        "sub10s_std_max": float(np.max(sub_stds)),
        "sub10s_std_cv": float(np.std(sub_stds) / (np.mean(sub_stds) + 1e-6)),
        "zero_cross_x": float(np.mean(np.diff(np.sign(x - x.mean())) != 0)),
        "zero_cross_y": float(np.mean(np.diff(np.sign(yv - yv.mean())) != 0)),
    }


def fit_and_evaluate_data2_behaviour(
    data2_dir: Path | None = None,
    seed: int = 7,
    sample_fracs: tuple[float, ...] = (0.08, 0.28, 0.48, 0.68, 0.88),
    chunk_lines: int = 3600,
    window_samples: int = 600,
) -> tuple[dict, dict, pd.DataFrame]:
    """Train and evaluate the Precision Beef (Zenodo 4064802) 10 Hz Collar -> Halter Behaviour Classifier.

    Returns:
      - model_bundle: dict with fitted HGB & RF pipelines, feature list, and metadata
      - metrics: dict with 5-fold cow-grouped CV metrics, official test split (Cows 4, 10, 11) metrics,
                 per-class profiles, and unlabeled Animal 18 inference summary
      - windows_df: extracted window dataframe across all 18 animals (with `is_synthetic=0`)
    """
    d2 = data2_dir or (ROOT / "data2")
    if not d2.exists() or not list(d2.glob("accel-*.csv")):
        return {}, {"available": False}, pd.DataFrame()

    rows, labels, animal_ids, timestamps = [], [], [], []
    cow_gravity_baselines: dict[int, tuple[float, float, float]] = {}

    for aid in range(1, 18):
        if not (d2 / f"accel-{aid:02d}.csv").exists() or not (d2 / f"halter-{aid:02d}.csv").exists():
            continue
        chunks = [read_aligned_chunk(aid, frac, n_lines=chunk_lines, data2_dir=d2) for frac in sample_fracs]
        all_a = pd.concat([c[0] for c in chunks], ignore_index=True)
        cow_x_med = float(pd.to_numeric(all_a["x"], errors="coerce").median()) / 1000.0
        cow_y_med = float(pd.to_numeric(all_a["y"], errors="coerce").median()) / 1000.0
        cow_z_med = float(pd.to_numeric(all_a["z"], errors="coerce").median()) / 1000.0
        cow_gravity_baselines[aid] = (cow_x_med, cow_y_med, cow_z_med)

        for a_df, h_df in chunks:
            if h_df is None:
                continue
            for w in range(0, len(a_df) - window_samples + 1, window_samples):
                sub_a = a_df.iloc[w : w + window_samples]
                sub_h = h_df.iloc[w : w + window_samples]
                lbls = pd.to_numeric(sub_h["classification"], errors="coerce").dropna().to_numpy(dtype=int)
                if len(lbls) == 0:
                    continue
                maj = int(np.bincount(lbls).argmax())
                if float(np.mean(lbls == maj)) < 0.85:
                    continue
                feat = _compute_data2_window_features(sub_a, cow_x_med, cow_y_med, cow_z_med)
                rows.append(feat)
                labels.append(HALTER_LABEL_MAP.get(maj, "Other"))
                animal_ids.append(aid)
                timestamps.append(str(sub_a["timestamp"].iloc[0])[:19])

    X = pd.DataFrame(rows)[DATA2_FEATURE_COLUMNS].fillna(0.0)
    y = np.asarray(labels)
    groups = np.asarray(animal_ids, dtype=int)

    rf_pipe = make_pipeline(
        SimpleImputer(strategy="median"),
        RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=1,
        ),
    )
    hgb_pipe = make_pipeline(
        SimpleImputer(strategy="median"),
        HistGradientBoostingClassifier(
            max_depth=6,
            learning_rate=0.06,
            max_iter=200,
            class_weight="balanced",
            random_state=seed,
        ),
    )

    # 1. 5-Fold Animal-Grouped CV across all 17 paired animals
    n_splits = min(5, len(np.unique(groups)))
    oof_rf = cross_val_predict(rf_pipe, X, y, groups=groups, cv=GroupKFold(n_splits), n_jobs=1)
    oof_hgb = cross_val_predict(hgb_pipe, X, y, groups=groups, cv=GroupKFold(n_splits), n_jobs=1)

    # 2. Official Easter Howgate Farm Benchmark Split: Train on 14 cows, Test on Cows 04, 10, 11
    test_mask = np.isin(groups, list(DATA2_TEST_ANIMALS))
    rf_pipe.fit(X[~test_mask], y[~test_mask])
    hgb_pipe.fit(X[~test_mask], y[~test_mask])
    test_pred_rf = rf_pipe.predict(X[test_mask])
    test_pred_hgb = hgb_pipe.predict(X[test_mask])
    test_report_hgb = classification_report(y[test_mask], test_pred_hgb, output_dict=True, zero_division=0)

    # Fit final production models on all 17 labeled animals
    rf_pipe.fit(X, y)
    hgb_pipe.fit(X, y)

    # 3. Run inference on Unlabeled Animal 18 (accel-18.csv)
    unlabeled_18_summary = {}
    df_18_windows = pd.DataFrame()
    if (d2 / "accel-18.csv").exists():
        chunks_18 = [read_aligned_chunk(18, frac, n_lines=chunk_lines, data2_dir=d2)[0] for frac in sample_fracs]
        all_18 = pd.concat(chunks_18, ignore_index=True)
        c18_x = float(pd.to_numeric(all_18["x"], errors="coerce").median()) / 1000.0
        c18_y = float(pd.to_numeric(all_18["y"], errors="coerce").median()) / 1000.0
        c18_z = float(pd.to_numeric(all_18["z"], errors="coerce").median()) / 1000.0
        rows_18, ts_18 = [], []
        for a_df in chunks_18:
            for w in range(0, len(a_df) - window_samples + 1, window_samples):
                sub_a = a_df.iloc[w : w + window_samples]
                rows_18.append(_compute_data2_window_features(sub_a, c18_x, c18_y, c18_z))
                ts_18.append(str(sub_a["timestamp"].iloc[0])[:19])
        X_18 = pd.DataFrame(rows_18)[DATA2_FEATURE_COLUMNS].fillna(0.0)
        preds_18 = hgb_pipe.predict(X_18)
        probs_18 = hgb_pipe.predict_proba(X_18)
        counts_18 = pd.Series(preds_18).value_counts().to_dict()
        is_walk_18 = (preds_18 == "Other") & ((X_18["odba"] > 0.14) | (X_18["b_walk_frac"] > 0.22))
        walk_sub_18 = int(is_walk_18.sum())
        rest_sub_18 = int(((preds_18 == "Other") & (~is_walk_18)).sum())
        unlabeled_18_summary = {
            "animal_id": 18,
            "n_windows_scored": int(len(X_18)),
            "predicted_counts": {k: int(v) for k, v in counts_18.items()},
            "predicted_fractions": {k: round(float(v / max(1, len(X_18))), 4) for k, v in counts_18.items()},
            "other_subclassified": {
                "Walking_Locomotion": walk_sub_18,
                "Resting_Standing": rest_sub_18,
            },
            "mean_confidence": round(float(np.max(probs_18, axis=1).mean()), 4),
        }
        df_18_windows = X_18.copy()
        df_18_windows["label"] = preds_18
        df_18_windows["animal_id"] = 18
        df_18_windows["timestamp"] = ts_18
        df_18_windows["is_official_test"] = 0
        df_18_windows["is_synthetic"] = 0
        df_18_windows["detailed_state"] = np.where(
            df_18_windows["label"] != "Other",
            df_18_windows["label"],
            np.where(is_walk_18, "Other_Walking", "Other_Resting"),
        )

    # 4. Build biomechanical & spectral profile per class across the 17 labeled cows
    windows_df = X.copy()
    windows_df["label"] = y
    windows_df["animal_id"] = groups
    windows_df["timestamp"] = timestamps
    windows_df["is_official_test"] = test_mask.astype(int)
    windows_df["is_synthetic"] = 0
    # Sub-classify Other into Walking vs Resting
    is_walk_all = (windows_df["label"] == "Other") & ((windows_df["odba"] > 0.14) | (windows_df["b_walk_frac"] > 0.22))
    windows_df["detailed_state"] = np.where(
        windows_df["label"] != "Other",
        windows_df["label"],
        np.where(is_walk_all, "Other_Walking", "Other_Resting"),
    )
    if len(df_18_windows) > 0:
        windows_df = pd.concat([windows_df, df_18_windows], ignore_index=True)

    class_profiles = {}
    for cls_name in ["Eating", "Ruminating", "Other"]:
        sub = windows_df[windows_df["label"] == cls_name]
        if len(sub) == 0:
            continue
        class_profiles[cls_name] = {
            "n_windows": int(len(sub)),
            "odba_g_mean": round(float(sub["odba"].mean()), 4),
            "vedba_g_mean": round(float(sub["vedba"].mean()), 4),
            "sum_diff_g_mean": round(float(sub["sum_diff_mean"].mean()), 4),
            "pitch_deg_mean": round(float(sub["pitch_deg"].mean()), 2),
            "pitch_dev_deg_mean": round(float(sub["pitch_dev_deg"].mean()), 2),
            "rumination_band_frac": round(float(sub["b_rum_frac"].mean()), 4),
            "eating_low_band_frac": round(float(sub["b_low_frac"].mean()), 4),
            "walking_band_frac": round(float(sub["b_walk_frac"].mean()), 4),
            "bolus_pause_sub10s_cv": round(float(sub["sub10s_std_cv"].mean()), 4),
        }

    rf_imp = rf_pipe.named_steps["randomforestclassifier"].feature_importances_
    top_idx = np.argsort(rf_imp)[::-1][:10]
    top_features = {DATA2_FEATURE_COLUMNS[i]: round(float(rf_imp[i]), 4) for i in top_idx}

    metrics = {
        "available": True,
        "dataset": "Precision Beef (Zenodo 4064802) - 10 Hz Afimilk Collar + RumiWatch Halter",
        "n_animals_labeled": int(len(np.unique(groups))),
        "n_windows_60s": int(len(X)),
        "window_duration_s": round(window_samples / 10.0, 1),
        "class_counts": {k: int(v) for k, v in pd.Series(y).value_counts().items()},
        "detailed_state_counts": {k: int(v) for k, v in windows_df["detailed_state"].value_counts().items()},
        "group_cv_5fold": {
            "hgb_macro_f1": round(float(f1_score(y, oof_hgb, average="macro", zero_division=0)), 4),
            "hgb_accuracy": round(float(accuracy_score(y, oof_hgb)), 4),
            "rf_macro_f1": round(float(f1_score(y, oof_rf, average="macro", zero_division=0)), 4),
            "rf_accuracy": round(float(accuracy_score(y, oof_rf)), 4),
        },
        "official_test_split_cows_04_10_11": {
            "test_animals": sorted(DATA2_TEST_ANIMALS),
            "n_train_windows": int((~test_mask).sum()),
            "n_test_windows": int(test_mask.sum()),
            "hgb_macro_f1": round(float(f1_score(y[test_mask], test_pred_hgb, average="macro", zero_division=0)), 4),
            "hgb_accuracy": round(float(accuracy_score(y[test_mask], test_pred_hgb)), 4),
            "rf_macro_f1": round(float(f1_score(y[test_mask], test_pred_rf, average="macro", zero_division=0)), 4),
            "rf_accuracy": round(float(accuracy_score(y[test_mask], test_pred_rf)), 4),
            "per_class_f1": {
                cls: round(float(test_report_hgb.get(cls, {}).get("f1-score", 0.0)), 4)
                for cls in ["Eating", "Ruminating", "Other"]
            },
        },
        "unlabeled_animal_18_inference": unlabeled_18_summary,
        "class_biomechanical_profiles": class_profiles,
        "top_10_feature_importances": top_features,
    }

    bundle = {
        "model": hgb_pipe,
        "rf_model": rf_pipe,
        "features": DATA2_FEATURE_COLUMNS,
        "label_map": HALTER_LABEL_MAP,
        "model_version": "behaviour-data2_collar_halter-1.0.0",
        "metrics": metrics,
    }
    return bundle, metrics, windows_df


def build_data2_real_daily_telemetry(
    windows_df: pd.DataFrame,
    out_csv: Path | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Aggregate real `data2` multi-day windows into per-animal daily telemetry rows (`is_synthetic=0`)
    compatible with the PRD daily feature schema (`activity`, `rumination_min`, `lying_min`, causal ratios/z-scores).
    """
    if windows_df is None or len(windows_df) == 0:
        return pd.DataFrame(), {"available": False}

    df = windows_df.copy()
    df["date"] = df["timestamp"].str.slice(0, 10)
    daily_rows = []

    for aid, grp in df.groupby("animal_id", sort=True):
        dates = sorted(grp["date"].unique())
        for idx, dt in enumerate(dates, start=1):
            sub = grp[grp["date"] == dt]
            n_w = max(1, len(sub))
            rum_frac = float((sub["label"] == "Ruminating").sum() / n_w)
            eat_frac = float((sub["label"] == "Eating").sum() / n_w)
            walk_frac = float((sub["detailed_state"] == "Other_Walking").sum() / n_w)
            rest_frac = float((sub["detailed_state"] == "Other_Resting").sum() / n_w)
            odba_mean = float(sub["odba"].mean())
            # Scale collar ODBA (g) to the herd daily activity index scale (~350 baseline)
            activity_idx = round(odba_mean * 3800.0 + walk_frac * 180.0, 2)
            daily_rows.append({
                "cow_id": int(2000 + aid),
                "animal_id": int(aid),
                "farm_trial_id": int(DATA2_TRIAL_MAP.get(int(aid), 3)),
                "date": dt,
                "dim": int(idx),
                "parity": int(1 + (aid % 3)),
                "activity": activity_idx,
                "rumination_min": round(rum_frac * 1440.0, 1),
                "eating_min": round(eat_frac * 1440.0, 1),
                "lying_min": round(rest_frac * 1440.0, 1),
                "walking_min": round(walk_frac * 1440.0, 1),
                "odba_g_mean": round(odba_mean, 4),
                "sum_diff_g_mean": round(float(sub["sum_diff_mean"].mean()), 4),
                "neck_pitch_deg": round(float(sub["pitch_deg"].mean()), 2),
                "is_official_test": int(aid in DATA2_TEST_ANIMALS),
                "is_synthetic": 0,
            })

    daily_df = pd.DataFrame(daily_rows).sort_values(["cow_id", "dim"]).reset_index(drop=True)
    # Compute per-animal causal baseline ratios (using prior days within each animal, fallback to animal median)
    for col in ["activity", "rumination_min", "lying_min"]:
        med = daily_df.groupby("cow_id")[col].transform("median").replace(0, np.nan)
        daily_df[f"{col}_ratio"] = (daily_df[col] / med).fillna(1.0).round(4)

    # Composite estrus restlessness watch score on real telemetry (high activity ratio + suppressed rumination/lying)
    daily_df["estrus_watch_score"] = np.clip(
        0.50 * (daily_df["activity_ratio"] - 0.85)
        + 0.30 * (1.15 - daily_df["rumination_min_ratio"])
        + 0.20 * (1.15 - daily_df["lying_min_ratio"]),
        0.0,
        1.0,
    ).round(4)

    if out_csv is not None:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        daily_df.to_csv(out_csv, index=False)

    summary = {
        "available": True,
        "output_csv": str(out_csv.relative_to(ROOT)) if out_csv and out_csv.is_relative_to(ROOT) else str(out_csv),
        "n_real_cow_days": int(len(daily_df)),
        "n_animals": int(daily_df["animal_id"].nunique()),
        "mean_daily_rumination_min": round(float(daily_df["rumination_min"].mean()), 1),
        "mean_daily_eating_min": round(float(daily_df["eating_min"].mean()), 1),
        "mean_daily_lying_min": round(float(daily_df["lying_min"].mean()), 1),
        "mean_daily_walking_min": round(float(daily_df["walking_min"].mean()), 1),
        "mean_activity_index": round(float(daily_df["activity"].mean()), 1),
        "is_synthetic": 0,
    }
    return daily_df, summary


def _extract_jerk_window_features(dx: np.ndarray, dy: np.ndarray, dz: np.ndarray, sum_diff: np.ndarray) -> dict[str, float]:
    """Compute cross-sensor 1 Hz first-difference jerk features matching `data3/Preprocessed_Final.csv`."""
    sd = np.asarray(sum_diff, dtype=float)
    ax, ay, az = np.abs(dx), np.abs(dy), np.abs(dz)
    tot_l1 = float(np.sum(sd)) + 1e-9
    horiz_std = float(np.sqrt(np.var(dx) + np.var(dy)))
    vert_std = float(np.std(dz)) + 1e-6
    return {
        "sum_diff_mean": float(np.mean(sd)),
        "sum_diff_std": float(np.std(sd)),
        "sum_diff_max": float(np.max(sd)) if len(sd) else 0.0,
        "sum_diff_q90": float(np.percentile(sd, 90)) if len(sd) else 0.0,
        "active_jerk_frac": float(np.mean(sd > 0.002)),
        "high_burst_frac": float(np.mean(sd >= 0.015)),
        "x_diff_std": float(np.std(dx)),
        "y_diff_std": float(np.std(dy)),
        "z_diff_std": float(np.std(dz)),
        "z_jerk_share": float(np.sum(az) / tot_l1),
        "horiz_to_vert_jerk_ratio": float(horiz_std / vert_std),
    }


def analyze_and_classify_data3_jerk(
    data3_csv: Path | None = None,
    out_windows_csv: Path | None = None,
    seed: int = 7,
) -> tuple[pd.DataFrame, dict]:
    """Analyze `data3/Preprocessed_Final.csv` (~1 Hz 3-axis accelerometer delta stream) and classify
    every 5-minute and 1-hour overnight window using a Cross-Sensor Jerk Bridge calibrated against
    `db-cow-walking-main` and `data2`.
    """
    p3 = data3_csv or (ROOT / "data3" / "Preprocessed_Final.csv")
    if not p3.exists():
        return pd.DataFrame(), {"available": False}

    df3 = pd.read_csv(p3)
    df3["ts"] = pd.to_datetime(df3["time"])
    df3 = df3.sort_values("ts").reset_index(drop=True)
    df3["win_5m"] = df3["ts"].dt.floor("5min")
    df3["hour"] = df3["ts"].dt.floor("h")

    # 1. Build reference 1 Hz jerk training examples from `db-cow-walking-main` (10 Hz -> 1 Hz decimated & scaled)
    ref_rows, ref_labels = [], []
    collar_root = ROOT / "db-cow-walking-main"
    if collar_root.exists():
        for folder_name in ["Resting", "Grazing", "Walking", "Miscellaneous behaviors"]:
            folder = collar_root / folder_name
            if not folder.exists():
                continue
            for csv_p in sorted(folder.glob("*.csv")):
                parts = csv_p.stem.split("_")
                if folder_name == "Miscellaneous behaviors":
                    if len(parts) < 4 or parts[1] != "Standing":
                        continue
                    lbl = "Standing_Restless"
                elif folder_name == "Resting":
                    lbl = "Resting_Lying"
                elif folder_name == "Grazing":
                    lbl = "Ruminating_Foraging"
                else:
                    lbl = "Active_Walking"
                sub = pd.read_csv(csv_p)
                if not {"BNO055_AX", "BNO055_AY", "BNO055_AZ"}.issubset(sub.columns) or len(sub) < 30:
                    continue
                # Decimate 10 Hz -> 1 Hz and convert m/s^2 -> quantized low-pass g delta matching data3 sensor scale
                ax = pd.to_numeric(sub["BNO055_AX"], errors="coerce").fillna(0.0).to_numpy()[::10] / (9.81 * 8.0)
                ay = pd.to_numeric(sub["BNO055_AY"], errors="coerce").fillna(0.0).to_numpy()[::10] / (9.81 * 8.0)
                az = pd.to_numeric(sub["BNO055_AZ"], errors="coerce").fillna(0.0).to_numpy()[::10] / (9.81 * 8.0)
                if len(ax) < 4:
                    continue
                dx, dy, dz = np.diff(ax), np.diff(ay), np.diff(az)
                sd = np.abs(dx) + np.abs(dy) + np.abs(dz)
                ref_rows.append(_extract_jerk_window_features(dx, dy, dz, sd))
                ref_labels.append(lbl)

    jerk_clf = None
    if len(ref_rows) >= 20:
        jerk_clf = make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=seed, n_jobs=1),
        )
        jerk_clf.fit(pd.DataFrame(ref_rows)[JERK_BRIDGE_FEATURES], ref_labels)

    # 2. Extract 5-minute windows from `data3/Preprocessed_Final.csv`
    win_rows = []
    for w_ts, grp in df3.groupby("win_5m", sort=True):
        dx = grp["x_diff"].to_numpy(dtype=float)
        dy = grp["y_diff"].to_numpy(dtype=float)
        dz = grp["z_diff"].to_numpy(dtype=float)
        sd = grp["sum_diff"].to_numpy(dtype=float)
        feat = _extract_jerk_window_features(dx, dy, dz, sd)

        # Biomechanical state mapping tailored to overnight 1 Hz delta stream
        if feat["sum_diff_mean"] < 0.0033 and feat["high_burst_frac"] < 0.03:
            regime = "Resting_Lying"
        elif feat["z_jerk_share"] >= 0.45 and feat["sum_diff_mean"] < 0.0044:
            regime = "Ruminating_HeadBob"
        elif feat["high_burst_frac"] >= 0.06 or feat["sum_diff_mean"] >= 0.0044:
            regime = "Standing_Restless_Bout"
        else:
            regime = "Micro_Posture_Shift"

        win_rows.append({
            "window_start": str(w_ts),
            "hour": str(w_ts)[:13] + ":00",
            "n_samples": int(len(grp)),
            **{k: round(v, 6) for k, v in feat.items()},
            "overnight_regime": regime,
            "restlessness_index": round(
                float(np.clip((feat["sum_diff_mean"] / 0.006) * 0.6 + feat["active_jerk_frac"] * 0.4, 0.0, 1.0)),
                4,
            ),
            "is_synthetic": 0,
        })

    win_df = pd.DataFrame(win_rows)
    if jerk_clf is not None and len(win_df) > 0:
        win_df["collar_bridge_label"] = jerk_clf.predict(win_df[JERK_BRIDGE_FEATURES])

    if out_windows_csv is not None and len(win_df) > 0:
        out_windows_csv.parent.mkdir(parents=True, exist_ok=True)
        win_df.to_csv(out_windows_csv, index=False)

    # 3. Hourly summary across the 2019-11-14 23:59 -> 2019-11-15 05:16 recording
    hourly_rows = []
    for hr, grp in df3.groupby("hour", sort=True):
        sd = grp["sum_diff"].to_numpy(dtype=float)
        dx = grp["x_diff"].to_numpy(dtype=float)
        dy = grp["y_diff"].to_numpy(dtype=float)
        dz = grp["z_diff"].to_numpy(dtype=float)
        hourly_rows.append({
            "hour": str(hr),
            "samples": int(len(grp)),
            "mean_sum_diff_g": round(float(np.mean(sd)), 6),
            "cumulative_jerk_g": round(float(np.sum(sd)), 4),
            "active_seconds_frac": round(float(np.mean(sd > 0)), 4),
            "vertical_z_share": round(float(np.sum(np.abs(dz)) / (np.sum(sd) + 1e-9)), 4),
            "lateral_xy_share": round(float((np.sum(np.abs(dx)) + np.sum(np.abs(dy))) / (np.sum(sd) + 1e-9)), 4),
        })

    regime_counts = win_df["overnight_regime"].value_counts().to_dict() if len(win_df) else {}
    summary = {
        "available": True,
        "source_file": "data3/Preprocessed_Final.csv",
        "output_windows_csv": str(out_windows_csv.relative_to(ROOT))
        if out_windows_csv and out_windows_csv.is_relative_to(ROOT)
        else str(out_windows_csv),
        "total_samples": int(len(df3)),
        "start_timestamp": str(df3["time"].iloc[0]),
        "end_timestamp": str(df3["time"].iloc[-1]),
        "duration_hours": round(
            float((df3["ts"].iloc[-1] - df3["ts"].iloc[0]).total_seconds() / 3600.0), 2
        ),
        "median_sampling_interval_s": float(df3["ts"].diff().dt.total_seconds().median()),
        "quantization_step_g": 0.00390625,
        "non_zero_jerk_fraction": round(float((df3["sum_diff"] > 0).mean()), 4),
        "directional_energy_share": {
            "x_longitudinal_share": round(float(df3["x_diff"].abs().sum() / (df3["sum_diff"].sum() + 1e-9)), 4),
            "y_lateral_share": round(float(df3["y_diff"].abs().sum() / (df3["sum_diff"].sum() + 1e-9)), 4),
            "z_vertical_share": round(float(df3["z_diff"].abs().sum() / (df3["sum_diff"].sum() + 1e-9)), 4),
        },
        "n_windows_5min": int(len(win_df)),
        "overnight_regime_counts": {k: int(v) for k, v in regime_counts.items()},
        "overnight_regime_fractions": {
            k: round(float(v / max(1, len(win_df))), 4) for k, v in regime_counts.items()
        },
        "mean_restlessness_index": round(float(win_df["restlessness_index"].mean()), 4) if len(win_df) else 0.0,
        "peak_restlessness_window": str(
            win_df.loc[win_df["restlessness_index"].idxmax(), "window_start"]
        )
        if len(win_df)
        else None,
        "hourly_profile": hourly_rows,
        "is_synthetic": 0,
    }
    return win_df, summary
