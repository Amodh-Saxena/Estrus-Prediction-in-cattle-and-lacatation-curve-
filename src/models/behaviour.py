"""Random-forest behaviour classifier trained from real collar IMU clips and synthetic IMU summaries (PRD Section 6.1).

Uses:
1. Real 10 Hz IMU collar CSV clips in db-cow-walking-main/ (Grazing, Walking, Resting + Miscellaneous behaviors including Standing).
2. 5-second windows at 10 Hz (50 samples) and full-clip statistical + spectral features.
3. Synthetic IMU clip summaries in synthetic_imu_clips_441.csv / data/synthetic_cattle.db.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline

ROOT = Path(__file__).resolve().parents[2]
SENSOR_COLUMNS = [
    "BNO055_ARX", "BNO055_ARY", "BNO055_ARZ",
    "BNO055_AX", "BNO055_AY", "BNO055_AZ",
    "BNO055_GX", "BNO055_GY", "BNO055_GZ",
    "BNO055_Q0", "BNO055_Q1", "BNO055_Q2", "BNO055_Q3",
    "MPU9250_AX", "MPU9250_AY", "MPU9250_AZ",
    "MPU9250_GX", "MPU9250_GY", "MPU9250_GZ",
]
CLASSES_3 = {"Grazing", "Walking", "Resting"}
CLASSES_4 = {"Grazing", "Walking", "Resting", "Standing"}


def _signal_features(frame: pd.DataFrame, sample_hz: int = 10) -> dict[str, float]:
    values: dict[str, float] = {}
    for column in SENSOR_COLUMNS:
        if column not in frame:
            continue
        signal = pd.to_numeric(frame[column], errors="coerce").dropna().to_numpy()
        if not len(signal):
            continue
        values[f"{column}_mean"] = float(np.mean(signal))
        values[f"{column}_std"] = float(np.std(signal))
        values[f"{column}_abs_mean"] = float(np.mean(np.abs(signal)))
        values[f"{column}_range"] = float(np.ptp(signal))
        values[f"{column}_q75_q25"] = float(np.percentile(signal, 75) - np.percentile(signal, 25))
        values[f"{column}_diff_std"] = float(np.std(np.diff(signal))) if len(signal) > 1 else 0.0

    # 1. Body-frame acceleration magnitude, ODBA, vertical/horizontal ratio, stride frequency & gait regularity
    if {"BNO055_AX", "BNO055_AY", "BNO055_AZ"}.issubset(frame.columns):
        ax = pd.to_numeric(frame["BNO055_AX"], errors="coerce").fillna(0).to_numpy()
        ay = pd.to_numeric(frame["BNO055_AY"], errors="coerce").fillna(0).to_numpy()
        az = pd.to_numeric(frame["BNO055_AZ"], errors="coerce").fillna(0).to_numpy()
        acc_mag = np.sqrt(ax**2 + ay**2 + az**2) / 9.81
        values["acc_mag_mean"] = float(np.mean(acc_mag))
        values["acc_mag_std"] = float(np.std(acc_mag))

        # Overall Dynamic Body Acceleration (ODBA) & Vector Dynamic Body Acceleration (VeDBA)
        dyn_x = ax - np.mean(ax)
        dyn_y = ay - np.mean(ay)
        dyn_z = az - np.mean(az)
        values["odba_mean"] = float(np.mean(np.abs(dyn_x) + np.abs(dyn_y) + np.abs(dyn_z)) / 9.81)
        values["vedba_mean"] = float(np.mean(np.sqrt(dyn_x**2 + dyn_y**2 + dyn_z**2)) / 9.81)

        # Horizontal vs vertical movement ratio (forward walking propulsion vs vertical grazing head sweeps)
        horiz_std = float(np.sqrt(np.var(ax) + np.var(ay)))
        vert_std = float(np.std(az)) + 1e-6
        values["horiz_vert_acc_ratio"] = float(horiz_std / vert_std)

        if len(acc_mag) >= 10:
            centered = acc_mag - np.mean(acc_mag)
            fft_vals = np.abs(np.fft.rfft(centered))
            freqs = np.fft.rfftfreq(len(centered), d=1.0 / sample_hz)
            if len(freqs) > 1:
                dom_idx = int(np.argmax(fft_vals[1:]) + 1)
                values["dom_freq_hz"] = float(freqs[dom_idx])
                # Walking band (1.0 - 2.5 Hz) vs Grazing band (0.3 - 1.0 Hz) power ratio
                walk_band = float(np.sum(fft_vals[(freqs >= 1.0) & (freqs <= 2.5)] ** 2))
                graze_band = float(np.sum(fft_vals[(freqs >= 0.3) & (freqs < 1.0)] ** 2)) + 1e-6
                values["walk_to_graze_spectral_ratio"] = float(walk_band / graze_band)
            else:
                values["dom_freq_hz"] = 0.0
                values["walk_to_graze_spectral_ratio"] = 0.0

            # Gait stride regularity via normalized autocorrelation in the walking stride lag (0.4s..1.0s = 4..10 samples)
            norm = float(np.sum(centered**2))
            if norm > 1e-8 and len(centered) > 12:
                ac_lags = [
                    float(np.sum(centered[:-lag] * centered[lag:]) / norm)
                    for lag in range(4, min(11, len(centered) - 1))
                ]
                values["stride_regularity_autocorr"] = float(np.max(ac_lags))
            else:
                values["stride_regularity_autocorr"] = 0.0
        else:
            values["dom_freq_hz"] = 0.0
            values["walk_to_graze_spectral_ratio"] = 0.0
            values["stride_regularity_autocorr"] = 0.0

    # 2. World-frame linear acceleration (BNO055_ARX, ARY, ARZ)
    if {"BNO055_ARX", "BNO055_ARY", "BNO055_ARZ"}.issubset(frame.columns):
        arx = pd.to_numeric(frame["BNO055_ARX"], errors="coerce").fillna(0).to_numpy()
        ary = pd.to_numeric(frame["BNO055_ARY"], errors="coerce").fillna(0).to_numpy()
        arz = pd.to_numeric(frame["BNO055_ARZ"], errors="coerce").fillna(0).to_numpy()
        world_horiz = np.sqrt(arx**2 + ary**2)
        values["world_horiz_acc_mean"] = float(np.mean(world_horiz))
        values["world_horiz_acc_std"] = float(np.std(world_horiz))
        values["world_vert_acc_std"] = float(np.std(arz))

    # 3. Quaternion neck/head orientation (pitch & roll from BNO055_Q0..Q3)
    if {"BNO055_Q0", "BNO055_Q1", "BNO055_Q2", "BNO055_Q3"}.issubset(frame.columns):
        q0 = pd.to_numeric(frame["BNO055_Q0"], errors="coerce").fillna(0).to_numpy(dtype=float)
        q1 = pd.to_numeric(frame["BNO055_Q1"], errors="coerce").fillna(0).to_numpy(dtype=float)
        q2 = pd.to_numeric(frame["BNO055_Q2"], errors="coerce").fillna(0).to_numpy(dtype=float)
        q3 = pd.to_numeric(frame["BNO055_Q3"], errors="coerce").fillna(0).to_numpy(dtype=float)
        q_norm = np.maximum(np.sqrt(q0**2 + q1**2 + q2**2 + q3**2), 1e-8)
        qw, qx, qy, qz = q0 / q_norm, q1 / q_norm, q2 / q_norm, q3 / q_norm
        sin_pitch = np.clip(2.0 * (qw * qy - qz * qx), -1.0, 1.0)
        pitch_deg = np.degrees(np.arcsin(sin_pitch))
        roll_deg = np.degrees(np.arctan2(2.0 * (qw * qx + qy * qz), 1.0 - 2.0 * (qx**2 + qy**2)))
        values["head_pitch_deg_mean"] = float(np.mean(pitch_deg))
        values["head_pitch_deg_std"] = float(np.std(pitch_deg))
        values["head_roll_deg_std"] = float(np.std(roll_deg))

    # 4. Angular velocity (Gyroscope magnitude)
    if {"BNO055_GX", "BNO055_GY", "BNO055_GZ"}.issubset(frame.columns):
        gx = pd.to_numeric(frame["BNO055_GX"], errors="coerce").fillna(0).to_numpy()
        gy = pd.to_numeric(frame["BNO055_GY"], errors="coerce").fillna(0).to_numpy()
        gz = pd.to_numeric(frame["BNO055_GZ"], errors="coerce").fillna(0).to_numpy()
        gyro_mag = np.sqrt(gx**2 + gy**2 + gz**2)
        values["gyro_std"] = float(np.std(gyro_mag))
        values["gyro_mean"] = float(np.mean(gyro_mag))

    values["n_samples"] = int(len(frame))
    values["duration_s"] = round(float(len(frame) / sample_hz), 1)
    return values


def extract_clip_features(path: Path) -> dict[str, float]:
    frame = pd.read_csv(path)
    return _signal_features(frame, sample_hz=10)


def load_real_clips(
    root: Path | None = None,
    include_standing: bool = False,
    include_all_misc: bool = False,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, list[str]]:
    """Load real collar CSV clips from db-cow-walking-main."""
    root = root or (ROOT / "db-cow-walking-main")
    rows, labels, groups, clip_ids = [], [], [], []
    for path in sorted(root.rglob("*.csv")):
        parts = path.stem.split("_")
        if len(parts) < 4:
            continue
        folder = path.parent.name
        if folder in CLASSES_3:
            label = folder
        elif folder == "Miscellaneous behaviors":
            raw_label = parts[1]
            if include_all_misc:
                label = raw_label
            elif include_standing and raw_label == "Standing":
                label = "Standing"
            else:
                continue
        else:
            continue

        cow = parts[2]
        rows.append(extract_clip_features(path))
        labels.append(label)
        groups.append(cow)
        clip_ids.append(path.stem)

    if not rows:
        raise FileNotFoundError(f"No labeled clips found under {root}")
    return pd.DataFrame(rows).fillna(0.0), np.asarray(labels), np.asarray(groups), clip_ids


def load_5s_windows(
    root: Path | None = None,
    window_samples: int = 50,
    max_windows_per_clip: int = 6,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Extract 5-second (50-sample at 10 Hz) IMU windows per PRD Section 6.1."""
    root = root or (ROOT / "db-cow-walking-main")
    rows, labels, groups = [], [], []
    for path in sorted(root.rglob("*.csv")):
        if path.parent.name not in CLASSES_3:
            continue
        parts = path.stem.split("_")
        if len(parts) < 4:
            continue
        label = path.parent.name
        cow = parts[2]
        frame = pd.read_csv(path)
        n = len(frame)
        if n < window_samples:
            rows.append(_signal_features(frame))
            labels.append(label)
            groups.append(cow)
            continue
        n_win = min(max_windows_per_clip, n // window_samples)
        for w in range(n_win):
            sub = frame.iloc[w * window_samples : (w + 1) * window_samples]
            rows.append(_signal_features(sub))
            labels.append(label)
            groups.append(cow)
    return pd.DataFrame(rows).fillna(0.0), np.asarray(labels), np.asarray(groups)


def evaluate_synthetic_imu_csv(csv_path: Path | None = None, seed: int = 7) -> dict:
    """Train and evaluate behaviour classification on synthetic_imu_clips_441.csv."""
    path = csv_path or (ROOT / "synthetic_imu_clips_441.csv")
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    feat_cols = ["duration_s", "acc_mag_mean", "acc_mag_std", "gyro_std", "dom_freq_hz"]
    df_3 = df[df["label"].isin(CLASSES_3)].reset_index(drop=True)

    est = make_pipeline(
        SimpleImputer(strategy="median"),
        RandomForestClassifier(
            n_estimators=200,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=1,
        ),
    )
    folds = min(5, df_3["cow_id"].nunique())
    oof_3 = cross_val_predict(est, df_3[feat_cols], df_3["label"], groups=df_3["cow_id"], cv=GroupKFold(folds), n_jobs=1)
    oof_all = cross_val_predict(est, df[feat_cols], df["label"], groups=df["cow_id"], cv=GroupKFold(folds), n_jobs=1)
    est.fit(df[feat_cols], df["label"])
    return {
        "source_csv": str(path.name),
        "n_clips_total": int(len(df)),
        "n_clips_3class": int(len(df_3)),
        "group_cv_macro_f1_3class": float(f1_score(df_3["label"], oof_3, average="macro", zero_division=0)),
        "group_cv_accuracy_3class": float(accuracy_score(df_3["label"], oof_3)),
        "group_cv_macro_f1_12class": float(f1_score(df["label"], oof_all, average="macro", zero_division=0)),
        "group_cv_accuracy_12class": float(accuracy_score(df["label"], oof_all)),
    }


def _make_rf_pipeline(seed: int = 7, rf_cfg: dict | None = None):
    cfg = rf_cfg or {}
    return make_pipeline(
        SimpleImputer(strategy="median"),
        RandomForestClassifier(
            n_estimators=int(cfg.get("n_estimators", 300)),
            min_samples_leaf=int(cfg.get("min_samples_leaf", 2)),
            class_weight=cfg.get("class_weight", "balanced_subsample"),
            random_state=int(cfg.get("random_state", seed)),
            n_jobs=1,
        ),
    )


def summarize_walking_patterns(x_df: pd.DataFrame, labels: np.ndarray, cows: np.ndarray) -> dict:
    """Summarize real collar walking gait, dynamic body acceleration, and neck posture patterns."""
    key_cols = [
        "odba_mean",
        "vedba_mean",
        "dom_freq_hz",
        "stride_regularity_autocorr",
        "world_horiz_acc_mean",
        "horiz_vert_acc_ratio",
        "head_pitch_deg_mean",
        "head_pitch_deg_std",
        "gyro_mean",
    ]
    present = [c for c in key_cols if c in x_df.columns]
    df = x_df[present].copy()
    df["label"] = labels
    df["cow_id"] = cows

    by_class = {}
    for cls, grp in df.groupby("label", sort=True):
        by_class[str(cls)] = {
            col: round(float(grp[col].mean()), 4) for col in present
        }
        by_class[str(cls)]["n_clips"] = int(len(grp))

    walk_df = df[df["label"] == "Walking"]
    by_cow_walking = {}
    for cow, grp in walk_df.groupby("cow_id", sort=True):
        by_cow_walking[str(cow)] = {
            "n_walking_clips": int(len(grp)),
            "odba_mean": round(float(grp["odba_mean"].mean()), 4) if "odba_mean" in grp else 0.0,
            "dom_freq_hz": round(float(grp["dom_freq_hz"].mean()), 4) if "dom_freq_hz" in grp else 0.0,
            "stride_regularity": round(float(grp["stride_regularity_autocorr"].mean()), 4)
            if "stride_regularity_autocorr" in grp
            else 0.0,
            "head_pitch_deg": round(float(grp["head_pitch_deg_mean"].mean()), 2)
            if "head_pitch_deg_mean" in grp
            else 0.0,
        }
    return {
        "gait_and_posture_by_behaviour": by_class,
        "per_cow_walking_gait_profile": by_cow_walking,
    }


def fit_and_score(seed: int = 7, rf_cfg: dict | None = None) -> tuple[tuple[object, list[str]], dict]:
    """Fit and score the 3-class and 4-class (with Standing) real collar models + synthetic IMU CSV."""
    x3, y3, g3, clip_ids3 = load_real_clips(include_standing=False)
    unique_groups = np.unique(g3)
    folds = min(5, len(unique_groups))
    if folds < 2:
        raise ValueError("Behaviour evaluation needs at least two distinct cow IDs.")

    estimator_3 = _make_rf_pipeline(seed=seed, rf_cfg=rf_cfg)
    gkf = GroupKFold(n_splits=folds)
    oof_3 = cross_val_predict(estimator_3, x3, y3, groups=g3, cv=gkf, n_jobs=1)

    fold_f1s, fold_accs = [], []
    for tr_idx, te_idx in gkf.split(x3, y3, groups=g3):
        m = _make_rf_pipeline(seed=seed, rf_cfg=rf_cfg)
        m.fit(x3.iloc[tr_idx], y3[tr_idx])
        pred = m.predict(x3.iloc[te_idx])
        fold_f1s.append(float(f1_score(y3[te_idx], pred, average="macro", zero_division=0)))
        fold_accs.append(float(accuracy_score(y3[te_idx], pred)))

    # Evaluate 4-class with Standing added (PRD Section 1 & 6.1)
    x4, y4, g4, _ = load_real_clips(include_standing=True)
    estimator_4 = _make_rf_pipeline(seed=seed, rf_cfg=rf_cfg)
    oof_4 = cross_val_predict(estimator_4, x4, y4, groups=g4, cv=GroupKFold(min(5, len(np.unique(g4)))), n_jobs=1)

    # Evaluate 5-second windows (50 samples at 10 Hz) per PRD Section 6.1
    xw, yw, gw = load_5s_windows()
    estimator_w = _make_rf_pipeline(seed=seed, rf_cfg=rf_cfg)
    oof_w = cross_val_predict(estimator_w, xw, yw, groups=gw, cv=GroupKFold(min(5, len(np.unique(gw)))), n_jobs=1)

    # Evaluate synthetic_imu_clips_441.csv
    synth_imu_metrics = evaluate_synthetic_imu_csv(seed=seed)
    walking_pattern_summary = summarize_walking_patterns(x4, y4, g4)

    macro_f1_3 = float(f1_score(y3, oof_3, average="macro", zero_division=0))
    acc_3 = float(accuracy_score(y3, oof_3))
    macro_f1_4 = float(f1_score(y4, oof_4, average="macro", zero_division=0))
    acc_4 = float(accuracy_score(y4, oof_4))

    metrics = {
        "source": "real db-cow-walking labeled CSV clips + synthetic_imu_clips_441.csv",
        "n_clips": int(len(y3)),
        "n_clips_with_standing": int(len(y4)),
        "n_cows": int(len(unique_groups)),
        "classes": sorted(np.unique(y3).tolist()),
        "group_cv_folds": folds,
        "group_cv_accuracy": acc_3,
        "group_cv_macro_f1": macro_f1_3,
        "fold_macro_f1_mean": float(np.mean(fold_f1s)),
        "fold_macro_f1_std": float(np.std(fold_f1s)),
        "fold_accuracy_mean": float(np.mean(fold_accs)),
        "fold_accuracy_std": float(np.std(fold_accs)),
        "with_standing_4class": {
            "classes": sorted(np.unique(y4).tolist()),
            "n_clips": int(len(y4)),
            "group_cv_accuracy": acc_4,
            "group_cv_macro_f1": macro_f1_4,
        },
        "window_5s_50samples_3class": {
            "n_windows": int(len(yw)),
            "group_cv_accuracy": float(accuracy_score(yw, oof_w)),
            "group_cv_macro_f1": float(f1_score(yw, oof_w, average="macro", zero_division=0)),
        },
        "synthetic_imu_441_csv": synth_imu_metrics,
        "walking_pattern_analysis": walking_pattern_summary,
        "acceptance_target_macro_f1": 0.85,
        "acceptance_met": bool(macro_f1_3 >= 0.85),
    }

    estimator_3.fit(x3, y3)
    return (estimator_3, list(x3.columns)), metrics
