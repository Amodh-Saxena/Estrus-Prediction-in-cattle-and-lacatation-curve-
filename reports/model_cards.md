# Model Cards — Estrus, Breeding, Behaviour and Lactation Prediction System

Generated: `2026-11-02T20:14:37.079284+00:00`

## 1. Estrus Detector (`estrus-hist_gradient_boosting-1.0.0` & `0.1.0`)
- **Data sources**: Synthetic PRD 300-cow tables (`data/daily.csv`, `data/cows.csv`) + 10-cow assets (`synthetic_daily_10cows.csv`, `synthetic_hourly_10cows.csv`, `data/synthetic_cattle.db`, `out/estrus_predictions.csv`).
- **Held-out test F1 (300 cows)**: `0.9101` (Target >= 0.85) | **PR-AUC**: `0.9550` (Target >= 0.90)
- **Activity Rule Baseline F1**: `0.8615` | **Logistic Baseline F1**: `0.8975`
- **Normal-heat Recall**: `0.9753` | **Silent-heat Recall**: `0.3019`
- **10-Cow Enriched (Hourly + Daily + Prediction CSV) F1**: `0.9333` (`PR-AUC`: `0.9815`)
- **`data4` Smid et al. (2024) Real Dairy Estrus Duration Regressor (5-Fold Cow CV)**: MAE = `1.555 h`, R² = `0.8391` (`94` SAS-valid estrus events across `45` cows).
- **`data4` Electronic Ear-Tag Temperature & Step Monitoring**: `7825` real animal-days (`Peri-estrus peak step ratio`: `1.59x`, `Peak thermal shift`: `+1.55 °C`).

## 2. Behaviour Classifier (`behaviour-random_forest-0.1.0` & `behaviour-data2_collar_halter-1.0.0`)
- **Data sources**: Real 10 Hz collar IMU CSV clips (`db-cow-walking-main`: 332 3-class clips, 356 4-class clips with `Standing`) + `data2/` (`Precision Beef Zenodo 4064802`: 18 cattle, 10 Hz Afimilk Collar + RumiWatch Halter) + `data3/Preprocessed_Final.csv` (1 Hz 3-axis jerk stream) + `synthetic_imu_clips_441.csv`.
- **5-Fold Cow-Grouped Macro F1 (`db-cow-walking`, 3 classes)**: `0.9248` (Target >= 0.85) | **Accuracy**: `0.9398`
- **4-Class with Standing Macro F1 (`db-cow-walking`)**: `0.8442` | **Accuracy**: `0.9156`
- **`data2` Precision Beef 5-Fold Cow-Grouped Macro F1 (17 cows)**: `0.8793` | **Official Test Split (`Cows 04, 10, 11`) Macro F1**: `0.9786` (`Accuracy`: `0.9865`)
- **`data3` Overnight Jerk Stream (`Preprocessed_Final.csv`)**: `65` five-minute windows classified (`Mean Restlessness`: `0.4959`).

## 3. Conception Model (`conception-calibrated_ensemble-1.0.0`)
- **Data sources**: `data/inseminations.csv` + causal baseline strengths from `data/daily.csv` (FR-F4) + out-of-fold `p_estrus` from estrus predictions.
- **Base-rate Brier**: `0.2262` (Conception rate: `34.6%`)
- **Calibrated Logistic 5-Fold CV Brier**: `0.2228` | **AUC**: `0.5691`
- **Calibrated Boosting 5-Fold CV Brier**: `0.2230` | **AUC**: `0.5732`

## 4. Lactation Model (`lactation-scaled_parity_hgb-1.0.0`)
- **Data sources**: `data/daily.csv` (DIM 5–60 early window -> DIM 61–305 forecast) + `data/cows.csv`.
- **Held-out Test MAE**: `1.616 kg/day` (`55.4%` below parity-mean curve MAE `3.620 kg/day`)
- **Remaining-Yield Error (DIM 61–305)**: `3.13%` (Target <= 5.0%) | **MAPE**: `4.80%`

## 5. Automated Leakage Checks (FR-E6)
- Disjoint cow splits: `True`
- Causal rolling baselines (no future leakage): `True`
- Baseline excludes today: `True`
- Parity curve built from training cows only: `True`

## 6. Known Limitations
- Estrus, conception, and lactation metrics are evaluated on synthetic data (`is_synthetic=1`) to validate the pipeline; real farm accuracy must be measured once real cow-day and breeding records arrive.
- Camera images in `Wagyu&Angus` are inventoried in `reports/data_inventory.json` and remain out of scope for v1.
