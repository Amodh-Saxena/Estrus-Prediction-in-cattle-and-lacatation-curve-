# PRD: Estrus, Breeding and Lactation Prediction System

Nov 2, 2026 · @Amodh

## 1. Overview

The system predicts, for every cow and day, whether she is in estrus, how likely an insemination is to succeed, and her milk yield across a 305-day lactation, so a farmer gets an estimated heat time, the best insemination day and an expected yield.

**Goals**

1. **G1, reproducible pipeline.** One command generates data, trains all models, evaluates them and writes `metrics.json`.
2. **G2, hybrid data.** Real collar data plus a documented synthetic simulator, with every synthetic row flagged `is_synthetic=1`.
3. **G3, estrus detector.** A per-cow-day probability with an alert threshold tuned on validation cows.
4. **G4, conception model.** A calibrated pregnancy probability for each insemination.
5. **G5, lactation model.** A daily milk forecast for days 61 to 305 from days 5 to 60.
6. **G6, real-data ready.** Real records drop in under the same schemas, and final results are reported on real data only.

**Non-goals for this version**

- Image or video features (cattle detection, tracking, thermal imaging).
- Deep sequence models such as LSTM or TCN.
- Genetic evaluation and heritability estimation (BLUP).
- Collar firmware, a mobile app or farm-management integrations.

**Success metrics.** Targets are proposed and should be confirmed before sign-off; status is from the synthetic run on 2 Nov 2026.

| Metric | Target | Status on synthetic data |
| --- | --- | --- |
| Estrus F1 on held-out cows | 0.85 or higher | 0.91, met |
| Estrus PR-AUC on held-out cows | 0.90 or higher | 0.961, met |
| Conception Brier score | Below the base-rate Brier of 0.226 | 0.249 (logistic) and 0.243 (boosting), not met |
| Lactation remaining-yield error (days 61 to 305) | 5% or lower | 2.4%, met |
| Behaviour macro F1 on unseen cows (3 classes) | 0.85 or higher | 0.88, met (0.67 with standing added) |
| Any metric on real farm data | To be set after the first real run | Not measured |

## 2. System architecture

Synthetic and real data enter through one table contract, and four code stages turn it into validated data, features, models and predictions.

&#91;embedded content: pipeline · 2 sources, 4 stages, 1 data contract\]

Read the drawing top to bottom; each stage writes the artifact on its right, and real farm tables replace the simulator without any code change.

**Model layer.** These components sit inside stages 2 to 4.

| Component | Reads | Produces |
| --- | --- | --- |
| Behaviour classifier | Collar IMU clips | Behaviour classes behind the activity and lying features |
| Estrus detector | Daily feature table | `p_estrus` for each cow-day |
| Conception model | Insemination rows | `p_conception` for each insemination |
| Lactation model | Milk on days 5 to 60 | Daily milk for days 61 to 305 |
| Breeding-window scheduler | The three model outputs | Alert and best insemination day; a rules layer, not built in v1 |

## 3. Data requirements

Three tables and a set of raw collar clips form the data contract; real farm data must arrive in the same columns.

**Sources**

| Source | Status | Use |
| --- | --- | --- |
| Collar clips (`db-cow-walking`) | Real, labelled: 10 Holstein cows, 441 clips, about 7.5 hours at 10 Hz | Behaviour classifier |
| Estrus-paper supplementary figures | Real, published, 3 figures and no raw data | Effect-size priors for the simulator |
| Wagyu and Angus images | Real, unlabelled: 277 pen frames, 10 aerial images with masks | Out of scope for v1 |
| Tables from `generate_data.py` | Synthetic: 300 cows, 91,500 cow-days, 837 inseminations | Develop and test the models |
| Earlier synthetic assets (hourly 10 cows, daily 10 cows, IMU clips 441, `gen_synthetic.py`) | Synthetic; listed from a screenshot, contents not reviewed | Reconcile with this spec before use |

**`cows.csv`, one row per cow**

| Column | Type | Meaning |
| --- | --- | --- |
| `cow_id` | int | Unique cow identifier |
| `parity` | int | 1, 2 or 3, where 3 means 3 or more |
| `pct_holstein` | float | Holstein fraction, 0.5 to 1.0 |
| `calving_doy` | int | Day of year of calving, 0 to 364 |
| `technician` | str | T1, T2 or T3 |
| `semen_plan` | str | sexed\_dairy, conventional\_dairy or beef |
| `g_activity`, `g_rumination`, `g_milk` | float | Latent genetic merits; synthetic only and empty in real data |
| `is_synthetic` | int | 1 for synthetic, 0 for real |

**`daily.csv`, one row per cow per day (DIM 1 to 305)**

| Column | Type | Meaning |
| --- | --- | --- |
| `cow_id`, `dim`, `parity` | int | Key and context |
| `activity` | float, nullable | Daily activity index |
| `rumination_min`, `lying_min` | float, nullable | Minutes per day |
| `body_temp_c` | float, nullable | Body temperature in degrees Celsius |
| `milk_kg` | float | Milk yield in kg per day |
| `estrus_day` | int | 1 if an observed heat occurred that day |
| `estrus_duration_h` | float | Heat duration in hours, 0 when none |
| `pregnant` | int | 1 from the conception day onward |
| `is_synthetic` | int | 1 for synthetic, 0 for real |

**`inseminations.csv`, one row per insemination**

| Column | Type | Meaning |
| --- | --- | --- |
| `cow_id`, `dim`, `parity` | int | Key and context |
| `semen_type`, `technician` | str | As in `cows.csv` |
| `estrus_activity_strength` | float | Heat-day activity divided by baseline |
| `estrus_rumination_strength` | float | Heat-day rumination divided by baseline |
| `estrus_duration_h` | float | Heat duration in hours |
| `silent_heat` | int | 1 for a weak or silent heat |
| `p_conception_true` | float | True simulator probability; synthetic only |
| `conceived` | int | 1 if the insemination led to pregnancy |
| `is_synthetic` | int | 1 for synthetic, 0 for real |

**Collar clips.** One file per clip with `cow_id`, `clip_id`, a label of walking, grazing, resting or standing, and a 10 Hz sample matrix. The exact axes and file format must be confirmed from the dataset.

**Validation rules (FR-D1).** Loading fails with a clear message if any of these break:

- `(cow_id, dim)` is unique in `daily.csv`, `dim` is between 1 and 305, and `parity` is 1, 2 or 3.
- `estrus_day`, `pregnant`, `conceived` and `silent_heat` contain only 0 or 1, and `pregnant` never returns to 0 within a cow.
- Sensor values fall in plausible ranges and no more than 10% of any sensor column is missing.
- `is_synthetic` is present everywhere, and the `g_*` columns are empty whenever it is 0.

## 4. Synthetic data generator spec

`generate_data.py` is the single source of synthetic data; it must be deterministic, parameterised and checked against the targets below.

**Requirements**

- **FR-S1.** Deterministic: default seed 42, and the same arguments produce identical CSVs.
- **FR-S2.** CLI `python generate_data.py <n_cows>` with a default of 300, writing to `data/`.
- **FR-S3.** Every row carries `is_synthetic=1`.
- **FR-S4.** Each cow gets a 305-day lactation; the estrus loop runs until she conceives or day 305.
- **FR-S5.** Move the constants below into a config file; today they are hard-coded.

**Cow-level draws (as built)**

| Quantity | Distribution |
| --- | --- |
| Parity (1, 2, 3+) | Probabilities 0.38, 0.30, 0.32 |
| Holstein fraction | Normal(0.88, 0.10), clipped to 0.5 to 1.0 |
| Calving day of year | Uniform integer, 0 to 364 |
| Genetic merits (activity, rumination, milk) | Standard normal each |
| Wood a | Parity base 16, 20, 23 times exp(0.08 × milk merit + Normal(0, 0.06)) |
| Wood b | Parity base 0.18, 0.22, 0.24 plus Normal(0, 0.015) |
| Wood c | Parity base 0.0020, 0.0030, 0.0034 plus Normal(0, 0.0002) |
| Baselines | Activity Normal(100, 12) + 4 × merit; rumination Normal(480, 35) + 15 × merit; lying Normal(600, 50); body temperature Normal(38.6, 0.15) |
| Technician | Uniform over T1, T2, T3 |
| Semen plan | Parity 3 draws 0.35 sexed, 0.40 conventional, 0.25 beef; parities 1 and 2 use sexed |
| Voluntary waiting period | Uniform integer, 50 to 74 days |
| First estrus after calving | Uniform integer, 22 to 47 days |

**Estrus and insemination events (as built)**

| Quantity | Rule |
| --- | --- |
| Cycle length | Normal(21, 2.5), rounded and clipped to 17 to 26 days |
| Silent heat | Probability 0.15; deviations shrink to 25% of a normal heat |
| Duration (h) | Normal(13 + 0.8 × activity merit, 3.5), clipped to 4 to 24 |
| Activity strength | Lognormal with median 1.9 + 0.12 × merit and sigma 0.25, clipped to 1.0 to 4.5 |
| Rumination strength | Normal(0.80 − 0.03 × merit, 0.08), clipped to 0.45 to 1.0 |
| Insemination | At a heat on or after the waiting period if not silent; a silent heat is inseminated with probability 0.3 |

Conception follows this log-odds, where s\_act and s\_rum are the heat-day activity and rumination strengths:

```latex
\mathrm{logit}\,P = -0.35 + 0.55\ln s_{act} - 1.0\,(1 - s_{rum}) - 0.175\,(parity - 1) - 0.25\,[\text{sexed}] + 0.15\,[\text{beef}] - 0.004\,\max(DIM - 100,\,0) + \tau
```

The technician term τ is +0.20 for T1, −0.05 for T3 and 0 for T2.

**Daily signals (as built)**

- Each sensor is its baseline plus AR(1) noise: activity (coefficient 0.5, SD 5), rumination (0.5, SD 18), lying (0.5, SD 30), temperature (0.4, SD 0.08). Milk gets 4% multiplicative noise.
- On a heat day activity is multiplied by s\_act and rumination by s\_rum; lying time by 1 − 0.12 × (s\_act − 1) / 1.5; temperature rises by 0.25 × (s\_act − 1); milk falls by 3% × (s\_act − 1).
- The day before a heat, activity rises by 15% of the next heat's excess.
- Pregnancy lowers milk by up to 12%, ramping in from 120 days after conception over 150 days.
- 2% of cow-days lose all four sensor readings.

**Validation targets (FR-S6).** The generator's own check must pass for 300 cows at seed 42.

| Statistic | Expected | Observed |
| --- | --- | --- |
| Cow-days | 91,500 | 91,500 |
| Estrus-day rate | About 1.6% | 1.64% |
| Conception rate | About 35% | 34.6% |
| Mean milk | About 34.5 kg a day | 34.5 kg a day |
| Mean cycle length | 21 days, within 1 | Test to add |
| Missing sensor rate | About 2% | Test to add |

**Extensions to build (FR-S7).** Hourly bins (24 per day) that preserve the daily totals while concentrating activity in the heat window, and 10 Hz IMU clips matching the collar dataset's classes. The earlier `gen_synthetic.py` assets must be reviewed and reconciled with these rules first.

**Known limits.** The activity spike is strong, conception effects are weak, and there is no seasonality, heat stress or herd effect. Scores on this data are a pipeline test, not a forecast of farm accuracy.

## 5. Feature engineering spec

The estrus detector reads one row per cow per day, and every feature uses only that day and earlier days so it works in real time.

**Requirements**

- **FR-F1, causal baselines.** A cow's baseline for day t is the median of days t−14 to t−1, with at least 5 days present. Today's value is never inside its own baseline.
- **FR-F2, warm-up.** Drop rows with DIM below 15.
- **FR-F3, missing data.** Leave sensor gaps as missing for boosting; for logistic regression use median imputation and standardisation.
- **FR-F4, same definition everywhere.** Heat-day strengths used by the conception model must be computed from the same baseline ratios as the estrus features.

**Estrus features (20 per row)**

| Feature | Definition |
| --- | --- |
| `<signal>_ratio` (5) | Today's value divided by the cow's baseline |
| `<signal>_z` (5) | (Today's value − baseline) divided by the rolling SD of days t−14 to t−1 |
| `<signal>_z_lag1` (5) | Yesterday's z-score |
| `act_z_roll3max` | Highest activity z-score over today and the two previous days |
| `days_since_alert` | Days since the last earlier day with activity ratio above 1.5; empty if none |
| `dim`, `parity`, `pct_holstein` | Context |

The five signals are activity, rumination, lying time, body temperature and milk yield. The label is `estrus_day`.

**Conception features (8 per insemination).** Heat-day activity strength, heat-day rumination strength, heat duration, parity, semen type, technician, DIM and the silent-heat flag. Duration needs hourly data; if only daily data exists, drop it and retrain. The label is `conceived`.

**Lactation features**

| Item | Definition |
| --- | --- |
| Early window | Milk on days 5 to 60 |
| Parity-mean curve | Mean milk by parity and DIM, computed from training cows only |
| Scale | s = Σ(y × f) / Σ(f²) over days 5 to 60, where y is the cow's milk and f the parity curve |
| Scaled forecast | s × the parity curve for days 61 to 305 |
| Correction inputs | Parity, % Holstein, early mean, early slope (mean of the last 14 early days over the first 14), scale, target DIM, scaled forecast |
| Correction target | Actual milk divided by the scaled forecast |

## 6. Model specifications

Four models share one feature layer and one evaluation protocol; settings below are the values used in the current run.

**6.1 Behaviour classifier**

| Aspect | Specification |
| --- | --- |
| Input | IMU windows of 5 seconds at 10 Hz (50 samples); axes to be confirmed |
| Output | Class: walking, grazing, resting; standing optional |
| Algorithm | Random forest; record its hyperparameters in the config |
| Split | By cow, 5-fold |
| Reference result | Macro F1 0.88 and accuracy 0.93 on unseen cows (3 classes); 0.67 with standing added |
| Acceptance | Macro F1 of 0.85 or higher on unseen cows for 3 classes |
| Status | Measured in an earlier session; the code is not yet in this repository (FR-M1) |

**6.2 Estrus detector**

| Aspect | Specification |
| --- | --- |
| Input | The 20 features in section 5 for one cow-day |
| Output | `p_estrus` in 0 to 1, and an alert when it reaches the threshold |
| Algorithm | Histogram gradient boosting classifier: depth 5, learning rate 0.06, 300 iterations, L2 1.0, early stopping on 15%, seed 7 |
| Threshold | The value maximising F1 on validation cows (0.71 as run) |
| Baselines | Balanced logistic regression; rule that alerts when the activity ratio exceeds a tuned value (1.33 as run) |
| Metrics | PR-AUC, precision, recall, F1, false alerts per 100 cow-days |
| Acceptance | F1 of 0.85 or higher, PR-AUC of 0.90 or higher, and F1 above the activity rule on test cows |

**6.3 Conception model**

| Aspect | Specification |
| --- | --- |
| Input | The 8 features in section 5 for one insemination |
| Output | `p_conception` in 0 to 1 |
| Algorithms | Standardised logistic regression; gradient boosting with depth 3, learning rate 0.05, 150 iterations, L2 2.0 |
| Metrics | Brier score, reliability curve, AUC |
| Acceptance | Brier score below the base-rate Brier (0.226 at a 34.6% conception rate) in 5-fold group cross-validation |
| Current status | Not met: logistic 0.249, boosting 0.243; AUC 0.555 and 0.486 against 0.595 for the true probability |
| Required fix (FR-M2) | Remove class weighting, which distorts probabilities; add cross-validated calibration (isotonic or Platt); use shallower trees with a minimum leaf size; publish the reliability curve |

**6.4 Lactation model**

| Aspect | Specification |
| --- | --- |
| Input | Milk on days 5 to 60, parity and % Holstein |
| Output | Daily milk for days 61 to 305 and the remaining-lactation total |
| Algorithm | Scaled parity-mean curve with a histogram gradient boosting correction: depth 4, learning rate 0.05, 200 iterations, L2 2.0, seed 7 |
| Baselines | Parity-mean curve (MAE 3.79 kg a day); free Wood fit on days 5 to 60 with a in 5 to 60, b in 0.02 to 0.6, c in 0 to 0.01 (MAE 3.83) |
| Metrics | MAE in kg a day, MAPE, remaining-yield error |
| Acceptance | Remaining-yield error of 5% or lower and MAE at least 30% below the parity-mean curve |
| Current status | Met: MAE 1.45, MAPE 4.3%, remaining-yield error 2.4% |

## 7. Evaluation protocol and acceptance criteria

Every reported score must come from cows the model never saw, and every model is reported beside its baseline.

**Protocol**

- **FR-E1.** Split by cow only: 60% train, 20% validation and 20% test cows (shuffle seed 7), plus 5-fold group cross-validation by cow for stability.
- **FR-E2.** Tune thresholds, including the estrus alert and the rule baseline, on validation cows and never on test cows.
- **FR-E3.** Final numbers come from test cows; once real data exists, from real cows only.
- **FR-E4.** Report mean and standard deviation across folds, and add cow-level bootstrap confidence intervals.
- **FR-E5.** Report recall for silent heats separately from normal heats.
- **FR-E6.** Automated leakage tests: no cow appears in two splits, baselines use only earlier days, and the parity-mean curve is built from training cows only.

**Acceptance status on synthetic data**

| Criterion | Target | Current | Status |
| --- | --- | --- | --- |
| Estrus F1 | 0.85 or higher | 0.91 | Met |
| Estrus PR-AUC | 0.90 or higher | 0.961 | Met |
| Estrus F1 against the activity rule | Higher | 0.91 against 0.88 | Met |
| Conception Brier score | Below 0.226 | 0.249 and 0.243 | Not met |
| Lactation remaining-yield error | 5% or lower | 2.4% | Met |
| Lactation MAE against the parity curve | At least 30% lower | 62% lower (1.45 against 3.79 kg) | Met |
| Behaviour macro F1 (3 classes) | 0.85 or higher | 0.88 | Met, not reproduced in this repository |
| Leakage tests | All pass | Not written | To do |

These targets apply to the synthetic data only. Real-data targets are set after the first real run, and real-data results always supersede synthetic ones.

## 8. Pipeline, repository layout and interfaces

One repository and one config file cover four steps: simulate, validate, train and predict.

**FR-P1, repository layout.** Today the code is two flat scripts, `generate_data.py` and `train.py`. Refactor to this layout:

```text
cattle-estrus/
  config.yaml          # simulator, feature, model and split constants
  data/                # cows.csv, daily.csv, inseminations.csv
  src/
    simulate.py        # generator logic (FR-S1 to FR-S7)
    validate.py        # schema and range checks (FR-D1)
    features.py        # causal baselines and all feature tables
    models/
      behaviour.py
      estrus.py
      conception.py
      lactation.py
    evaluate.py        # splits, metrics, leakage tests
    predict.py         # batch scoring
  models/              # versioned model files
  reports/             # metrics.json, plots, model cards
  tests/
```

**Commands**

| Command | Output |
| --- | --- |
| `python -m src.simulate --n-cows 300 --seed 42` | `data/*.csv` |
| `python -m src.validate data/` | Pass or fail report |
| `python -m src.train --config config.yaml` | `models/` and `reports/metrics.json` |
| `python -m src.predict --daily data/daily.csv --out out/` | One predictions file per model |

A single `make all` runs the four steps in order.

**FR-P2, configuration.** All constants live in `config.yaml`: data seed 42, split seed 7, split fractions, window sizes, thresholds and model settings. A model file stores the config and threshold it was trained with, and its version string follows `<task>-<algorithm>-<semver>`.

**FR-P3, prediction outputs**

| Model | Columns |
| --- | --- |
| Estrus | `cow_id`, `dim`, `p_estrus`, `alert`, `threshold`, `model_version` |
| Conception | `cow_id`, `dim`, `semen_type`, `p_conception`, `model_version` |
| Lactation | `cow_id`, `dim`, `milk_kg_pred` for days 61 to 305, plus `total_305_pred` per cow |

**FR-P4, optional service (proposed).** A thin REST wrapper with `POST /v1/estrus`, `/v1/conception` and `/v1/lactation` taking the same columns as the input tables and returning the rows above, plus `GET /healthz`. Build it only after the batch path passes acceptance.

## 9. Non-functional requirements and testing

The pipeline must be reproducible, fast enough to re-run on every change and traceable back to its data.

| ID | Requirement | Target |
| --- | --- | --- |
| NFR-1 | Reproducibility | Same config and seeds give the same data and metrics to 4 decimals; pin Python dependencies (numpy, pandas, scipy, scikit-learn 1.9.1 in the current run) |
| NFR-2 | Runtime | Full train and evaluate on 300 cows in under 60 seconds on 2 CPU cores; the current run takes about 17 seconds |
| NFR-3 | Memory | Under 2 GB for the 300-cow daily table |
| NFR-4 | Traceability | Each model file records the data hash, config, code commit and metrics |
| NFR-5 | Data safety | No farm identifiers beyond `cow_id`; the synthetic flag is mandatory; real data stays out of the repository |
| NFR-6 | Failure behaviour | A validation failure stops the run with a message naming the column and rule |
| NFR-7 | Logging | Each stage logs row counts, split sizes and thresholds |
| NFR-8 | Model cards | Each report states the data source (synthetic or real), number of cows, metrics and known limits |

**Test plan**

| Level | Tests |
| --- | --- |
| Unit | Simulator gives identical files for the same seed; validation rejects each broken rule; Wood function and scale formula match a hand-worked toy curve |
| Leakage | Changing any value after day t leaves day t's features unchanged; a cow never appears in two splits; the baseline excludes today |
| Integration | `make all` on 30 cows finishes and writes `metrics.json` with every expected key |
| Regression | On 300 cows at seed 42, estrus F1, conception Brier and lactation error stay within a stated tolerance of the section 7 table |

## 10. Milestones, risks and open questions

The baseline pipeline is done; the next gate is fixing conception calibration, then hardening the repository before any real data arrives.

**Milestones, in order**

1. **M0, baseline pipeline (done).** Generator, features and three models scored on 300 synthetic cows, with `metrics.json`.
2. **M1, conception calibration.** Apply FR-M2. Exit: Brier score below 0.226 in 5-fold group cross-validation.
3. **M2, repository hardening.** Layout, config, validation and leakage tests (FR-P1, FR-P2, FR-D1, FR-E6). Exit: `make all` and the full test suite pass.
4. **M3, behaviour classifier in the repository.** FR-M1. Exit: macro F1 of 0.85 or higher on unseen cows for 3 classes.
5. **M4, hourly and IMU synthetic extensions.** FR-S7, reconciled with the earlier assets. Exit: validation targets pass and heat duration is supported.
6. **M5, real-data ingestion.** Exit: real tables pass validation, real-data targets are set and a real-only report exists.
7. **M6, optional service.** FR-P4. Exit: endpoints return the section 8 schemas.

**Risks**

| Risk | Impact | Mitigation |
| --- | --- | --- |
| Real estrus signal is weaker than simulated | Scores drop sharply; published AUCs for simple models on real sensors are roughly 0.68 to 0.75 | Report real-only results, retune the threshold, keep the activity-rule baseline |
| Estrus labels come from the same sensor being predicted | Inflated scores | Require observed heats: visual checks, tail paint or progesterone |
| Conception is barely predictable at this effect size | The model adds no value over the base rate | Judge by calibration and fall back to the base rate by parity and semen type |
| Too few real cows | High variance | Cow-level bootstrap intervals; collect 100 or more cows over a full lactation |
| No daily milk records | Lactation model cannot be trained | Ship the estrus model first |
| Synthetic assumptions mislead | False confidence | Config-driven simulator, documented limits, ablation by `is_synthetic` |

**Open questions**

1. Which axes and file format do the collar clips use, and which random-forest settings produced the 0.88 result?
2. What do the earlier synthetic assets (hourly 10 cows, daily 10 cows, IMU clips 441, `gen_synthetic.py`) contain, and should they replace or extend section 4?
3. Will real data include hourly sensor values, which heat duration needs, or daily values only?
4. How will real estrus events be recorded?
5. Who signs off the real-data targets?
6. Is the optional service needed?
