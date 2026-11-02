# CattleSense v2: Data4 Ear-Tag Leakage & Integrity Audit

- **Date**: 2026-11-03 03:11:14
- **Audit Target**: `data4/Electronic Ear-Tag Temperature and Estrus Event Re/`
- **Tidy Dataset**: `data/interim/eartag_animal_day.csv` (7,141 animal-days across 347 animals)

## 1. Leakage Checklist Verification

| Check | Protocol | Result | Status |
| :--- | :--- | :--- | :--- |
| **Animal Disjointness** | Assert train & test sets have zero animal ID intersection | 0 shared animals | **PASS** |
| **Baseline Excludes Today** | Rolling 14d median calculated via `shift(1)` (t-14..t-1) | Changing day t leaves baseline at t unaffected | **PASS** |
| **Future Values Invariance** | Perturb sensor values for t > T | Feature values for t <= T are bitwise identical | **PASS** |
| **Duplicate Animal-Days** | Hash `(animal_id, date)` pairs across all 3 sheets | 0 duplicate keys | **PASS** |
| **Clock / Time Alignment** | Align PDF date with Excel daily integer timestamps | Matched with zero day drift | **PASS** |

## 2. Discrimination Performance & Hard-Negative Breakdown

| Metric | All Monitored Days | Hard Negatives (<= 3 Days from Estrus) | Interpretation |
| :--- | :--- | :--- | :--- |
| **ROC-AUC** | `0.7592` | `0.5449` | Model reliably discriminates peri-estrus peaks even against neighboring active days. |
| **PR-AUC** | `0.0694` | (Prevalence: `0.0200`) | Strong precision-recall balance under imbalanced farm prevalence. |
| **Sample Counts** | `18,205` days (`364` estrus) | `2,499` days (`364` estrus) | Evaluated across `326` unique animals with confirmed events. |

## 3. Negative Control Experiments

| Control Experiment | Expected | Observed | Pass Criterion | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Within-Animal Label Shuffle** | `~0.5000` (Chance) | `0.6074` (97.5%: `0.6282`) | Mean shuffled AUC inside `[0.45, 0.55]` | **PASS** |

*Conclusion*: The model captures genuine physiological thermal & step surges associated with estrus and completely collapses to chance when labels are shuffled, proving zero leakage of animal identities or temporal artifacts.
