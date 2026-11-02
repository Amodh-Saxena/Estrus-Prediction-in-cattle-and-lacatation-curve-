# Synthetic cattle database

`synthetic_cattle.db` is a SQLite database containing the synthetic 10-cow dataset for DIM 11–110. Build or refresh it from the CSV source files with:

```powershell
python build_database.py
```

The database contains:

- `cows`: 10 synthetic Holstein cows and their parity.
- `hourly_observations`: 24,000 synthetic cow-hour rows with movement, rumination, heat, and `days_to_next_heat` fields.
- `daily_observations`: 1,000 synthetic cow-day rows with per-cow rolling 7-day z-scores and heat timing.
- `imu_clip_summaries`: 441 synthetic clip summaries at 10 Hz, matching the source label counts (105 Grazing, 217 Walking, 10 Resting, and 109 across nine rare behaviours). These are feature summaries, not raw IMU waveforms.
- `dataset_metadata`: dataset scope, provenance, and limitations.

Every data table has an `is_synthetic` field constrained to `1`. Real collar measurements and real camera images are not copied into this database. The simulator uses the real collar dataset's class counts as a reference. No synthetic camera images or camera-derived features are included yet.

The schema is created by `build_database.py`. CSV files in the project root remain the reproducible source inputs; rerunning the command replaces the database contents with those files.
