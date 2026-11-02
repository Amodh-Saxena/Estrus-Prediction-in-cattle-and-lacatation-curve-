"""Validate the PRD CSV tables (FR-D1) and the synthetic SQLite database."""
from __future__ import annotations

import argparse
from contextlib import closing
import sqlite3
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

SENSOR_RANGES = {
    "activity": (5.0, 800.0),
    "rumination_min": (30.0, 1000.0),
    "lying_min": (60.0, 1300.0),
    "body_temp_c": (35.0, 43.0),
    "milk_kg": (0.5, 100.0),
}


def validate_prd_tables(
    cows: pd.DataFrame,
    daily: pd.DataFrame,
    inseminations: pd.DataFrame,
) -> None:
    """Enforce every validation rule in PRD Section 3 (FR-D1) and NFR-6."""
    # 1. Check is_synthetic presence and values across all three tables
    for table_name, df in [("cows.csv", cows), ("daily.csv", daily), ("inseminations.csv", inseminations)]:
        if "is_synthetic" not in df.columns:
            raise ValueError(f"{table_name} [is_synthetic]: rule missing_is_synthetic - column is required.")
        if not set(df["is_synthetic"].dropna().unique()).issubset({0, 1}):
            raise ValueError(f"{table_name} [is_synthetic]: rule binary_flag - must contain only 0 or 1.")

    # 2. Check g_* columns empty when is_synthetic == 0 in cows.csv
    real_cows = cows[cows["is_synthetic"] == 0]
    if len(real_cows) > 0:
        for g_col in ["g_activity", "g_rumination", "g_milk"]:
            if g_col in real_cows.columns and real_cows[g_col].notna().any():
                raise ValueError(
                    f"cows.csv [{g_col}]: rule real_data_latent_merit - must be empty when is_synthetic is 0."
                )

    # 3. Check uniqueness of (cow_id, dim), dim in 1..305, parity in 1..3
    if daily.duplicated(subset=["cow_id", "dim"]).any():
        raise ValueError("daily.csv [cow_id, dim]: rule unique_cow_dim - duplicate (cow_id, dim) rows found.")
    if ((daily["dim"] < 1) | (daily["dim"] > 305)).any():
        raise ValueError("daily.csv [dim]: rule dim_range - dim must be between 1 and 305.")
    for table_name, df in [("cows.csv", cows), ("daily.csv", daily), ("inseminations.csv", inseminations)]:
        if len(df) > 0 and not set(df["parity"].unique()).issubset({1, 2, 3}):
            raise ValueError(f"{table_name} [parity]: rule parity_domain - parity must be 1, 2, or 3.")

    # 4. Binary flags and monotonic pregnancy
    for table_name, df, col in [
        ("daily.csv", daily, "estrus_day"),
        ("daily.csv", daily, "pregnant"),
        ("inseminations.csv", inseminations, "conceived"),
        ("inseminations.csv", inseminations, "silent_heat"),
    ]:
        if len(df) > 0 and not set(df[col].dropna().unique()).issubset({0, 1}):
            raise ValueError(f"{table_name} [{col}]: rule binary_domain - must contain only 0 or 1.")

    daily_sorted = daily.sort_values(["cow_id", "dim"])
    for cow_id, grp in daily_sorted.groupby("cow_id", sort=False):
        preg = grp["pregnant"].to_numpy(dtype=int)
        if np.any(np.diff(preg) < 0):
            raise ValueError(
                f"daily.csv [pregnant]: rule monotonic_pregnancy - cow {cow_id} returned from 1 to 0."
            )

    # 5. Sensor ranges and missing rate <= 10%
    for col, (low, high) in SENSOR_RANGES.items():
        if col not in daily.columns:
            raise ValueError(f"daily.csv [{col}]: rule required_sensor - column missing.")
        miss_rate = float(daily[col].isna().mean())
        if miss_rate > 0.10:
            raise ValueError(
                f"daily.csv [{col}]: rule missing_rate_limit - {miss_rate*100:.1f}% missing exceeds 10% limit."
            )
        valid = daily[col].dropna()
        if ((valid < low) | (valid > high)).any():
            raise ValueError(
                f"daily.csv [{col}]: rule plausible_sensor_range - values outside [{low}, {high}]."
            )


def validate_sqlite(db_path: Path) -> None:
    if not db_path.exists():
        raise ValueError(f"Database missing: {db_path}. Run python build_database.py first.")
    with closing(sqlite3.connect(db_path)) as conn:
        required = {
            "cows": 10,
            "hourly_observations": 24000,
            "daily_observations": 1000,
            "imu_clip_summaries": 441,
        }
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        for table, expected in required.items():
            if table not in tables:
                raise ValueError(f"Missing required table: {table}")
            count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            if count != expected:
                raise ValueError(f"{table} row count expected {expected}, got {count}")
            flags = conn.execute(f"SELECT DISTINCT is_synthetic FROM {table}").fetchall()
            if flags != [(1,)]:
                raise ValueError(f"{table}.is_synthetic must be 1 for every row")
        duplicates = conn.execute(
            "SELECT cow_id,dim,hour,COUNT(*) FROM hourly_observations GROUP BY 1,2,3 HAVING COUNT(*)>1"
        ).fetchone()
        if duplicates:
            raise ValueError(f"Duplicate hourly key found: {duplicates[:3]}")
        duplicates = conn.execute(
            "SELECT cow_id,dim,COUNT(*) FROM daily_observations GROUP BY 1,2 HAVING COUNT(*)>1"
        ).fetchone()
        if duplicates:
            raise ValueError(f"Duplicate daily key found: {duplicates[:2]}")
        labels = dict(conn.execute("SELECT label,COUNT(*) FROM imu_clip_summaries GROUP BY label"))
        for label, expected in {"Grazing": 105, "Walking": 217, "Resting": 10}.items():
            if labels.get(label) != expected:
                raise ValueError(f"IMU label {label} expected {expected}, got {labels.get(label)}")
        foreign_key_errors = conn.execute("PRAGMA foreign_key_check").fetchall()
        if foreign_key_errors:
            raise ValueError(f"Foreign key violations: {foreign_key_errors[:3]}")
    print(f"SQLite validation passed: {db_path} (10 cows, 24,000 hourly, 1,000 daily, 441 IMU summaries)")


def validate(target: Path) -> None:
    data_dir = target.parent if target.is_file() else target
    db_path = target if target.suffix.lower() in {".db", ".sqlite", ".sqlite3"} else (data_dir / "synthetic_cattle.db")
    if db_path.exists():
        validate_sqlite(db_path)

    cows_csv = data_dir / "cows.csv"
    daily_csv = data_dir / "daily.csv"
    insem_csv = data_dir / "inseminations.csv"
    if cows_csv.exists() and daily_csv.exists() and insem_csv.exists():
        cows = pd.read_csv(cows_csv)
        daily = pd.read_csv(daily_csv)
        insem = pd.read_csv(insem_csv)
        validate_prd_tables(cows, daily, insem)
        print(
            f"PRD CSV validation passed: {len(cows)} cows, {len(daily)} daily rows, "
            f"{len(insem)} inseminations in {data_dir}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", nargs="?", default=str(ROOT / "data"))
    args = parser.parse_args()
    validate(Path(args.data_dir))


if __name__ == "__main__":
    main()
