"""Build the documented synthetic 10-cow SQLite database from the CSV assets."""

from __future__ import annotations

import argparse
import csv
import sqlite3
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_DB = ROOT / "data" / "synthetic_cattle.db"
SOURCES = {
    "hourly_observations": ROOT / "synthetic_hourly_10cows.csv",
    "daily_observations": ROOT / "synthetic_daily_10cows.csv",
    "imu_clip_summaries": ROOT / "synthetic_imu_clips_441.csv",
}

SCHEMA = """
PRAGMA foreign_keys = ON;
CREATE TABLE IF NOT EXISTS cows (
    cow_id TEXT PRIMARY KEY,
    breed TEXT NOT NULL,
    parity INTEGER NOT NULL CHECK (parity BETWEEN 1 AND 4),
    dim_start INTEGER NOT NULL DEFAULT 11,
    dim_end INTEGER NOT NULL DEFAULT 110,
    is_synthetic INTEGER NOT NULL DEFAULT 1 CHECK (is_synthetic = 1)
);
CREATE TABLE IF NOT EXISTS hourly_observations (
    cow_id TEXT NOT NULL REFERENCES cows(cow_id),
    dim INTEGER NOT NULL CHECK (dim BETWEEN 11 AND 110),
    hour INTEGER NOT NULL CHECK (hour BETWEEN 0 AND 23),
    t_index INTEGER NOT NULL,
    activity_index REAL NOT NULL,
    rumination_min REAL NOT NULL,
    walking_frac REAL NOT NULL,
    grazing_frac REAL NOT NULL,
    lying_frac REAL NOT NULL,
    standing_frac REAL NOT NULL,
    distance_m REAL NOT NULL,
    mounting_events INTEGER NOT NULL,
    neighbour_proximity REAL NOT NULL,
    head_temp_c REAL NOT NULL,
    thi REAL NOT NULL,
    heat INTEGER NOT NULL CHECK (heat IN (0, 1)),
    days_to_next_heat REAL,
    is_synthetic INTEGER NOT NULL DEFAULT 1 CHECK (is_synthetic = 1),
    PRIMARY KEY (cow_id, dim, hour)
);
CREATE TABLE IF NOT EXISTS daily_observations (
    cow_id TEXT NOT NULL REFERENCES cows(cow_id),
    dim INTEGER NOT NULL CHECK (dim BETWEEN 11 AND 110),
    activity_index REAL NOT NULL,
    rumination_min REAL NOT NULL,
    walking_frac REAL NOT NULL,
    lying_frac REAL NOT NULL,
    distance_m REAL NOT NULL,
    mounting_events INTEGER NOT NULL,
    head_temp_c REAL NOT NULL,
    thi REAL NOT NULL,
    heat INTEGER NOT NULL CHECK (heat IN (0, 1)),
    activity_index_z REAL,
    rumination_min_z REAL,
    distance_m_z REAL,
    head_temp_c_z REAL,
    days_to_next_heat REAL,
    is_synthetic INTEGER NOT NULL DEFAULT 1 CHECK (is_synthetic = 1),
    PRIMARY KEY (cow_id, dim)
);
CREATE TABLE IF NOT EXISTS imu_clip_summaries (
    clip_id INTEGER PRIMARY KEY,
    cow_id TEXT NOT NULL REFERENCES cows(cow_id),
    label TEXT NOT NULL,
    duration_s REAL NOT NULL,
    sample_hz INTEGER NOT NULL,
    acc_mag_mean REAL NOT NULL,
    acc_mag_std REAL NOT NULL,
    gyro_std REAL NOT NULL,
    dom_freq_hz REAL NOT NULL,
    is_synthetic INTEGER NOT NULL DEFAULT 1 CHECK (is_synthetic = 1)
);
CREATE TABLE IF NOT EXISTS dataset_metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_hourly_heat ON hourly_observations(heat, cow_id, dim);
CREATE INDEX IF NOT EXISTS idx_daily_heat ON daily_observations(heat, cow_id, dim);
CREATE INDEX IF NOT EXISTS idx_clips_label ON imu_clip_summaries(label);
"""


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Required input CSV not found: {path}")
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def number(value: str | None):
    if value in (None, ""):
        return None
    return float(value)


def build(db_path: Path) -> None:
    hourly = read_csv(SOURCES["hourly_observations"])
    daily = read_csv(SOURCES["daily_observations"])
    clips = read_csv(SOURCES["imu_clip_summaries"])

    cows: dict[str, tuple[str, int]] = {}
    for row in hourly:
        cows[row["cow_id"]] = (row["breed"], int(row["parity"]))

    # Use minimum hourly lead time as the daily value; the daily CSV remains the
    # source for the published rolling 7-day baseline z-scores.
    next_heat: dict[tuple[str, int], float | None] = {}
    for row in hourly:
        key = (row["cow_id"], int(row["dim"]))
        lead = number(row["days_to_next_heat"])
        if lead is not None and (key not in next_heat or lead < next_heat[key]):
            next_heat[key] = lead

    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.executescript(SCHEMA)
        conn.execute("DELETE FROM hourly_observations")
        conn.execute("DELETE FROM daily_observations")
        conn.execute("DELETE FROM imu_clip_summaries")
        conn.execute("DELETE FROM cows")
        conn.execute("DELETE FROM dataset_metadata")
        conn.executemany(
            "INSERT INTO cows(cow_id,breed,parity) VALUES (?,?,?)",
            [(cow_id, breed, parity) for cow_id, (breed, parity) in sorted(cows.items())],
        )

        hourly_fields = [
            "cow_id", "dim", "hour", "t_index", "activity_index", "rumination_min",
            "walking_frac", "grazing_frac", "lying_frac", "standing_frac", "distance_m",
            "mounting_events", "neighbour_proximity", "head_temp_c", "thi", "heat",
            "days_to_next_heat",
        ]
        conn.executemany(
            f"INSERT INTO hourly_observations({','.join(hourly_fields)}) VALUES ({','.join('?' for _ in hourly_fields)})",
            [tuple(number(r[k]) if k != "cow_id" else r[k] for k in hourly_fields) for r in hourly],
        )

        daily_fields = [
            "cow_id", "dim", "activity_index", "rumination_min", "walking_frac", "lying_frac",
            "distance_m", "mounting_events", "head_temp_c", "thi", "heat", "activity_index_z",
            "rumination_min_z", "distance_m_z", "head_temp_c_z",
        ]
        daily_values = []
        for r in daily:
            values = [r["cow_id"] if k == "cow_id" else number(r[k]) for k in daily_fields]
            values.append(next_heat.get((r["cow_id"], int(r["dim"]))))
            daily_values.append(tuple(values))
        columns = daily_fields + ["days_to_next_heat"]
        conn.executemany(
            f"INSERT INTO daily_observations({','.join(columns)}) VALUES ({','.join('?' for _ in columns)})",
            daily_values,
        )

        clip_fields = ["cow_id", "label", "duration_s", "sample_hz", "acc_mag_mean", "acc_mag_std", "gyro_std", "dom_freq_hz"]
        conn.executemany(
            f"INSERT INTO imu_clip_summaries(clip_id,{','.join(clip_fields)}) VALUES ({','.join('?' for _ in range(len(clip_fields)+1))})",
            [(i, *(r[k] if k in ("cow_id", "label") else number(r[k]) for k in clip_fields)) for i, r in enumerate(clips, 1)],
        )

        metadata = {
            "dataset_name": "synthetic_cattle_10cows_dim11_110",
            "dataset_type": "synthetic_only",
            "source_description": "Seed-42 simulator calibrated to real dataset summary counts; no real sensor rows are included.",
            "cow_count": str(len(cows)),
            "dim_range": "11-110",
            "hourly_rows": str(len(hourly)),
            "daily_rows": str(len(daily)),
            "imu_clip_summary_rows": str(len(clips)),
            "imu_sample_rate_hz": "10",
            "imu_rows_are_summaries": "true; raw 10 Hz waveforms are not present",
            "camera_images": "not included; no synthetic images were created",
            "daily_zscore_baseline": "per-cow previous 7 days, rolling median and standard deviation, min_periods=3",
            "all_rows_is_synthetic": "1",
        }
        conn.executemany("INSERT INTO dataset_metadata(key,value) VALUES (?,?)", metadata.items())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_DB, help="Output SQLite database path")
    args = parser.parse_args()
    build(args.out.resolve())
    print(f"Built synthetic database: {args.out.resolve()}")


if __name__ == "__main__":
    main()
