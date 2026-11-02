"""Ingest and window-feature extractor for `data2/` (Zenodo 4064802 Precision Beef)."""
from __future__ import annotations

import io
from pathlib import Path
import numpy as np
import pandas as pd

from src.common import ROOT, require

DATA2_DIR = ROOT / "data2"
OUT_DAILY_CSV = ROOT / "data" / "real_data2_daily.csv"

DATA2_TEST_ANIMALS = {4, 10, 11}
DATA2_TRIAL_MAP = {
    1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1,
    7: 2, 8: 2, 9: 2, 10: 2, 11: 2, 12: 2,
    13: 3, 14: 3, 15: 3, 16: 3, 17: 3, 18: 3,
}
HALTER_LABEL_MAP = {0: "Other", 1: "Ruminating", 2: "Eating"}

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


def _seek_csv_by_ts(path: Path, target_ts: str, n_lines: int = 3600) -> pd.DataFrame:
    """Binary-search a multi-hundred-MB timestamp CSV in O(log N) seeks."""
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


def read_aligned_chunk(animal_id: int, frac: float, n_lines: int = 3600) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Read `n_lines` at relative offset `frac` from accel-XX and halter-XX."""
    require("data2_collar", "behaviour_train")
    a_path = DATA2_DIR / f"accel-{animal_id:02d}.csv"
    h_path = DATA2_DIR / f"halter-{animal_id:02d}.csv"
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


def compute_window_features(sub_a: pd.DataFrame, cow_x_med: float, cow_y_med: float, cow_z_med: float) -> dict[str, float]:
    """Extract biomechanical and spectral features for a 10 Hz window."""
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
        "x_mean": float(x.mean()), "y_mean": float(yv.mean()), "z_mean": float(z.mean()),
        "x_dev_cow": float(x.mean() - cow_x_med), "y_dev_cow": float(yv.mean() - cow_y_med), "z_dev_cow": float(z.mean() - cow_z_med),
        "x_std": float(x.std()), "y_std": float(yv.std()), "z_std": float(z.std()),
        "x_iqr": float(np.percentile(x, 75) - np.percentile(x, 25)),
        "y_iqr": float(np.percentile(yv, 75) - np.percentile(yv, 25)),
        "z_iqr": float(np.percentile(z, 75) - np.percentile(z, 25)),
        "odba": odba, "vedba": vedba, "mag_std": float(mag.std()),
        "sum_diff_mean": float(np.mean(sum_diff)), "sum_diff_std": float(np.std(sum_diff)),
        "active_jerk_frac": float(np.mean(sum_diff > 0.015)),
        "pitch_deg": float(np.mean(np.degrees(np.arctan2(x, np.sqrt(yv**2 + z**2) + 1e-5)))),
        "pitch_dev_deg": float(np.mean(np.degrees(np.arctan2(x - cow_x_med, 1.0)))),
        "b_low_frac": b_low / tot_pow, "b_eat_frac": b_eat / tot_pow,
        "b_rum_frac": b_rum / tot_pow, "b_walk_frac": b_walk / tot_pow,
        "rum_to_low_ratio": b_rum / (b_low + 1e-6),
        "sub10s_std_min": float(np.min(sub_stds)), "sub10s_std_max": float(np.max(sub_stds)),
        "sub10s_std_cv": float(np.std(sub_stds) / (np.mean(sub_stds) + 1e-6)),
        "zero_cross_x": float(np.mean(np.diff(np.sign(x - x.mean())) != 0)),
        "zero_cross_y": float(np.mean(np.diff(np.sign(yv - yv.mean())) != 0)),
    }


def extract_data2_windows(
    sample_fracs: tuple[float, ...] = (0.08, 0.28, 0.48, 0.68, 0.88),
    chunk_lines: int = 3600,
    window_samples: int = 600,
) -> tuple[pd.DataFrame, pd.Series, np.ndarray]:
    """Extract labelled windows across 17 paired animals in data2."""
    rows, labels, animal_ids = [], [], []
    for aid in range(1, 18):
        if not (DATA2_DIR / f"accel-{aid:02d}.csv").exists() or not (DATA2_DIR / f"halter-{aid:02d}.csv").exists():
            continue
        chunks = [read_aligned_chunk(aid, frac, n_lines=chunk_lines) for frac in sample_fracs]
        all_a = pd.concat([c[0] for c in chunks], ignore_index=True)
        cx = float(pd.to_numeric(all_a["x"], errors="coerce").median()) / 1000.0
        cy = float(pd.to_numeric(all_a["y"], errors="coerce").median()) / 1000.0
        cz = float(pd.to_numeric(all_a["z"], errors="coerce").median()) / 1000.0

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
                rows.append(compute_window_features(sub_a, cx, cy, cz))
                labels.append(HALTER_LABEL_MAP.get(maj, "Other"))
                animal_ids.append(aid)

    X = pd.DataFrame(rows)[DATA2_FEATURE_COLUMNS].fillna(0.0)
    y = pd.Series(labels)
    groups = np.asarray(animal_ids, dtype=int)
    return X, y, groups


if __name__ == "__main__":
    X, y, groups = extract_data2_windows()
    print(f"Extracted {len(X)} windows across {len(np.unique(groups))} cows.")

