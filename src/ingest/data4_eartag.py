"""Ingest 3-sheet electronic ear-tag monitoring workbook and build tidy animal-day records."""
from __future__ import annotations

import datetime as dt_mod
from pathlib import Path
import re
import xml.etree.ElementTree as ET
import zipfile
import numpy as np
import pandas as pd

from src.common import ROOT, require
from src.ingest.data4_labels import load_pdf_labels_df

INTERIM_CSV = ROOT / "data" / "interim" / "eartag_animal_day.csv"
CACHED_DAILY = ROOT / "data" / "real_data4_eartag_daily.csv"
RAW_XLSX = (
    ROOT
    / "data4"
    / "Electronic Ear-Tag Temperature and Estrus Event Re"
    / "sow_ear_tag_monitoring_records.xlsx"
)


def _stream_all_3_sheets_to_daily(xlsx_path: Path) -> tuple[pd.DataFrame, dict[str, int]]:
    """Stream all 2.66M rows across sheets 1, 2, and 3 in 16MB chunks using fast regex scanning."""
    pat = re.compile(
        rb'<c r="A\d+"([^>]*)><v>([^<]+)</v></c>'
        rb'<c r="B\d+"([^>]*)><v>([^<]+)</v></c>.*?'
        rb'<c r="F\d+"[^>]*><v>([^<]+)</v></c>'
        rb'<c r="G\d+"[^>]*><v>([^<]+)</v></c>'
        rb'<c r="H\d+"[^>]*><v>([^<]+)</v></c>'
    )
    buckets: dict[tuple[str, int], list] = {}
    last_step_by_animal: dict[str, float] = {}
    sheet_row_counts: dict[str, int] = {}

    with zipfile.ZipFile(xlsx_path) as z:
        strings: list[str] = []
        if "xl/sharedStrings.xml" in z.namelist():
            for _, elem in ET.iterparse(z.open("xl/sharedStrings.xml"), events=("end",)):
                if elem.tag.endswith("si"):
                    strings.append("".join(t.text or "" for t in elem.iter() if t.tag.endswith("t")).strip())
                    elem.clear()

        for sheet_idx, sname in enumerate(
            ["xl/worksheets/sheet1.xml", "xl/worksheets/sheet2.xml", "xl/worksheets/sheet3.xml"], 1
        ):
            if sname not in z.namelist():
                continue
            s_rows = 0
            with z.open(sname) as f:
                buf = b""
                while True:
                    chunk = f.read(16 * 1024 * 1024)
                    if not chunk:
                        break
                    data = buf + chunk
                    last_row = data.rfind(b"</row>")
                    if last_row == -1:
                        buf = data
                        continue
                    proc, buf = data[: last_row + 6], data[last_row + 6 :]
                    for m in pat.finditer(proc):
                        a_attr, a_val, b_attr, b_val, f_val, g_val, h_val = m.groups()
                        if b_val == b"1" and b't="s"' in b_attr and len(strings) > 1 and strings[1] == "sSowsNo":
                            continue
                        try:
                            ts_f = float(f_val)
                            temp_c = float(g_val)
                            step_v = float(h_val)
                        except ValueError:
                            continue
                        s_rows += 1
                        aid = strings[int(b_val)] if b't="s"' in b_attr else b_val.decode("ascii", errors="ignore")
                        tag = strings[int(a_val)] if b't="s"' in a_attr else a_val.decode("ascii", errors="ignore")
                        day_int = int(ts_f)
                        frac_day = ts_f - day_int
                        is_night = frac_day < 0.2917 or frac_day >= 0.7917

                        prev_step = last_step_by_animal.get(aid)
                        if prev_step is not None and step_v >= prev_step and (step_v - prev_step) <= 5000.0:
                            d_step = step_v - prev_step
                        else:
                            d_step = 0.0
                        last_step_by_animal[aid] = step_v

                        key = (aid, day_int)
                        b = buckets.get(key)
                        if b is None:
                            buckets[key] = [
                                1, temp_c, temp_c * temp_c, temp_c,
                                1 if temp_c >= 34.0 else 0,
                                d_step, d_step, 1 if d_step >= 20.0 else 0,
                                d_step if is_night else 0.0,
                                tag, sheet_idx,
                            ]
                        else:
                            b[0] += 1
                            b[1] += temp_c
                            b[2] += temp_c * temp_c
                            if temp_c > b[3]:
                                b[3] = temp_c
                            if temp_c >= 34.0:
                                b[4] += 1
                            b[5] += d_step
                            if d_step > b[6]:
                                b[6] = d_step
                            if d_step >= 20.0:
                                b[7] += 1
                            if is_night:
                                b[8] += d_step
            sheet_row_counts[f"sheet{sheet_idx}"] = s_rows

    origin = dt_mod.date(1899, 12, 30)
    rows = []
    for (aid, day_int), b in buckets.items():
        n_up = b[0]
        t_mean = b[1] / n_up
        t_var = max(0.0, (b[2] / n_up) - t_mean * t_mean)
        t_std = float(np.sqrt(t_var))
        t_max = b[3]
        t_q90 = min(t_max, t_mean + 1.2816 * t_std)
        d_steps = b[5]
        rows.append({
            "animal_id": str(aid).strip(),
            "ear_tag_code": b[9],
            "sheet_id": int(b[10]),
            "date": (origin + dt_mod.timedelta(days=int(day_int))).isoformat(),
            "n_uploads": int(n_up),
            "temp_mean": round(t_mean, 3),
            "temp_std": round(t_std, 3),
            "temp_q90": round(t_q90, 3),
            "temp_max": round(t_max, 3),
            "warm_34c_frac": round(b[4] / n_up, 4),
            "steps_sum": round(d_steps, 1),
            "max_interval_steps": round(b[6], 1),
            "active_interval_frac": round(b[7] / n_up, 4),
            "night_steps_frac": round(b[8] / max(1.0, d_steps), 4),
        })

    df = pd.DataFrame(rows).sort_values(["animal_id", "date"]).reset_index(drop=True)
    return df, sheet_row_counts


def build_animal_day(force_rescan: bool = False) -> pd.DataFrame:
    """Build or load tidy interim table `eartag_animal_day.csv` with strict animal filtering and zero duplicates."""
    require("data4_eartag", "estrus_train")

    if INTERIM_CSV.exists() and not force_rescan:
        df = pd.read_csv(INTERIM_CSV, parse_dates=["date"])
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()
        df["animal_id"] = df["animal_id"].astype(str).str.strip()
        return df

    if CACHED_DAILY.exists() and not force_rescan:
        cached = pd.read_csv(CACHED_DAILY)
        if len(cached) >= 7000 and "sheet_id" in cached.columns:
            daily = cached.rename(columns={
                "sSowsNo": "animal_id",
                "daily_steps": "steps_sum",
                "temp_mean_c": "temp_mean",
                "temp_std_c": "temp_std",
                "temp_q90_c": "temp_q90",
                "temp_max_c": "temp_max",
            })
            daily["animal_id"] = daily["animal_id"].astype(str).str.strip()
        else:
            daily, _ = _stream_all_3_sheets_to_daily(RAW_XLSX)
    else:
        daily, _ = _stream_all_3_sheets_to_daily(RAW_XLSX)

    daily["date"] = pd.to_datetime(daily["date"]).dt.normalize()
    labels_df = load_pdf_labels_df()

    # Join PDF labels
    merged = daily.merge(
        labels_df[["animal_id", "date", "estrus", "session"]],
        on=["animal_id", "date"],
        how="left",
    )
    merged["estrus"] = merged["estrus"].fillna(0).astype(int)
    merged["session"] = merged["session"].fillna("")

    # Filter to animals present in the PDF schedule (known ground truth)
    labeled_animals = set(labels_df["animal_id"].unique())
    df = merged[merged["animal_id"].isin(labeled_animals)].copy().reset_index(drop=True)

    # Sort and verify uniqueness
    df = df.sort_values(["animal_id", "date"]).reset_index(drop=True)
    assert not df.duplicated(["animal_id", "date"]).any(), "Duplicate animal-days found!"

    df["source"] = "data4_eartag"
    df["is_synthetic"] = 0

    INTERIM_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(INTERIM_CSV, index=False)
    return df


if __name__ == "__main__":
    df = build_animal_day(force_rescan=False)
    print(f"Built eartag_animal_day: {len(df)} rows, {df['animal_id'].nunique()} animals, {df['estrus'].sum()} estrus events.")

