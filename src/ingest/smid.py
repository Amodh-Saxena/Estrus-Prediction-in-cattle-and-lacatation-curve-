"""Ingest Smid et al. (2021/2024) Dairy Cow Estrus Behaviour dataset with SAS validation."""
from __future__ import annotations

import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import pandas as pd

from src.common import ROOT, require

RAW_SMID_XLSX = (
    ROOT
    / "data4"
    / "Smid et al., 2024 estrus behaviour outdoor pack"
    / "Smid et al., 2021 indoor versus outdoor cows_26Feb2024.xlsx"
)
OUT_CSV = ROOT / "data" / "real_data4_smid_estrus.csv"

# SAS exclusion rules from Smid et al,. 2021 estrus behavior_26Feb2024.sas (lines 22-29: AM observed or cystic)
SMID_SAS_EXCLUSIONS = {
    (5113, 1),
    (5115, 1),
    (5090, 6),
    (5085, 5),
    (4118, 3),
    (1064, 3),
    (2127, 3),
    (4092, 1),
}


def _col_to_idx(col_ref: str) -> int:
    col = "".join(ch for ch in col_ref if ch.isalpha())
    idx = 0
    for ch in col:
        idx = idx * 26 + (ord(ch.upper()) - ord("A") + 1)
    return idx - 1


def read_xlsx_sheet_stdlib(xlsx_path: Path, sheet_name: str = "xl/worksheets/sheet1.xml") -> pd.DataFrame:
    """Dependency-free XLSX reader using Python standard library."""
    ns = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
    with zipfile.ZipFile(xlsx_path) as z:
        strings: list[str] = []
        if "xl/sharedStrings.xml" in z.namelist():
            for _, elem in ET.iterparse(z.open("xl/sharedStrings.xml"), events=("end",)):
                if elem.tag == ns + "si":
                    strings.append("".join(t.text or "" for t in elem.iter(ns + "t")))
                    elem.clear()
        rows: list[dict[int, str | None]] = []
        max_col = 0
        for _, elem in ET.iterparse(z.open(sheet_name), events=("end",)):
            if elem.tag == ns + "row":
                r_dict: dict[int, str | None] = {}
                for c in elem.findall(ns + "c"):
                    ref = c.attrib.get("r", "A1")
                    c_idx = _col_to_idx(ref)
                    max_col = max(max_col, c_idx + 1)
                    t = c.attrib.get("t")
                    v = c.find(ns + "v")
                    val = v.text if v is not None else None
                    if t == "s" and val is not None:
                        val = strings[int(val)]
                    r_dict[c_idx] = val
                rows.append(r_dict)
                elem.clear()
    if not rows:
        return pd.DataFrame()
    hdr = [rows[0].get(i, f"col_{i}") for i in range(max_col)]
    data = [[r.get(i, None) for i in range(max_col)] for r in rows[1:]]
    return pd.DataFrame(data, columns=hdr)


def load_smid_events() -> pd.DataFrame:
    """Load and clean Smid et al. dataset per SAS specification."""
    require("smid_estrus", "findings")
    if not RAW_SMID_XLSX.exists():
        if OUT_CSV.exists():
            return pd.read_csv(OUT_CSV)
        return pd.DataFrame()

    raw = read_xlsx_sheet_stdlib(RAW_SMID_XLSX)
    num_cols = [
        "cow", "estrus_no", "Hours_in", "Hours_out",
        "s", "m", "x", "w", "y", "z",
        "estrus_hour_start", "BCS", "GS", "ME",
        "DIM", "LACT", "in_cows_heat", "out_cows_heat", "cows_heat",
        "Duration", "day", "month", "year", "THI",
    ]
    for c in num_cols:
        if c in raw.columns:
            raw[c] = pd.to_numeric(raw[c], errors="coerce")

    raw["sas_valid"] = [
        0 if (int(r.cow), int(r.estrus_no)) in SMID_SAS_EXCLUSIONS else 1
        for _, r in raw.iterrows()
    ]
    raw["total_time"] = raw["Hours_in"] + raw["Hours_out"]
    raw["all_mounts"] = raw["s"] + raw["m"] + raw["x"] + raw["w"] + raw["y"] + raw["z"]
    raw["all_intensity"] = raw["all_mounts"] / raw["total_time"].replace(0, np.nan)
    raw["is_outdoor"] = (raw["PackTreat"].astype(str).str.lower() == "outdoor").astype(int)
    raw["is_sound_gait"] = (raw["GS"] <= 2).astype(int)
    raw["gait_group"] = np.where(raw["GS"] <= 2, "Sound (GS 1-2)", "Lame (GS 3-5)")
    raw["parity_group"] = np.where(raw["LACT"] == 1, "Primiparous", "Multiparous")
    raw["is_night_onset"] = ((raw["estrus_hour_start"] < 7) | (raw["estrus_hour_start"] >= 19)).astype(int)
    raw["timeblock"] = np.where(raw["is_night_onset"] == 1, "Night (19:00-07:00)", "Day (07:00-19:00)")
    raw["is_synthetic"] = 0
    raw["source"] = "smid_estrus"

    valid = raw[raw["sas_valid"] == 1].copy().reset_index(drop=True)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    valid.to_csv(OUT_CSV, index=False)
    return valid


if __name__ == "__main__":
    df = load_smid_events()
    print(f"Loaded Smid et al. events: {len(df)} SAS-valid events across {df['cow'].nunique()} Holstein cows.")

