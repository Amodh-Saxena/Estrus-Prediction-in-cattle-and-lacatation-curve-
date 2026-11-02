"""Summarize supplied real collar, halter, jerk-stream, and camera files without copying source media."""
from __future__ import annotations

import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

DATA2_TEST_ANIMALS = {4, 10, 11}
DATA2_TRIAL_MAP = {
    1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1,
    7: 2, 8: 2, 9: 2, 10: 2,
    11: 3, 12: 3, 13: 3, 14: 3, 15: 3, 16: 3, 17: 3, 18: 3,
}


def build_inventory() -> dict:
    clips = []
    collar_root = ROOT / "db-cow-walking-main"
    if collar_root.exists():
        for folder in sorted(p for p in collar_root.iterdir() if p.is_dir()):
            files = list(folder.glob("*.csv"))
            if files:
                clips.append({"folder": folder.name, "csv_files": len(files)})

    camera = []
    camera_root = ROOT / "Wagyu&Angus"
    if camera_root.exists():
        for folder in sorted(p for p in camera_root.iterdir() if p.is_dir()):
            files = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
            camera.append({"folder": folder.name, "image_files": len(files)})

    data2_summary = {"available": False}
    data2_root = ROOT / "data2"
    if data2_root.exists():
        accel_files = sorted(data2_root.glob("accel-*.csv"))
        halter_files = sorted(p for p in data2_root.glob("halter-*.csv") if "(" not in p.name)
        animals = []
        total_bytes = 0
        for acc_p in accel_files:
            try:
                aid = int(acc_p.stem.split("-")[1])
            except (IndexError, ValueError):
                continue
            hal_p = data2_root / f"halter-{aid:02d}.csv"
            acc_sz = acc_p.stat().st_size
            hal_sz = hal_p.stat().st_size if hal_p.exists() else 0
            total_bytes += acc_sz + hal_sz
            animals.append({
                "animal_id": aid,
                "farm_trial_id": DATA2_TRIAL_MAP.get(aid, 3),
                "is_official_test": aid in DATA2_TEST_ANIMALS,
                "accel_file": acc_p.name,
                "accel_mb": round(acc_sz / (1024 * 1024), 2),
                "halter_file": hal_p.name if hal_p.exists() else None,
                "halter_mb": round(hal_sz / (1024 * 1024), 2) if hal_p.exists() else 0.0,
                "has_paired_halter": hal_p.exists(),
            })
        data2_summary = {
            "available": True,
            "dataset_title": "Precision Beef - Animal Behaviour Classification (Zenodo 4064802)",
            "location": "Easter Howgate Farm, Edinburgh, UK",
            "sampling_rate_hz": 10,
            "sensors": ["Afimilk Silent Herdsman 3-axis Collar Accelerometer (x,y,z in mg)", "RumiWatch Pressure Halter (0=Other, 1=Ruminating, 2=Eating)"],
            "n_animals_total": len(accel_files),
            "n_animals_paired_halter": len(halter_files),
            "official_test_animals": sorted(DATA2_TEST_ANIMALS),
            "unlabeled_inference_animals": [a["animal_id"] for a in animals if not a["has_paired_halter"]],
            "total_size_gb": round(total_bytes / (1024 ** 3), 2),
            "animals": animals,
        }

    data3_summary = {"available": False}
    data3_csv = ROOT / "data3" / "Preprocessed_Final.csv"
    if data3_csv.exists():
        df3 = pd.read_csv(data3_csv)
        data3_summary = {
            "available": True,
            "file": "data3/Preprocessed_Final.csv",
            "size_kb": round(data3_csv.stat().st_size / 1024.0, 2),
            "rows": int(len(df3)),
            "columns": list(df3.columns),
            "start_time": str(df3["time"].iloc[0]),
            "end_time": str(df3["time"].iloc[-1]),
            "non_zero_activity_fraction": round(float((df3["sum_diff"] > 0).mean()), 4),
            "mean_sum_diff_g": round(float(df3["sum_diff"].mean()), 6),
            "max_sum_diff_g": round(float(df3["sum_diff"].max()), 6),
        }

    data4_summary = {"available": False}
    data4_root = ROOT / "data4"
    if data4_root.exists():
        subfolders = []
        for sf in sorted(p for p in data4_root.iterdir() if p.is_dir()):
            f_list = [
                {"name": f.name, "size_kb": round(f.stat().st_size / 1024.0, 2)}
                for f in sorted(sf.iterdir())
                if f.is_file()
            ]
            subfolders.append({"folder": sf.name, "files": f_list})
        data4_summary = {
            "available": True,
            "subfolders": subfolders,
            "datasets": [
                "Smid et al. (2021/2024) Dairy Cow Estrus Behaviour (102 estrus events, 94 SAS-validated across 45 cows, Agassiz BC)",
                "Electronic Ear-Tag Temperature & Step Estrus Records (108.9 MB XLSX + 4-page confirmed estrus PDF)",
                "Expression of Estrus in Dairy Cows Supplementary Figures 1-3 PDF (Conception & Heritability Priors)",
            ],
        }

    return {
        "real_collar_sources": clips,
        "data2_precision_beef": data2_summary,
        "data3_preprocessed_jerk": data3_summary,
        "data4_estrus_and_eartag_pack": data4_summary,
        "camera_sources": camera,
        "policy": "Original real data remains in its supplied folders (is_synthetic=0) and is never mixed with synthetic training rows without explicit provenance flags.",
    }


def main():
    out = ROOT / "reports" / "data_inventory.json"
    out.parent.mkdir(exist_ok=True)
    inventory = build_inventory()
    out.write_text(json.dumps(inventory, indent=2), encoding="utf-8")
    print(json.dumps(inventory, indent=2))


if __name__ == "__main__":
    main()
