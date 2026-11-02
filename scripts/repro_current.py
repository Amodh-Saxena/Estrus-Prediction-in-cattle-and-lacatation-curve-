"""Reproduce and audit reported baseline metrics against reports/metrics.json."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
METRICS_FILE = ROOT / "reports" / "metrics.json"

CLAIMED = {
    "eartag_roc_auc": 0.9544,
    "eartag_pr_auc": 0.8605,
    "data2_cv_macro_f1": 0.8793,
    "data2_test_macro_f1": 0.9786,
    "smid_r2": 0.8391,
}


def main() -> None:
    if not METRICS_FILE.exists():
        print(f"Error: {METRICS_FILE} not found. Run training first.")
        return

    m = json.loads(METRICS_FILE.read_text(encoding="utf-8"))
    models = m.get("models", {})
    estrus_m = models.get("estrus", {})
    beh_m = models.get("behaviour", {})

    et = estrus_m.get("data4_eartag_temperature_step_monitoring", {}).get("eartag_estrus_classifier_5fold_cow_cv", {})
    d2 = beh_m.get("data2_precision_beef", {})
    smid = estrus_m.get("data4_smid_estrus_behaviour", {}).get("estrus_duration_regressor_5fold_cv", {})

    actual = {
        "eartag_roc_auc": et.get("roc_auc"),
        "eartag_pr_auc": et.get("pr_auc"),
        "data2_cv_macro_f1": d2.get("group_cv_5fold", {}).get("hgb_macro_f1"),
        "data2_test_macro_f1": d2.get("official_test_split_cows_04_10_11", {}).get("hgb_macro_f1"),
        "smid_r2": smid.get("r2_score"),
    }

    print("\n================= CattleSense Metric Audit =================")
    all_ok = True
    for key, expected in CLAIMED.items():
        got = actual.get(key)
        if got is None:
            print(f"  {key:<22s}: CLAIMED {expected:.4f} | GOT None -> MISSING")
            all_ok = False
        else:
            diff = abs(got - expected)
            status = "OK" if diff <= 0.005 else f"DIFF ({diff:+.4f})"
            if status != "OK":
                all_ok = False
            print(f"  {key:<22s}: CLAIMED {expected:.4f} | GOT {got:.4f} -> {status}")
    print("============================================================\n")


if __name__ == "__main__":
    main()

