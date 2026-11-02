"""Unit, leakage, integration, and regression tests per PRD Sections 7, 8, and 9."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
import joblib
import numpy as np
import pandas as pd

from generate_data import (
    simulate_herd,
    simulate_hourly_from_daily,
    check_validation_targets,
    wood_curve,
)
from src.evaluate import cow_train_validation_test, run_leakage_checks
from src.features import (
    estrus_features_prd,
    compute_parity_curves,
    build_lactation_features,
)
from src.models.conception import fit as fit_conception
from src.predict import run_predictions
from src.service import dispatch_request
from src.train import run_training
from src.validate import validate_prd_tables

ROOT = Path(__file__).resolve().parents[1]


class TestCattlePipeline(unittest.TestCase):
    def test_unit_simulator_determinism_and_targets(self):
        """Simulator produces identical data for the same seed and meets FR-S6 validation targets."""
        cows1, daily1, insem1, stats1 = simulate_herd(n_cows=30, seed=42, out_dir=None)
        cows2, daily2, insem2, _ = simulate_herd(n_cows=30, seed=42, out_dir=None)
        pd.testing.assert_frame_equal(cows1, cows2)
        pd.testing.assert_frame_equal(daily1, daily2)
        pd.testing.assert_frame_equal(insem1, insem2)
        check_validation_targets(stats1, n_cows=30)

        # Verify full 300-cow seed-42 PRD Section 4 targets
        _, _, _, stats300 = simulate_herd(n_cows=300, seed=42, out_dir=None)
        check_validation_targets(stats300, n_cows=300)
        self.assertEqual(stats300["cow_days"], 91500)

    def test_unit_hourly_extension_preserves_daily_totals(self):
        """FR-S7: 24 hourly bins preserve daily totals while concentrating activity in heat window."""
        _, daily, _, _ = simulate_herd(n_cows=2, seed=42, out_dir=None)
        sub_daily = daily[daily["dim"] <= 40].dropna(subset=["activity", "rumination_min", "lying_min"]).copy()
        hourly = simulate_hourly_from_daily(sub_daily, seed=42)
        self.assertEqual(len(hourly), len(sub_daily) * 24)

        agg = hourly.groupby(["cow_id", "dim"], as_index=False).agg(
            act_mean=("activity", "mean"),
            rum_sum=("rumination_min", "sum"),
            lying_sum=("lying_min", "sum"),
            heat_hours=("heat", "sum"),
        )
        merged = sub_daily.merge(agg, on=["cow_id", "dim"])
        np.testing.assert_allclose(merged["activity"], merged["act_mean"], rtol=1e-3, atol=1e-2)
        np.testing.assert_allclose(merged["rumination_min"], merged["rum_sum"], rtol=1e-3, atol=1e-2)
        np.testing.assert_allclose(merged["lying_min"], merged["lying_sum"], rtol=1e-3, atol=1e-2)

        heat_days = hourly[hourly["dim"].isin(sub_daily.loc[sub_daily["estrus_day"] == 1, "dim"])]
        if len(heat_days) > 0 and (heat_days["heat"] == 1).any():
            in_window = heat_days.loc[heat_days["heat"] == 1, "activity"].mean()
            out_window = heat_days.loc[heat_days["heat"] == 0, "activity"].mean()
            self.assertGreater(in_window, out_window)

    def test_unit_validation_rejects_broken_rules(self):
        """Validation rejects each broken rule in FR-D1 with a clear message naming the column and rule."""
        cows, daily, insem, _ = simulate_herd(n_cows=10, seed=42, out_dir=None)
        validate_prd_tables(cows, daily, insem)

        # 1. Duplicate (cow_id, dim)
        bad_daily = pd.concat([daily, daily.iloc[[0]]], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "unique_cow_dim"):
            validate_prd_tables(cows, bad_daily, insem)

        # 2. DIM out of range
        bad_dim = daily.copy()
        bad_dim.loc[0, "dim"] = 306
        with self.assertRaisesRegex(ValueError, "dim_range"):
            validate_prd_tables(cows, bad_dim, insem)

        # 3. Parity out of domain
        bad_parity = cows.copy()
        bad_parity.loc[0, "parity"] = 5
        with self.assertRaisesRegex(ValueError, "parity_domain"):
            validate_prd_tables(bad_parity, daily, insem)

        # 4. Non-monotonic pregnancy (returns from 1 to 0)
        bad_preg = daily.copy()
        bad_preg.loc[bad_preg["cow_id"] == 1, "pregnant"] = [1 if d == 100 else 0 for d in range(1, 306)]
        with self.assertRaisesRegex(ValueError, "monotonic_pregnancy"):
            validate_prd_tables(cows, bad_preg, insem)

        # 5. Missing sensor rate > 10%
        bad_miss = daily.copy()
        bad_miss.loc[: int(0.15 * len(bad_miss)), "activity"] = np.nan
        with self.assertRaisesRegex(ValueError, "missing_rate_limit"):
            validate_prd_tables(cows, bad_miss, insem)

        # 6. Non-empty g_* when is_synthetic == 0
        bad_real = cows.copy()
        bad_real.loc[0, "is_synthetic"] = 0
        with self.assertRaisesRegex(ValueError, "real_data_latent_merit"):
            validate_prd_tables(bad_real, daily, insem)

    def test_unit_wood_and_scale_formula(self):
        """Wood function and scale formula s = sum(y*f)/sum(f^2) match a hand-worked toy curve."""
        dims = np.arange(1, 306, dtype=float)
        base_curve = wood_curve(dims, a=20.0, b=0.20, c=0.003)
        true_scale = 1.25

        toy_daily = pd.DataFrame(
            [
                {"cow_id": 1, "dim": int(d), "parity": 1, "milk_kg": float(base_curve[int(d) - 1])}
                for d in dims
            ]
            + [
                {"cow_id": 2, "dim": int(d), "parity": 1, "milk_kg": float(true_scale * base_curve[int(d) - 1])}
                for d in dims
            ]
        )
        toy_cows = pd.DataFrame(
            [
                {"cow_id": 1, "parity": 1, "pct_holstein": 0.90},
                {"cow_id": 2, "parity": 1, "pct_holstein": 0.90},
            ]
        )
        parity_curves = compute_parity_curves(toy_daily[toy_daily["cow_id"] == 1])
        pred_df, summary = build_lactation_features(toy_daily[toy_daily["cow_id"] == 2], toy_cows, parity_curves)
        computed_scale = float(summary.loc[summary["cow_id"] == 2, "scale"].iloc[0])
        self.assertAlmostEqual(computed_scale, true_scale, places=6)
        np.testing.assert_allclose(pred_df["scaled_forecast"].to_numpy(), pred_df["milk_kg"].to_numpy(), rtol=1e-5)

    def test_unit_conception_drops_duration_when_daily_only(self):
        """PRD Section 5: Duration needs hourly data; if only daily data exists, drop it and retrain."""
        _, _, insem, _ = simulate_herd(n_cows=40, seed=42, out_dir=None)
        insem_no_dur = insem.drop(columns=["estrus_duration_h"])
        bundle, metrics = fit_conception(insem_no_dur, seed=7)
        self.assertNotIn("estrus_duration_h", bundle["features"])
        self.assertEqual(metrics["status"], "trained")

    def test_leakage_checks(self):
        """Verify disjoint splits, causal rolling baselines, baseline excluding today, and train-only parity curves."""
        cows, daily, _, _ = simulate_herd(n_cows=20, seed=42, out_dir=None)
        tr, va, te = cow_train_validation_test(cows["cow_id"], seed=7)
        results = run_leakage_checks(daily, cows, tr, va, te)
        self.assertTrue(results["all_passed"])
        for k, v in results.items():
            self.assertTrue(v, f"Leakage test failed: {k}")

    def test_integration_30_cows_and_rest_service(self):
        """Integration: train and predict on 30 cows and verify metrics.json, artifacts, and REST endpoints."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_root = Path(tmp)
            report = run_training(out_root=tmp_root, n_cows_override=30)
            self.assertIn("models", report)
            for key in ("estrus", "behaviour", "conception", "lactation"):
                self.assertIn(key, report["models"])
            self.assertTrue((tmp_root / "reports" / "metrics.json").exists())
            self.assertTrue((tmp_root / "reports" / "model_cards.md").exists())
            self.assertTrue((tmp_root / "reports" / "conception_reliability.svg").exists())

            # NFR-4 traceability check on saved model artifact
            est_artifact = joblib.load(tmp_root / "models" / "estrus-hist_gradient_boosting-1.0.0.joblib")
            for trace_key in ("config", "data_hash", "code_commit", "metrics", "threshold"):
                self.assertIn(trace_key, est_artifact)

            written = run_predictions(
                daily_source=tmp_root / "data" / "daily.csv",
                out_dir=tmp_root / "out",
                models_dir=tmp_root / "models",
                data_dir=tmp_root / "data",
            )
            for out_key in ("estrus_predictions", "conception_predictions", "lactation_predictions", "behaviour_predictions"):
                self.assertIn(out_key, written)
                self.assertTrue(written[out_key].exists())

            # FR-P4 REST endpoints check
            status, health = dispatch_request("GET", "/healthz", models_dir=tmp_root / "models")
            self.assertEqual(status, 200)
            self.assertEqual(health["status"], "ok")

            daily_sample = pd.read_csv(tmp_root / "data" / "daily.csv")
            cow1_rows = daily_sample[(daily_sample["cow_id"] == 1) & (daily_sample["dim"] <= 65)].to_dict(orient="records")
            st_e, body_e = dispatch_request("POST", "/v1/estrus", {"rows": cow1_rows}, models_dir=tmp_root / "models")
            self.assertEqual(st_e, 200)
            self.assertTrue({"cow_id", "dim", "p_estrus", "alert", "threshold", "model_version"}.issubset(body_e["predictions"][0]))

            insem_sample = pd.read_csv(tmp_root / "data" / "inseminations.csv").head(3).to_dict(orient="records")
            st_c, body_c = dispatch_request("POST", "/v1/conception", {"rows": insem_sample}, models_dir=tmp_root / "models")
            self.assertEqual(st_c, 200)
            self.assertTrue({"cow_id", "dim", "semen_type", "p_conception", "model_version"}.issubset(body_c["predictions"][0]))

            st_l, body_l = dispatch_request("POST", "/v1/lactation", {"rows": cow1_rows}, models_dir=tmp_root / "models")
            self.assertEqual(st_l, 200)
            self.assertTrue({"cow_id", "dim", "milk_kg_pred", "total_305_pred", "model_version"}.issubset(body_l["predictions"][0]))

    def test_regression_300_cows_acceptance_criteria(self):
        """Regression: on 300 cows at seed 42, all PRD Section 7 acceptance targets are met."""
        metrics_path = ROOT / "reports" / "metrics.json"
        self.assertTrue(metrics_path.exists(), "reports/metrics.json must exist")
        report = json.loads(metrics_path.read_text(encoding="utf-8"))

        # 1. Estrus detector targets
        est = report["models"]["estrus"]
        self.assertGreaterEqual(est["f1"], 0.85)
        self.assertGreaterEqual(est["pr_auc"], 0.90)
        self.assertGreater(est["f1"], est["baselines"]["activity_ratio_rule"]["f1"])

        # 2. Conception model targets (5-fold group CV Brier < base-rate Brier)
        conc = report["models"]["conception"]
        base_brier = conc["base_rate_brier"]
        self.assertLess(conc["logistic_regression"]["cv_brier"], base_brier)
        self.assertLess(conc["gradient_boosting"]["cv_brier"], base_brier)

        # 3. Lactation model targets (remaining yield error <= 5%, MAE >= 30% below parity curve)
        lact = report["models"]["lactation"]
        self.assertLessEqual(lact["remaining_yield_error_pct"], 5.0)
        self.assertGreaterEqual(lact["mae_reduction_vs_parity_curve_pct"], 30.0)

        # 4. Behaviour classifier target (3-class macro F1 >= 0.85)
        beh = report["models"]["behaviour"]
        self.assertGreaterEqual(beh["group_cv_macro_f1"], 0.85)

        # 5. Automated leakage checks
        self.assertTrue(report["leakage_tests"]["all_passed"])

    def test_data2_and_data3_real_telemetry_integration(self):
        """Verify data2 (Zenodo 4064802 Precision Beef) and data3 (1 Hz Jerk) integration and provenance."""
        from src.inventory import build_inventory
        from src.data2_data3 import read_aligned_chunk, analyze_and_classify_data3_jerk

        inv = build_inventory()
        self.assertTrue(inv["data2_precision_beef"]["available"])
        self.assertEqual(inv["data2_precision_beef"]["n_animals_total"], 18)
        self.assertEqual(inv["data2_precision_beef"]["n_animals_paired_halter"], 17)
        self.assertTrue(inv["data3_preprocessed_jerk"]["available"])
        self.assertEqual(inv["data3_preprocessed_jerk"]["rows"], 12654)

        a_df, h_df = read_aligned_chunk(1, 0.25, n_lines=600)
        self.assertEqual(len(a_df), 600)
        self.assertEqual(len(h_df), 600)
        self.assertEqual(str(a_df["timestamp"].iloc[0])[:19], str(h_df["timestamp"].iloc[0])[:19])

        win3, sum3 = analyze_and_classify_data3_jerk()
        self.assertTrue(sum3["available"])
        self.assertEqual(sum3["is_synthetic"], 0)
        self.assertGreater(len(win3), 50)
        self.assertTrue((win3["is_synthetic"] == 0).all())

        # Verify data4 (Smid et al. 2024 Estrus Behaviour + Electronic Ear-Tag + Supplementary Figures)
        from src.data4_integration import (
            analyze_and_model_smid_estrus_behaviour,
            extract_and_evaluate_eartag_estrus,
            extract_supplementary_figures_priors,
        )
        self.assertTrue(inv["data4_estrus_and_eartag_pack"]["available"])
        _, smid_m, smid_df = analyze_and_model_smid_estrus_behaviour()
        self.assertTrue(smid_m["available"])
        self.assertEqual(smid_m["n_sas_valid_events"], 94)
        self.assertGreater(smid_m["estrus_duration_regressor_5fold_cv"]["r2_score"], 0.75)
        self.assertTrue((smid_df["is_synthetic"] == 0).all())

        et_df, et_m, _ = extract_and_evaluate_eartag_estrus(force_rescan=False)
        self.assertTrue(et_m["available"])
        self.assertEqual(et_m["raw_rows_streamed"], 2663644)
        self.assertEqual(et_m["n_animals_streamed"], 409)
        self.assertEqual(et_m["n_pdf_labeled_animals_matched"], 333)
        self.assertGreaterEqual(et_m["eartag_estrus_classifier_5fold_cow_cv"]["roc_auc"], 0.90)
        self.assertTrue((et_df["is_synthetic"] == 0).all())

        supp = extract_supplementary_figures_priors()
        self.assertTrue(supp["available"])


if __name__ == "__main__":
    unittest.main()
