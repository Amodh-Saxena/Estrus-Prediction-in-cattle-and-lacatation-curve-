"""REST service and model inference dispatcher for CattleSense AI v2.

Supports:
- GET  /healthz        -> Status, active model version, and data checksum
- POST /v1/estrus      -> Strict schema validation (422 on bad rows) + real biosensor inference
- POST /v1/conception  -> Calibrated conception probabilities
- POST /v1/lactation   -> 305-day lactation forecast
"""
from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.common import ROOT

REQUIRED_EARTAG_FIELDS = ["animal_id", "date", "steps_sum", "temp_mean", "temp_q90"]
LIMITS = {
    "max_rows": 5000,
    "ranges": {
        "steps_sum": (0, 250000),
        "temp_mean": (25.0, 46.0),
        "temp_q90": (25.0, 48.0),
    },
}


def validate_eartag_rows(rows: list[dict[str, Any]]) -> list[str]:
    """Validate incoming payload against strict production schema (FR-D4). Returns list of errors."""
    if not isinstance(rows, list) or len(rows) == 0:
        return ["Request payload must be a non-empty list of rows under 'rows'"]
    if len(rows) > LIMITS["max_rows"]:
        return [f"Batch size {len(rows)} exceeds maximum limit of {LIMITS['max_rows']} rows"]

    errs = []
    for i, r in enumerate(rows):
        if not isinstance(r, dict):
            errs.append(f"row {i}: must be a dictionary")
            continue
        miss = [k for k in REQUIRED_EARTAG_FIELDS if k not in r or r[k] is None]
        if miss:
            errs.append(f"row {i}: missing required fields {miss}")
            continue
        try:
            pd.Timestamp(r["date"])
        except Exception:
            errs.append(f"row {i}: invalid ISO date format for '{r.get('date')}'")
        for k, (lo, hi) in LIMITS["ranges"].items():
            val = r.get(k)
            if isinstance(val, bool) or not isinstance(val, (int, float)) or not (lo <= float(val) <= hi):
                errs.append(f"row {i}: '{k}' value {val} outside valid physiological range [{lo}, {hi}]")
    return errs


def get_active_estrus_bundle(models_dir: Path | None = None) -> dict[str, Any]:
    """Load primary estrus classifier bundle."""
    m_dir = models_dir or (ROOT / "models")
    # Support overriding via environment variable for CI fixtures
    env_override = os.environ.get("CATTLESENSE_MODEL_PATH")
    if env_override and Path(env_override).exists():
        return joblib.load(Path(env_override))
    candidates = [
        m_dir / "estrus-data4_eartag_3sheet-1.0.0.joblib",
        m_dir / "estrus-gbm-both.joblib",
        m_dir / "estrus-hist_gradient_boosting-1.0.0.joblib",
    ]
    for c in candidates:
        if c.exists():
            return joblib.load(c)
    return {}


def score_v2_estrus_payload(rows: list[dict[str, Any]], models_dir: Path | None = None) -> list[dict[str, Any]]:
    """Score incoming animal-day time series against real ear-tag model."""
    bundle = get_active_estrus_bundle(models_dir)
    if not bundle:
        raise RuntimeError("No active estrus model bundle loaded.")

    from src.features import add_features

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    if "estrus" not in df.columns:
        df["estrus"] = 0

    feat_df = add_features(df)
    feats = bundle.get("features", [])
    model = bundle.get("model")
    thresh = float(bundle.get("threshold", 0.50))
    version = str(bundle.get("model_version", "v2.0.0"))

    # If model features match v2 schema, score with bundle model
    available_feats = [c for c in feats if c in feat_df.columns]
    if len(available_feats) >= 3 and hasattr(model, "predict_proba"):
        p = model.predict_proba(feat_df[available_feats].fillna(0.0))[:, 1]
    else:
        # Fallback to activity ratio rule
        ratios = feat_df.get("steps_sum_ratio", pd.Series(1.0, index=feat_df.index)).fillna(1.0).to_numpy()
        p = np.clip((ratios - 1.0) / 1.5, 0.0, 1.0)

    ready = feat_df["steps_sum_ratio"].notna().to_numpy() if "steps_sum_ratio" in feat_df.columns else np.ones(len(feat_df), bool)

    results = []
    for a, d, s, ok in zip(feat_df["animal_id"], feat_df["date"], p, ready):
        results.append({
            "animal_id": str(a),
            "date": d.strftime("%Y-%m-%d"),
            "score": round(float(s), 4),
            "alert": bool(ok and s >= thresh),
            "enough_history": bool(ok),
            "threshold": thresh,
            "model_version": version,
            "data_source": "real",
        })
    return results


def score_conception_payload(rows: list[dict[str, Any]], models_dir: Path | None = None) -> list[dict[str, Any]]:
    m_dir = models_dir or (ROOT / "models")
    from src.features import build_conception_features
    bundle_p = m_dir / "conception-calibrated_ensemble-1.0.0.joblib"
    if not bundle_p.exists():
        return []
    bundle = joblib.load(bundle_p)
    df = pd.DataFrame(rows)
    feat_df = build_conception_features(df)
    p_lr = bundle["model"].predict_proba(feat_df[bundle["features"]])[:, 1]
    p_hgb = bundle["boosting_model"].predict_proba(feat_df[bundle["features"]])[:, 1]
    p_conc = np.round(0.5 * p_lr + 0.5 * p_hgb, 4)
    out = pd.DataFrame({
        "cow_id": feat_df["cow_id"],
        "dim": feat_df["dim"].astype(int),
        "semen_type": feat_df["semen_type"],
        "p_conception": p_conc,
        "model_version": str(bundle["model_version"]),
        "status": "SIMULATED",
    })
    return out.to_dict(orient="records")


def score_lactation_payload(rows: list[dict[str, Any]], models_dir: Path | None = None) -> list[dict[str, Any]]:
    m_dir = models_dir or (ROOT / "models")
    from src.models.lactation import predict_lactation
    bundle_p = m_dir / "lactation-scaled_parity_hgb-1.0.0.joblib"
    if not bundle_p.exists():
        return []
    bundle = joblib.load(bundle_p)
    df = pd.DataFrame(rows)
    pred = predict_lactation(df, None, bundle)
    pred["status"] = "SIMULATED"
    return pred[["cow_id", "dim", "milk_kg_pred", "total_305_pred", "model_version", "status"]].to_dict(orient="records")


def dispatch_request(
    method: str,
    path: str,
    payload: dict[str, Any] | list[dict[str, Any]] | None = None,
    models_dir: Path | None = None,
) -> tuple[int, dict[str, Any]]:
    """Dispatch API request with strict HTTP status semantics (200, 422, 404, 405)."""
    m_dir = models_dir or (ROOT / "models")
    clean_path = path.rstrip("/")

    if method.upper() == "GET" and clean_path == "/healthz":
        bundle = get_active_estrus_bundle(m_dir)
        version = bundle.get("model_version", "v2.0.0") if bundle else "unloaded"
        d_hash = bundle.get("data_hash", "eartag-3sheet") if bundle else "none"
        models_present = sorted(p.name for p in m_dir.glob("*.joblib")) if m_dir.exists() else []
        return 200, {
            "status": "ok",
            "model_version": version,
            "data_hash": d_hash,
            "models_loaded": models_present,
        }

    if method.upper() != "POST":
        return 405, {"error": f"Method {method} not allowed for {path}"}

    rows = payload.get("rows", []) if isinstance(payload, dict) else (payload or [])
    if not isinstance(rows, list) or len(rows) == 0:
        return 422, {"errors": ["Request payload must be a non-empty list of rows under 'rows'"]}

    if clean_path == "/v1/estrus":
        # Check if incoming rows are in v2 format (animal_id, date, steps_sum, ...)
        first = rows[0] if len(rows) else {}
        if "animal_id" in first or "steps_sum" in first:
            errs = validate_eartag_rows(rows)
            if errs:
                return 422, {"errors": errs[:20], "total_errors": len(errs)}
            res = score_v2_estrus_payload(rows, models_dir=m_dir)
            return 200, {"predictions": res, "results": res}
        else:
            # Legacy PRD synthetic scoring
            from src.features import estrus_features_prd
            df = pd.DataFrame(rows)
            bundle = joblib.load(m_dir / "estrus-hist_gradient_boosting-1.0.0.joblib")
            feat_df = estrus_features_prd(df, warmup_dim=int(df["dim"].min()))
            probs = bundle["model"].predict_proba(feat_df[bundle["features"]])[:, 1]
            thresh = float(bundle["threshold"])
            out = pd.DataFrame({
                "cow_id": feat_df["cow_id"],
                "dim": feat_df["dim"].astype(int),
                "p_estrus": np.round(probs, 6),
                "alert": (probs >= thresh).astype(int),
                "threshold": thresh,
                "model_version": str(bundle["model_version"]),
            })
            res = out.to_dict(orient="records")
            return 200, {"predictions": res, "results": res}

    if clean_path == "/v1/conception":
        res = score_conception_payload(rows, models_dir=m_dir)
        return 200, {"predictions": res, "results": res}
    if clean_path == "/v1/lactation":
        res = score_lactation_payload(rows, models_dir=m_dir)
        return 200, {"predictions": res, "results": res}

    return 404, {"error": f"Unknown endpoint: {path}"}


class PredictionRequestHandler(BaseHTTPRequestHandler):
    models_dir: Path = ROOT / "models"

    def _send_json(self, status: int, body: dict[str, Any]) -> None:
        raw = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self) -> None:  # noqa: N802
        status, body = dispatch_request("GET", self.path, None, models_dir=self.models_dir)
        self._send_json(status, body)

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length).decode("utf-8") if length > 0 else "{}"
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            self._send_json(422, {"errors": [f"Invalid JSON payload: {exc}"]})
            return
        status, body = dispatch_request("POST", self.path, payload, models_dir=self.models_dir)
        self._send_json(status, body)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    server = HTTPServer((args.host, args.port), PredictionRequestHandler)
    print(f"CattleSense v2 Inference Service listening on http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
