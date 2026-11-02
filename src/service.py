"""Thin REST service wrapper for Estrus, Conception, and Lactation scoring (PRD Section 8, FR-P4).

Endpoints:
- GET  /healthz        -> {"status": "ok", "models_loaded": [...]}
- POST /v1/estrus      -> rows with cow_id, dim, p_estrus, alert, threshold, model_version
- POST /v1/conception  -> rows with cow_id, dim, semen_type, p_conception, model_version
- POST /v1/lactation   -> rows with cow_id, dim, milk_kg_pred, total_305_pred, model_version
"""
from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.features import estrus_features, estrus_features_prd, build_conception_features
from src.models.lactation import predict_lactation

ROOT = Path(__file__).resolve().parents[1]


def score_estrus_payload(rows: list[dict[str, Any]], models_dir: Path | None = None) -> list[dict[str, Any]]:
    m_dir = models_dir or (ROOT / "models")
    df = pd.DataFrame(rows)
    if "activity" in df.columns:
        bundle = joblib.load(m_dir / "estrus-hist_gradient_boosting-1.0.0.joblib")
        feat_df = estrus_features_prd(df, warmup_dim=int(df["dim"].min()))
        probs = bundle["model"].predict_proba(feat_df[bundle["features"]])[:, 1]
        thresh = float(bundle["threshold"])
        version = str(bundle["model_version"])
    else:
        bundle = joblib.load(m_dir / "estrus-hist_gradient_boosting-0.1.0.joblib")
        feat_df = estrus_features(df)
        probs = bundle["model"].predict_proba(feat_df[bundle["features"]])[:, 1]
        thresh = float(bundle["threshold"])
        version = str(bundle["model_version"])

    out = pd.DataFrame(
        {
            "cow_id": feat_df["cow_id"],
            "dim": feat_df["dim"].astype(int),
            "p_estrus": np.round(probs, 6),
            "alert": (probs >= thresh).astype(int),
            "threshold": thresh,
            "model_version": version,
        }
    )
    return out.to_dict(orient="records")


def score_conception_payload(rows: list[dict[str, Any]], models_dir: Path | None = None) -> list[dict[str, Any]]:
    m_dir = models_dir or (ROOT / "models")
    df = pd.DataFrame(rows)
    bundle = joblib.load(m_dir / "conception-calibrated_ensemble-1.0.0.joblib")
    feat_df = build_conception_features(df)
    p_lr = bundle["model"].predict_proba(feat_df[bundle["features"]])[:, 1]
    p_hgb = bundle["boosting_model"].predict_proba(feat_df[bundle["features"]])[:, 1]
    p_conc = np.round(0.5 * p_lr + 0.5 * p_hgb, 4)
    out = pd.DataFrame(
        {
            "cow_id": feat_df["cow_id"],
            "dim": feat_df["dim"].astype(int),
            "semen_type": feat_df["semen_type"],
            "p_conception": p_conc,
            "model_version": str(bundle["model_version"]),
        }
    )
    return out.to_dict(orient="records")


def score_lactation_payload(rows: list[dict[str, Any]], models_dir: Path | None = None) -> list[dict[str, Any]]:
    m_dir = models_dir or (ROOT / "models")
    df = pd.DataFrame(rows)
    bundle = joblib.load(m_dir / "lactation-scaled_parity_hgb-1.0.0.joblib")
    # Ensure target rows 61..305 exist for each cow if only early window 5..60 was supplied
    full_rows = []
    for cow_id, grp in df.groupby("cow_id", sort=False):
        existing_dims = set(grp["dim"].astype(int))
        parity = int(grp["parity"].iloc[0])
        pct_hol = float(grp["pct_holstein"].iloc[0]) if "pct_holstein" in grp.columns else 0.88
        full_rows.append(grp)
        missing_dims = [d for d in range(61, 306) if d not in existing_dims]
        if missing_dims:
            full_rows.append(
                pd.DataFrame(
                    {
                        "cow_id": cow_id,
                        "dim": missing_dims,
                        "parity": parity,
                        "pct_holstein": pct_hol,
                        "milk_kg": np.nan,
                    }
                )
            )
    full_df = pd.concat(full_rows, ignore_index=True)
    pred = predict_lactation(full_df, None, bundle)
    return pred[["cow_id", "dim", "milk_kg_pred", "total_305_pred", "model_version"]].to_dict(orient="records")


def dispatch_request(
    method: str,
    path: str,
    payload: dict[str, Any] | list[dict[str, Any]] | None = None,
    models_dir: Path | None = None,
) -> tuple[int, dict[str, Any]]:
    m_dir = models_dir or (ROOT / "models")
    if method.upper() == "GET" and path.rstrip("/") == "/healthz":
        models_present = sorted(p.name for p in m_dir.glob("*.joblib")) if m_dir.exists() else []
        return 200, {"status": "ok", "models_loaded": models_present}

    if method.upper() != "POST":
        return 405, {"error": f"Method {method} not allowed for {path}"}

    rows = payload.get("rows", []) if isinstance(payload, dict) else (payload or [])
    if not isinstance(rows, list) or len(rows) == 0:
        return 400, {"error": "Request body must be a non-empty list of rows or {'rows': [...]}"}

    clean_path = path.rstrip("/")
    if clean_path == "/v1/estrus":
        return 200, {"predictions": score_estrus_payload(rows, models_dir=m_dir)}
    if clean_path == "/v1/conception":
        return 200, {"predictions": score_conception_payload(rows, models_dir=m_dir)}
    if clean_path == "/v1/lactation":
        return 200, {"predictions": score_lactation_payload(rows, models_dir=m_dir)}
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
            self._send_json(400, {"error": f"Invalid JSON: {exc}"})
            return
        status, body = dispatch_request("POST", self.path, payload, models_dir=self.models_dir)
        self._send_json(status, body)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    server = HTTPServer((args.host, args.port), PredictionRequestHandler)
    print(f"Serving cattle prediction REST API on http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()

