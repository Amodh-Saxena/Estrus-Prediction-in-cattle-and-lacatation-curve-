"""API contract and schema validation tests (FR-D4)."""
from __future__ import annotations

import pandas as pd

from src.service import dispatch_request


def test_api_healthz() -> None:
    code, body = dispatch_request("GET", "/healthz")
    assert code == 200
    assert body["status"] == "ok"
    assert "model_version" in body


def test_api_rejects_empty_payload() -> None:
    code, body = dispatch_request("POST", "/v1/estrus", {"rows": []})
    assert code == 422
    assert "errors" in body


def test_api_rejects_missing_fields() -> None:
    bad_rows = [{"animal_id": "cow_01", "steps_sum": 100}]  # missing date, temp_mean, temp_q90
    code, body = dispatch_request("POST", "/v1/estrus", {"rows": bad_rows})
    assert code == 422
    assert any("missing required fields" in err for err in body.get("errors", []))


def test_api_rejects_out_of_range_temperature() -> None:
    fever_rows = [{
        "animal_id": "cow_01",
        "date": "2025-01-01",
        "steps_sum": 500,
        "temp_mean": 65.0,  # Physically impossible skin temperature
        "temp_q90": 38.5,
    }]
    code, body = dispatch_request("POST", "/v1/estrus", {"rows": fever_rows})
    assert code == 422
    assert any("outside valid physiological range" in err for err in body.get("errors", []))


def test_api_scores_valid_series() -> None:
    valid_rows = [
        {
            "animal_id": "cow_01",
            "date": d.strftime("%Y-%m-%d"),
            "steps_sum": 120.0 + (50.0 if i == 15 else 0.0),
            "temp_mean": 38.6 + (0.4 if i == 15 else 0.0),
            "temp_q90": 38.8 + (0.4 if i == 15 else 0.0),
        }
        for i, d in enumerate(pd.date_range("2025-01-01", periods=20))
    ]
    code, body = dispatch_request("POST", "/v1/estrus", {"rows": valid_rows})
    assert code == 200
    assert "results" in body
    results = body["results"]
    assert len(results) == 20
    first = results[0]
    assert "score" in first
    assert "alert" in first
    assert "threshold" in first
    assert "model_version" in first
    assert first["data_source"] == "real"

