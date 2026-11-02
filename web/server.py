"""Local development and dashboard web server for CattleSense AI.

Serves the interactive UI at http://127.0.0.1:8080 and handles REST prediction endpoints:
- GET  /              -> Serves web/index.html
- GET  /data.json     -> Serves web/data.json
- GET  /healthz       -> Health check and model registry status
- POST /v1/estrus     -> Scored estrus predictions
- POST /v1/conception -> Calibrated conception probabilities
- POST /v1/lactation  -> 305-day lactation forecast
"""
from __future__ import annotations

import argparse
from http.server import HTTPServer, SimpleHTTPRequestHandler
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.service import dispatch_request

ROOT = Path(__file__).resolve().parents[1]
WEB_DIR = ROOT / "web"


class CattleDashboardHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(WEB_DIR), **kwargs)

    def do_GET(self) -> None:  # noqa: N802
        if self.path.rstrip("/") == "/healthz":
            status, body = dispatch_request("GET", "/healthz")
            self._send_json(status, body)
            return
        if self.path == "/":
            self.path = "/index.html"
        super().do_GET()

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length).decode("utf-8") if length > 0 else "{}"
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            self._send_json(400, {"error": f"Invalid JSON payload: {exc}"})
            return
        status, body = dispatch_request("POST", self.path, payload)
        self._send_json(status, body)

    def _send_json(self, status: int, body: dict) -> None:
        raw = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1", help="Host address (default 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8080, help="Port (default 8080)")
    args = parser.parse_args()

    server = HTTPServer((args.host, args.port), CattleDashboardHandler)
    print(f"\n=======================================================")
    print(f" CattleSense AI Dashboard running at:")
    print(f" -> http://{args.host}:{args.port}")
    print(f"=======================================================\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping CattleSense AI Dashboard server.")


if __name__ == "__main__":
    main()

