"""Raw data verification and checksum auditing.

Usage:
  python scripts/fetch_data.py --write    # Generate data/checksums.json
  python scripts/fetch_data.py            # Verify existing checksums
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
RAW_DIRS = ["data2", "data3", "data4", "db-cow-walking-main"]
CHECKSUMS_FILE = ROOT / "data" / "checksums.json"


def sha256_file(path: Path, block: int = 1 << 20) -> str:
    """Compute SHA-256 hash streaming in 1 MB blocks."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(block):
            h.update(chunk)
    return h.hexdigest()


def collect_hashes() -> dict[str, str]:
    """Scan all raw folders and return path-to-hash dictionary."""
    records = {}
    for d in RAW_DIRS:
        dp = ROOT / d
        if not dp.exists():
            continue
        for p in sorted(dp.rglob("*")):
            if p.is_file():
                rel = str(p.relative_to(ROOT)).replace("\\", "/")
                records[rel] = sha256_file(p)
    return records


def main() -> None:
    if "--write" in sys.argv:
        print("Scanning raw datasets and computing SHA-256 checksums...")
        hashes = collect_hashes()
        CHECKSUMS_FILE.parent.mkdir(parents=True, exist_ok=True)
        CHECKSUMS_FILE.write_text(json.dumps(hashes, indent=2), encoding="utf-8")
        print(f"Recorded {len(hashes)} file checksums in {CHECKSUMS_FILE}")
        return

    if not CHECKSUMS_FILE.exists():
        print(f"Checksum file {CHECKSUMS_FILE} does not exist. Creating it now...")
        hashes = collect_hashes()
        CHECKSUMS_FILE.write_text(json.dumps(hashes, indent=2), encoding="utf-8")
        print(f"Recorded {len(hashes)} file checksums in {CHECKSUMS_FILE}")
        return

    saved = json.loads(CHECKSUMS_FILE.read_text(encoding="utf-8"))
    current = collect_hashes()
    mismatches = []
    for rel_path, expected in saved.items():
        if rel_path not in current:
            mismatches.append(f"MISSING: {rel_path}")
        elif current[rel_path] != expected:
            mismatches.append(f"MODIFIED: {rel_path}")

    if mismatches:
        print("Data integrity check FAILED:")
        for m in mismatches:
            print("  ", m)
        sys.exit(1)
    else:
        print(f"All {len(saved)} raw data files verified successfully against SHA-256 signatures.")


if __name__ == "__main__":
    main()

