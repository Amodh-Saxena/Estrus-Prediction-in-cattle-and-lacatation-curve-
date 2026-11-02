# CattleSense AI v2 Makefile
PY=python
.PHONY: all fetch data audit train web test serve simulate database validate predict

fetch:
	$(PY) scripts/fetch_data.py

data:
	$(PY) -m src.ingest.data4_eartag
	$(PY) -m src.ingest.smid
	$(PY) -m src.ingest.data2_collar

audit:
	$(PY) -m src.audit

train:
	$(PY) -m src.train --config config.yaml

web:
	$(PY) -m src.export_web

test:
	$(PY) -m pytest -q

serve:
	$(PY) web/server.py --port 8080

simulate:
	$(PY) -m src.simulate --n-cows 300 --seed 42

database:
	$(PY) simulation/build_database.py

validate:
	$(PY) -m src.validate

predict:
	$(PY) -m src.predict --daily data/synthetic_cattle.db --out out

all: fetch data audit train web test
