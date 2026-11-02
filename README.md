# Estrus, Breeding and Lactation Prediction System

This repository implements the project PRD and keeps the 10-cow synthetic demo as a separate dataset. It includes the 300-cow synthetic simulator, schema validation, causal feature generation, model training and evaluation, batch predictions, and data/model reports.

## Run the pipeline

Install the pinned dependencies:

    python -m pip install -r requirements.txt

Generate the PRD tables (300 cows, DIM 1–305):

    python -m src.simulate --n-cows 300 --seed 42

Validate the PRD tables and the 10-cow SQLite database:

    python -m src.validate data/

Train all available models and write metrics, model cards, inventory, and versioned artifacts:

    python -m src.train --config config.yaml

Generate batch outputs:

    python -m src.predict --daily data/synthetic_cattle.db --out out/

Launch the interactive web UI and prediction dashboard:

    python web/server.py --port 8080

Then open [http://localhost:8080](http://localhost:8080) in your web browser (or directly open `web/index.html`).

The prediction command writes estrus, conception, lactation, and behavior outputs, plus a breeding schedule when the required model outputs are present. The Makefile target all runs the full sequence, including the repository test suite.

## Data sources

- **PRD synthetic herd:** data/cows.csv, data/daily.csv, and data/inseminations.csv; 300 cows × 305 days by default. The simulator is seeded and checks its own summary targets.
- **10-cow synthetic demo:** data/synthetic_cattle.db and the three root CSVs; 10 cows, DIM 11–110, 24,000 hourly rows, 1,000 daily rows with 7-day z-scores, and 441 synthetic IMU summaries.
- **Real collar clips:** raw labeled signals under db-cow-walking-main/. The behavior model uses the real 10 Hz clips, grouped by cow, and reports three-class results plus the optional Standing class.
- **Camera images:** 277 pen frames and 10 aerial images under Wagyu&Angus/. They are inventoried but remain out of scope for v1; no images are copied into the synthetic database.

## Model status

Estrus, conception, lactation, and behavior models are implemented. The default synthetic run's held-out estrus F1 is 0.91, conception Brier scores are below the 0.226 base-rate score, and remaining lactation yield error is below 5%. Behavior macro F1 is above the three-class target. See reports/metrics.json and reports/model_cards.md for the complete scores, splits, baselines, calibration summaries, and limitations.

Synthetic metrics check the pipeline and do not establish farm performance. Real estrus, conception, and lactation records are still needed for real-data evaluation.
