.PHONY: all simulate database validate train predict test serve

all: simulate validate train predict test

simulate:
	python -m src.simulate --n-cows 300 --seed 42

database:
	python build_database.py

validate:
	python -m src.validate

train:
	python -m src.train --config config.yaml

predict:
	python -m src.predict --daily data/synthetic_cattle.db --out out

test:
	python -m unittest discover -s tests -v

serve:
	python web/server.py --port 8080
