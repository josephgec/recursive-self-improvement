.PHONY: install test lint download-data run-audit clean

install:
	pip install -e .

test:
	pytest tests/ -v --tb=short

lint:
	python -m py_compile src/**/*.py

download-data:
	python scripts/download_data.py

run-audit:
	python scripts/run_audit.py --config configs/default.yaml

clean:
	rm -rf data/processed data/embeddings data/models data/reserve
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
