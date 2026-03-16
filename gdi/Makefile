.PHONY: test coverage lint clean install

install:
	pip install -e ".[dev]"

test:
	python -m pytest tests/ -v

coverage:
	python -m pytest tests/ --cov=src --cov-report=term-missing --cov-report=html -v

lint:
	python -m py_compile src/signals/base.py
	python -m py_compile src/composite/gdi.py

clean:
	rm -rf __pycache__ .pytest_cache htmlcov .coverage
	find . -name '__pycache__' -exec rm -rf {} +
	find . -name '*.pyc' -delete

collect-reference:
	python scripts/collect_reference.py

calibrate:
	python scripts/calibrate_thresholds.py

monitor:
	python scripts/run_monitoring.py

compute-gdi:
	python scripts/compute_gdi.py

report:
	python scripts/generate_report.py
