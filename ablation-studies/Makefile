.PHONY: test coverage lint clean run-all

test:
	python -m pytest tests/ -v

coverage:
	python -m pytest tests/ --cov=src --cov-report=term-missing --cov-report=html

lint:
	python -m py_compile src/suites/base.py
	python -m py_compile src/execution/runner.py

clean:
	rm -rf data/results/* data/figures/* data/tables/* data/reports/*
	rm -rf __pycache__ .pytest_cache htmlcov .coverage
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

run-all:
	python scripts/run_all_suites.py

paper-assets:
	python scripts/generate_paper_assets.py

check-power:
	python scripts/check_power.py
