.PHONY: install test lint build-ctm-table run-synthesis run-comparison report clean

install:
	pip install -e ".[dev]"

test:
	python -m pytest tests/ -v --cov=src --cov-report=term-missing

lint:
	ruff check src/ tests/ scripts/
	ruff format --check src/ tests/ scripts/

format:
	ruff format src/ tests/ scripts/
	ruff check --fix src/ tests/ scripts/

build-ctm-table:
	python scripts/build_ctm_table.py

run-synthesis:
	python scripts/run_synthesis.py

run-comparison:
	python scripts/run_comparison.py

evaluate-library:
	python scripts/evaluate_library.py

report:
	python scripts/generate_report.py

clean:
	rm -rf __pycache__ .pytest_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
