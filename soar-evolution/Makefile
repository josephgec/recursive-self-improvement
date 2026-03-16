.PHONY: test test-cov lint clean run-search run-benchmark

test:
	python -m pytest tests/ -v

test-cov:
	python -m pytest tests/ -v --cov=src --cov-report=term-missing --cov-report=html

lint:
	python -m py_compile src/arc/grid.py
	python -m py_compile src/arc/loader.py
	python -m py_compile src/arc/evaluator.py

clean:
	rm -rf __pycache__ .pytest_cache htmlcov .coverage
	find . -name '*.pyc' -delete
	find . -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true

run-search:
	python scripts/run_search.py

run-benchmark:
	python scripts/run_benchmark.py

run-single:
	python scripts/run_single_task.py

analyze:
	python scripts/analyze_search.py

report:
	python scripts/generate_report.py
