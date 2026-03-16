.PHONY: test test-cov lint run-comparison run-single clean

test:
	python -m pytest tests/ -v

test-cov:
	python -m pytest tests/ --cov=src --cov-report=term-missing --cov-report=html -v

lint:
	python -m py_compile src/hybrid/pipeline.py
	python -m py_compile src/integrative/pipeline.py
	python -m py_compile src/evaluation/benchmark_suite.py

run-comparison:
	python scripts/run_comparison.py

run-single:
	python scripts/run_single_axis.py --axis generalization

train-integrative:
	python scripts/train_integrative.py

package:
	python scripts/package_deliverables.py

report:
	python scripts/generate_report.py

clean:
	rm -rf __pycache__ .pytest_cache htmlcov .coverage
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
