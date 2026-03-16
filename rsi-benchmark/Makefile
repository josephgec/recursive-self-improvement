.PHONY: test test-cov lint clean run-eval run-ablation run-report

test:
	python -m pytest tests/ -v

test-cov:
	python -m pytest tests/ -v --cov=src --cov-report=term-missing --cov-report=html

lint:
	python -m py_compile src/benchmarks/registry.py
	python -m py_compile src/evaluation/iteration_evaluator.py

clean:
	rm -rf __pycache__ .pytest_cache htmlcov .coverage
	find . -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true
	find . -name '*.pyc' -delete 2>/dev/null || true

run-eval:
	python scripts/run_evaluation.py

run-single:
	python scripts/run_single_benchmark.py

run-ablation:
	python scripts/run_ablation.py

run-collapse:
	python scripts/run_collapse_comparison.py

run-report:
	python scripts/generate_report.py

package:
	python scripts/package_deliverables.py
