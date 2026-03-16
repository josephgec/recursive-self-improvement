.PHONY: test test-cov lint clean evolve compare ablation report

test:
	python -m pytest tests/ -v

test-cov:
	python -m pytest tests/ -v --cov=src --cov-report=term-missing --cov-report=html

lint:
	python -m py_compile src/genome/prompt_genome.py
	python -m py_compile src/ga/engine.py

clean:
	rm -rf __pycache__ .pytest_cache htmlcov .coverage
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true

evolve:
	python scripts/run_evolution.py

compare:
	python scripts/run_comparison.py

ablation:
	python scripts/run_ablation.py

report:
	python scripts/generate_report.py
