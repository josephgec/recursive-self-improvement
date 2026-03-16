.PHONY: test coverage lint clean run-all

test:
	python -m pytest tests/ -v

coverage:
	python -m pytest tests/ --cov=src --cov-report=term-missing --cov-report=html

lint:
	python -m flake8 src/ tests/

clean:
	rm -rf __pycache__ .pytest_cache htmlcov .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +

run-all:
	python scripts/run_all_experiments.py

run-frequency:
	python scripts/run_experiment.py --experiment modification_frequency

run-hindsight:
	python scripts/run_experiment.py --experiment hindsight_target

run-depth:
	python scripts/run_experiment.py --experiment rlm_depth

generate-config:
	python scripts/generate_optimal_config.py

interaction-study:
	python scripts/run_interaction_study.py
