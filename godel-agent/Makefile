.PHONY: install test lint run-debug run-math run-generalist inspect-history report clean

install:
	pip install -e .

test:
	pytest tests/ -v --tb=short

lint:
	python -m py_compile src/**/*.py

run-debug:
	python scripts/run_agent.py --config configs/debug.yaml --tasks math --iterations 5

run-math:
	python scripts/run_agent.py --config configs/math_specialist.yaml --tasks math

run-generalist:
	python scripts/run_agent.py --config configs/generalist.yaml --tasks all

inspect-history:
	python scripts/inspect_history.py data/audit_logs/latest/

report:
	python scripts/generate_report.py

clean:
	rm -rf data/checkpoints data/audit_logs data/reports
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
