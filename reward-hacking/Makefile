.PHONY: test coverage lint clean install

install:
	pip install -e ".[dev]"

test:
	python -m pytest tests/ -v

coverage:
	python -m pytest tests/ --cov=src --cov-report=term-missing --cov-report=html -v

lint:
	python -m py_compile src/eppo/trainer.py
	python -m py_compile src/bounding/reward_clipper.py
	python -m py_compile src/energy/energy_tracker.py
	python -m py_compile src/detection/composite_detector.py
	python -m py_compile src/deliverables/phase_gate.py

clean:
	rm -rf __pycache__ .pytest_cache htmlcov .coverage
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true

run-eppo:
	python scripts/run_eppo_training.py

monitor-energy:
	python scripts/monitor_energy.py

check-hacking:
	python scripts/check_reward_hacking.py

package-safety:
	python scripts/package_safety.py

report:
	python scripts/generate_report.py
