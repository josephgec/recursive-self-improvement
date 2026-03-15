.PHONY: install test lint build-images up down smoke-test clean

install:
	pip install -e .

test:
	pytest -v --tb=short

lint:
	python -m py_compile repl/src/*.py symbolic/src/*.py tracking/src/*.py sdk/*.py

build-images:
	./scripts/build_images.sh

up:
	docker compose up -d

down:
	docker compose down

smoke-test:
	python scripts/run_smoke_test.py

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
