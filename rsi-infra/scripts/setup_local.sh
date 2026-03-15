#!/usr/bin/env bash
# setup_local.sh — Install dependencies and run smoke tests.
# Usage:  ./scripts/setup_local.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "=== rsi-infra local setup ==="
echo "Project root: $PROJECT_ROOT"

# 1. Install in editable mode
echo ""
echo "--- Installing dependencies ---"
cd "$PROJECT_ROOT"
pip install -e . 2>&1 | tail -5

# 2. Run smoke tests
echo ""
echo "--- Running smoke tests ---"
python scripts/run_smoke_test.py all

echo ""
echo "=== Setup complete ==="
