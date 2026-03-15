#!/usr/bin/env bash
# build_images.sh — Build Docker images for rsi-infra subsystems.
# This is a placeholder for future Docker-based deployments.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "=== rsi-infra Docker image build ==="
echo "Project root: $PROJECT_ROOT"
echo ""
echo "NOTE: Docker image building is not yet implemented."
echo "The following images will be built in future iterations:"
echo "  - rsi-repl:latest        (sandboxed REPL)"
echo "  - rsi-symbolic:latest    (SymPy + Z3 runner)"
echo ""
echo "For now, use the local backends:"
echo "  python scripts/run_smoke_test.py all"
