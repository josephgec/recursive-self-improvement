# Risk Mitigations

Operational risk management across 6 domains for recursive self-improvement.

## Domains

1. **Collapse** - Model collapse detection, alpha scheduling, data reserves, halt protocols
2. **Self-Modification** - Staging environments, complexity budgets, blast radius, quarantine
3. **Reward** - Adversarial evaluation, eval rotation, reward auditing, sanity checks
4. **Cost** - Budget management, circuit breakers, cost forecasting, optimization
5. **Constraints** - Graduated relaxation, compensation monitoring, tightness detection
6. **Publication** - Deadline tracking, draft generation, fallback planning, readiness

## Commands

```bash
make install    # Install with dev dependencies
make test       # Run tests
make coverage   # Run tests with coverage report
make clean      # Clean build artifacts
```

## Architecture

- `src/` - Domain implementations
- `src/orchestration/` - Cross-domain risk registry and dashboard
- `src/analysis/` - Retrospective analysis and reporting
- `configs/` - YAML configuration files
- `scripts/` - CLI entry points
- `tests/` - Pytest test suite (target >= 90% coverage)
- `data/` - Runtime data directories

All modules are fully self-contained with deterministic mocks. No external services required.
