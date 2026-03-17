# Mechanistic Interpretability Integration

Monitors model INTERNAL representations to detect silent reorganization, reward hacking, and deceptive alignment signals. Three detection levels: activation diffs, head tracking, and behavioral-internal divergence.

## Quick Start

```bash
pip install -e .
make test    # 201 tests, 99% coverage
python scripts/probe_model.py --output data/snapshots/
python scripts/check_anomalies.py --iteration 5
```

## Three Detection Levels

| Level | Detects | When |
|-------|---------|------|
| Activation diff | Structural changes per layer | Every modification |
| Head tracking | Specialization shifts, reward correlation | During training |
| Divergence detector | Internal change without behavioral change | Every iteration |

## Deceptive Alignment Probes

- Monitoring sensitivity: does the model respond to "you are being evaluated"?
- Latent capabilities: does the model encode knowledge it doesn't express?
- Context-dependent safety: does safety behavior depend on evaluation cues?

## License

MIT
