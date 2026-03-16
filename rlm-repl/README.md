# RLM REPL Environment

Production-grade sandboxed code execution for the Recursive Language Model. 6-layer defense-in-depth safety stack. Three backends: LocalREPL (dev), DockerREPL (production), ModalREPL (cloud scale).

## Quick Start

```bash
pip install -e .
make test    # 357 tests, 95% coverage
python scripts/run_safety_audit.py
```

## 6 Safety Layers

| Layer | What | Mechanism |
|-------|------|-----------|
| AST scan | Reject forbidden code | Static analysis before execution |
| Timeout | Kill long-running code | SIGALRM / process kill |
| Memory | Kill memory-hungry code | cgroups / setrlimit |
| Output | Truncate excessive output | String truncation |
| Depth | Prevent infinite recursion | Counter check at spawn |
| Cascade | Kill orphan processes | Tree walk + kill |

## FINAL Protocol

```python
FINAL("The answer is 42")      # Direct text result
FINAL_VAR("my_dataframe")      # Result by variable reference
```

## License

MIT
