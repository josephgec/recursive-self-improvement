# RLM-REPL

Production-grade sandboxed code execution environment for the RLM architecture.

## Quick Start

```bash
make install
make test
make test-cov
```

## Architecture

6-layer safety stack:
1. AST scanning (forbidden imports, builtins, dunder access, obfuscation)
2. Execution timeout (signal-based with threading fallback)
3. Memory limiting (resource-based process limits)
4. Output limiting (truncation with notification)
5. Depth limiting (spawn depth control)
6. Cascade killing (parent-child process tree management)

## Backends

- **LocalREPL**: exec()-based, restricted builtins, full safety stack
- **DockerREPL**: Docker container isolation, falls back to LocalREPL
- **ModalREPL**: Stub for Modal cloud execution (not yet implemented)

## Testing

All tests run without Docker or Modal using LocalREPL.

```bash
make test-cov   # Target >= 90% coverage
```

## Project Layout

- `src/interface/` - ABC and types
- `src/safety/` - 6-layer safety stack
- `src/backends/` - Execution backends
- `src/memory/` - Variable store, serialization, snapshots
- `src/protocol/` - FINAL protocol for result extraction
- `src/pool/` - REPL pool management
- `configs/` - YAML configuration files
- `data/malicious_samples/` - Test payloads for safety auditing
