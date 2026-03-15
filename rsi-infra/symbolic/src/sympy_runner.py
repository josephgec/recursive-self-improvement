"""SymPy execution runner with subprocess isolation."""

from __future__ import annotations

from symbolic.src.executor import run_in_subprocess
from symbolic.src.result_types import SymPyResult

# Namespace setup code executed before user code in the subprocess.
_SYMPY_SETUP = """\
from sympy.abc import *
from sympy import *
from sympy.parsing.sympy_parser import parse_expr
"""


class SymPyRunner:
    """Execute SymPy code and verify symbolic expressions.

    All code runs in a fresh subprocess so that timeouts, segfaults, and
    memory leaks are isolated from the caller.

    Parameters
    ----------
    timeout:
        Maximum wall-clock seconds for a single execution.
    max_memory_mb:
        Soft memory hint (not enforced at OS level in subprocess mode).
    """

    def __init__(self, timeout: float = 30.0, max_memory_mb: int = 2048) -> None:
        self.timeout = timeout
        self.max_memory_mb = max_memory_mb

    def execute(self, code: str) -> SymPyResult:
        """Execute SymPy code and return a structured result.

        The code is run with ``from sympy import *`` already available.
        If the code defines a variable named ``result``, that is used as
        the primary expression.  Otherwise the runner attempts to capture
        the last assigned variable.
        """
        # Wrap code to capture result metadata
        wrapped = _RESULT_WRAPPER.format(user_code=code)

        raw = run_in_subprocess(
            code=wrapped,
            namespace_setup=_SYMPY_SETUP,
            timeout=self.timeout,
        )

        if not raw["success"]:
            return SymPyResult(
                success=False,
                error=raw["error"],
                execution_time_ms=raw.get("execution_time_ms", 0.0),
            )

        variables = raw["variables"]
        return SymPyResult(
            success=True,
            expression=variables.get("_expr_str"),
            numeric_value=variables.get("_numeric_value"),
            latex=variables.get("_latex_str"),
            steps=variables.get("_steps", []),
            execution_time_ms=raw.get("execution_time_ms", 0.0),
        )

    def verify_equality(self, expr_a: str, expr_b: str) -> bool:
        """Check whether two symbolic expressions are mathematically equal.

        Attempts multiple strategies: simplify(a - b) == 0, expand,
        trigsimp, and numerical evaluation at random points.
        """
        code = _EQUALITY_CHECK.format(expr_a=expr_a, expr_b=expr_b)
        raw = run_in_subprocess(
            code=code,
            namespace_setup=_SYMPY_SETUP,
            timeout=self.timeout,
        )
        if not raw["success"]:
            return False
        return bool(raw["variables"].get("_equal", False))

    def check_numeric(
        self,
        symbolic_result: str,
        expected: float,
        tolerance: float = 1e-9,
    ) -> bool:
        """Check whether a symbolic expression numerically matches *expected*."""
        code = (
            f"_expr = parse_expr({symbolic_result!r})\n"
            f"_val = float(_expr.evalf())\n"
            f"_match = abs(_val - {expected}) < {tolerance}\n"
        )
        raw = run_in_subprocess(
            code=code,
            namespace_setup=_SYMPY_SETUP,
            timeout=self.timeout,
        )
        if not raw["success"]:
            return False
        return bool(raw["variables"].get("_match", False))


# ---------------------------------------------------------------------------
# Code templates
# ---------------------------------------------------------------------------

_RESULT_WRAPPER = """\
# ---- user code ----
{user_code}
# ---- capture result ----
import sympy as _sympy

def _capture():
    _g = {{k: v for k, v in dict(globals()).items() if not k.startswith('_')}}
    # Prefer explicit 'result' variable
    if 'result' in _g:
        _target = _g['result']
    else:
        # Fall back to last assigned variable
        _candidates = [v for k, v in _g.items()
                       if not callable(v) and not isinstance(v, type)]
        _target = _candidates[-1] if _candidates else None

    if _target is None:
        return None, None, None

    _expr_str = str(_target)
    try:
        _latex_str = _sympy.latex(_target)
    except Exception:
        _latex_str = None
    try:
        _numeric_value = float(_sympy.N(_target))
    except Exception:
        _numeric_value = None
    return _expr_str, _latex_str, _numeric_value

_expr_str, _latex_str, _numeric_value = _capture()
_steps = _steps if '_steps' in dir() else []
"""

_EQUALITY_CHECK = """\
_a = parse_expr({expr_a!r})
_b = parse_expr({expr_b!r})

_equal = False

# Strategy 1: simplify(a - b) == 0
try:
    if simplify(_a - _b) == 0:
        _equal = True
except Exception:
    pass

# Strategy 2: expand
if not _equal:
    try:
        if expand(_a - _b) == 0:
            _equal = True
    except Exception:
        pass

# Strategy 3: trigsimp
if not _equal:
    try:
        if trigsimp(_a - _b) == 0:
            _equal = True
    except Exception:
        pass

# Strategy 4: numerical evaluation at random points
if not _equal:
    try:
        import random as _rng
        _syms = list((_a - _b).free_symbols)
        if not _syms:
            # No free symbols -- direct comparison
            _equal = abs(complex(N(_a - _b))) < 1e-10
        else:
            _all_close = True
            for _ in range(5):
                _subs = {{s: _rng.uniform(0.5, 2.5) for s in _syms}}
                _diff = complex((_a - _b).subs(_subs).evalf())
                if abs(_diff) > 1e-8:
                    _all_close = False
                    break
            _equal = _all_close
    except Exception:
        pass
"""
