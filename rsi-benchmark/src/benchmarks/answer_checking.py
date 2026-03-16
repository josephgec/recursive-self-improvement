"""Per-benchmark answer checkers: numeric tolerance, exact match, code execution, symbolic."""

from __future__ import annotations

from typing import Any, List, Tuple


class NumericChecker:
    """Check numeric answers with tolerance."""

    @staticmethod
    def check(predicted: Any, expected: Any, tolerance: float = 0.01) -> bool:
        try:
            pred_val = float(predicted)
            exp_val = float(expected)
        except (TypeError, ValueError):
            return False
        if exp_val == 0:
            return abs(pred_val) < tolerance
        return abs(pred_val - exp_val) / max(abs(exp_val), 1e-10) < tolerance


class ExactChecker:
    """Check exact match answers."""

    @staticmethod
    def check(predicted: Any, expected: Any) -> bool:
        return predicted == expected


class CodeChecker:
    """Check code answers by verifying the code string or test cases."""

    @staticmethod
    def check(predicted: Any, expected: Any, test_cases: List[Tuple] = None) -> bool:
        # First try exact string match (normalized)
        pred_str = str(predicted).strip()
        exp_str = str(expected).strip()
        if pred_str == exp_str:
            return True

        # If test cases provided, try to validate via execution
        if test_cases:
            return CodeChecker._run_tests(pred_str, test_cases)
        return False

    @staticmethod
    def _run_tests(code: str, test_cases: List[Tuple]) -> bool:
        """Execute code and run test cases. Safe for mock usage."""
        try:
            namespace: dict = {}
            exec(code, namespace)
            # Find the function in namespace
            func = None
            for val in namespace.values():
                if callable(val) and val.__class__.__name__ == "function":
                    func = val
                    break
            if func is None:
                return False

            for tc in test_cases:
                if len(tc) == 2:
                    inp, expected = tc
                    if not isinstance(inp, tuple):
                        inp = (inp,)
                    result = func(*inp)
                elif len(tc) == 3:
                    a, b, expected = tc
                    result = func(a, b)
                else:
                    continue
                if result != expected:
                    return False
            return True
        except Exception:
            return False


class SymbolicChecker:
    """Check symbolic math answers with normalization."""

    @staticmethod
    def check(predicted: Any, expected: Any) -> bool:
        pred_norm = SymbolicChecker._normalize(str(predicted))
        exp_norm = SymbolicChecker._normalize(str(expected))
        return pred_norm == exp_norm

    @staticmethod
    def _normalize(expr: str) -> str:
        """Basic symbolic normalization."""
        expr = expr.replace(" ", "").lower()
        expr = expr.replace("**", "^")
        expr = expr.replace("*", "")
        return expr
