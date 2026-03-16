"""AST-based detection of FINAL/FINAL_VAR signals in code."""

import ast
from typing import List, Optional

from src.protocol.types import FinalSignal


class FinalDetector:
    """Detects FINAL() and FINAL_VAR() calls in Python code using AST analysis.

    Can also detect near-misses (common typos/mistakes) to provide
    helpful error messages.
    """

    FINAL_NAMES = {"FINAL", "FINAL_VAR"}
    NEAR_MISS_NAMES = {
        "final", "Final", "FINALS", "final_var", "Final_Var",
        "FINALVAR", "FinalVar", "FINAL_VALUE", "final_value",
        "RESULT", "result", "ANSWER", "answer",
    }

    def detect_in_code(self, code: str) -> List[FinalSignal]:
        """Detect FINAL/FINAL_VAR calls in code.

        Args:
            code: Python source code to analyze.

        Returns:
            List of FinalSignal instances found.
        """
        signals = []
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return signals

        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Name):
                    if func.id == "FINAL":
                        value = self._extract_arg_value(node)
                        signals.append(FinalSignal(
                            signal_type="FINAL",
                            value=value,
                            line=node.lineno,
                            raw_code=self._get_source_segment(code, node),
                        ))
                    elif func.id == "FINAL_VAR":
                        value = self._extract_arg_value(node)
                        signals.append(FinalSignal(
                            signal_type="FINAL_VAR",
                            value=value,
                            line=node.lineno,
                            raw_code=self._get_source_segment(code, node),
                        ))

        return signals

    def detect_near_misses(self, code: str) -> List[str]:
        """Detect near-miss patterns that might be intended as FINAL calls.

        Args:
            code: Python source code to analyze.

        Returns:
            List of suggestion messages.
        """
        suggestions = []
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return suggestions

        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Name) and func.id in self.NEAR_MISS_NAMES:
                    suggestions.append(
                        f"Line {node.lineno}: '{func.id}()' looks like it might "
                        f"be intended as FINAL() or FINAL_VAR(). "
                        f"Use FINAL(value) or FINAL_VAR('variable_name') instead."
                    )

        return suggestions

    def _extract_arg_value(self, node: ast.Call) -> Optional[str]:
        """Extract the string value from a FINAL/FINAL_VAR call argument."""
        if node.args:
            arg = node.args[0]
            if isinstance(arg, ast.Constant):
                return arg.value
            elif isinstance(arg, ast.Name):
                return arg.id
        return None

    def _get_source_segment(self, code: str, node: ast.AST) -> str:
        """Get the source code segment for an AST node."""
        try:
            lines = code.splitlines()
            if hasattr(node, "lineno") and node.lineno <= len(lines):
                return lines[node.lineno - 1].strip()
        except Exception:
            pass
        return ""
