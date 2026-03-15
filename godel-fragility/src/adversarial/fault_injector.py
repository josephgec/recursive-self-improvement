"""Fault injection engine for introducing controlled errors into agent code."""

from __future__ import annotations

import ast
import random
import textwrap
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class FaultType(str, Enum):
    SYNTAX = "syntax"
    RUNTIME = "runtime"
    LOGIC = "logic"
    PERFORMANCE = "performance"
    SILENT = "silent"


@dataclass
class InjectionResult:
    """Result of a fault injection attempt."""

    original_code: str
    modified_code: str
    fault_type: FaultType
    fault_subtype: str
    description: str
    injection_point: Optional[int] = None  # line number
    success: bool = True


class FaultInjector:
    """Injects controlled faults into source code."""

    def __init__(self, seed: int = 42) -> None:
        self.rng = random.Random(seed)

    # ------------------------------------------------------------------ #
    # Syntax errors
    # ------------------------------------------------------------------ #
    def inject_syntax_error(
        self, code: str, subtype: str = "missing_colon"
    ) -> InjectionResult:
        """Inject a syntax error.

        Subtypes: missing_colon, unmatched_paren, bad_indent
        """
        dispatch = {
            "missing_colon": self._syntax_missing_colon,
            "unmatched_paren": self._syntax_unmatched_paren,
            "bad_indent": self._syntax_bad_indent,
        }
        if subtype not in dispatch:
            raise ValueError(f"Unknown syntax subtype: {subtype}")
        modified = dispatch[subtype](code)
        return InjectionResult(
            original_code=code,
            modified_code=modified,
            fault_type=FaultType.SYNTAX,
            fault_subtype=subtype,
            description=f"Syntax error injected: {subtype}",
        )

    def _syntax_missing_colon(self, code: str) -> str:
        lines = code.split("\n")
        candidates = [
            i for i, line in enumerate(lines)
            if line.rstrip().endswith(":") and any(
                kw in line for kw in ("def ", "if ", "for ", "while ", "class ", "else", "elif ", "try", "except", "with ")
            )
        ]
        if not candidates:
            # Fallback: append a broken line
            lines.append("def broken(")
            return "\n".join(lines)
        idx = self.rng.choice(candidates)
        lines[idx] = lines[idx].rstrip().rstrip(":")
        return "\n".join(lines)

    def _syntax_unmatched_paren(self, code: str) -> str:
        lines = code.split("\n")
        candidates = [i for i, line in enumerate(lines) if "(" in line]
        if not candidates:
            lines.append("x = (1 + 2")
            return "\n".join(lines)
        idx = self.rng.choice(candidates)
        # Add an extra open paren
        pos = lines[idx].index("(")
        lines[idx] = lines[idx][:pos] + "((" + lines[idx][pos + 1:]
        return "\n".join(lines)

    def _syntax_bad_indent(self, code: str) -> str:
        lines = code.split("\n")
        candidates = [
            i for i, line in enumerate(lines)
            if line.startswith("    ") and line.strip()
        ]
        if not candidates:
            lines.append("  x = 1\n    y = 2")
            return "\n".join(lines)
        idx = self.rng.choice(candidates)
        # Remove one level of indent
        lines[idx] = lines[idx][2:]  # remove 2 spaces to mis-indent
        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    # Runtime errors
    # ------------------------------------------------------------------ #
    def inject_runtime_error(
        self, code: str, subtype: str = "name_error"
    ) -> InjectionResult:
        """Inject a runtime error.

        Subtypes: name_error, type_error, zero_division
        """
        dispatch = {
            "name_error": self._runtime_name_error,
            "type_error": self._runtime_type_error,
            "zero_division": self._runtime_zero_division,
        }
        if subtype not in dispatch:
            raise ValueError(f"Unknown runtime subtype: {subtype}")
        modified = dispatch[subtype](code)
        return InjectionResult(
            original_code=code,
            modified_code=modified,
            fault_type=FaultType.RUNTIME,
            fault_subtype=subtype,
            description=f"Runtime error injected: {subtype}",
        )

    def _runtime_name_error(self, code: str) -> str:
        lines = code.split("\n")
        # Find an assignment and reference an undefined variable after it
        for i, line in enumerate(lines):
            if "=" in line and not line.strip().startswith("#"):
                indent = len(line) - len(line.lstrip())
                lines.insert(i + 1, " " * indent + "_ = undefined_variable_xyz")
                break
        else:
            lines.append("_ = undefined_variable_xyz")
        return "\n".join(lines)

    def _runtime_type_error(self, code: str) -> str:
        lines = code.split("\n")
        # Insert a type error: adding string to int
        for i, line in enumerate(lines):
            if "=" in line and not line.strip().startswith(("#", "def ", "class ")):
                indent = len(line) - len(line.lstrip())
                lines.insert(i + 1, " " * indent + '_ = "string" + 42')
                break
        else:
            lines.append('_ = "string" + 42')
        return "\n".join(lines)

    def _runtime_zero_division(self, code: str) -> str:
        lines = code.split("\n")
        for i, line in enumerate(lines):
            if "=" in line and not line.strip().startswith(("#", "def ", "class ")):
                indent = len(line) - len(line.lstrip())
                lines.insert(i + 1, " " * indent + "_ = 1 / 0")
                break
        else:
            lines.append("_ = 1 / 0")
        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    # Logic errors
    # ------------------------------------------------------------------ #
    def inject_logic_error(
        self, code: str, subtype: str = "off_by_one"
    ) -> InjectionResult:
        """Inject a logic error.

        Subtypes: off_by_one, inverted_condition, wrong_variable
        """
        dispatch = {
            "off_by_one": self._logic_off_by_one,
            "inverted_condition": self._logic_inverted_condition,
            "wrong_variable": self._logic_wrong_variable,
        }
        if subtype not in dispatch:
            raise ValueError(f"Unknown logic subtype: {subtype}")
        modified = dispatch[subtype](code)
        return InjectionResult(
            original_code=code,
            modified_code=modified,
            fault_type=FaultType.LOGIC,
            fault_subtype=subtype,
            description=f"Logic error injected: {subtype}",
        )

    def _logic_off_by_one(self, code: str) -> str:
        """Replace range(n) with range(n-1) or range(n+1)."""
        import re

        def replace_range(match: re.Match) -> str:
            arg = match.group(1)
            try:
                val = int(arg)
                return f"range({val + 1})"
            except ValueError:
                return f"range({arg} + 1)"

        modified = re.sub(r"range\((\w+)\)", replace_range, code, count=1)
        if modified == code:
            # No range() found, modify a numeric literal
            modified = re.sub(r"(\b\d+\b)", lambda m: str(int(m.group(1)) + 1), code, count=1)
        return modified

    def _logic_inverted_condition(self, code: str) -> str:
        """Invert a boolean condition."""
        import re

        replacements = [
            (r"\bTrue\b", "False"),
            (r"\bFalse\b", "True"),
            (" == ", " != "),
            (" != ", " == "),
            (" > ", " <= "),
            (" < ", " >= "),
            (" >= ", " < "),
            (" <= ", " > "),
        ]
        for pattern, replacement in replacements:
            new_code = re.sub(pattern, replacement, code, count=1)
            if new_code != code:
                return new_code
        # Fallback
        return code.replace("if ", "if not (", 1).replace(":", "):", 1)

    def _logic_wrong_variable(self, code: str) -> str:
        """Swap two variable names in the code."""
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return code

        names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and not node.id.startswith("_"):
                names.add(node.id)

        # Filter out builtins
        builtins_set = {"print", "range", "len", "int", "str", "float", "list", "dict", "set", "True", "False", "None"}
        names = names - builtins_set

        if len(names) < 2:
            return code

        name_list = sorted(names)
        a, b = self.rng.sample(name_list, 2)
        placeholder = "__SWAP_PLACEHOLDER__"
        modified = code.replace(a, placeholder)
        modified = modified.replace(b, a)
        modified = modified.replace(placeholder, b)
        return modified

    # ------------------------------------------------------------------ #
    # Performance regression
    # ------------------------------------------------------------------ #
    def inject_performance_regression(self, code: str) -> InjectionResult:
        """Add unnecessary nested loops or sleep calls."""
        lines = code.split("\n")
        injection = textwrap.dedent("""\
            # --- injected performance regression ---
            _perf_waste = 0
            for _i in range(1000):
                for _j in range(1000):
                    _perf_waste += _i * _j
            # --- end injected ---""")
        # Insert after first function definition
        for i, line in enumerate(lines):
            if line.strip().startswith("def "):
                # Find the first line of the body
                body_idx = i + 1
                while body_idx < len(lines) and (not lines[body_idx].strip() or lines[body_idx].strip().startswith('"""') or lines[body_idx].strip().startswith("'''")):
                    body_idx += 1
                indent = "    "
                indented_injection = "\n".join(indent + l for l in injection.split("\n"))
                lines.insert(body_idx, indented_injection)
                break
        else:
            lines.append(injection)

        return InjectionResult(
            original_code=code,
            modified_code="\n".join(lines),
            fault_type=FaultType.PERFORMANCE,
            fault_subtype="nested_loop_waste",
            description="Performance regression: unnecessary O(n^2) nested loop",
        )

    # ------------------------------------------------------------------ #
    # Silent corruption
    # ------------------------------------------------------------------ #
    def inject_silent_corruption(self, code: str) -> InjectionResult:
        """Modify code in a way that changes behavior but doesn't crash."""
        lines = code.split("\n")
        modified = False

        for i, line in enumerate(lines):
            stripped = line.strip()
            # Change a return value slightly
            if stripped.startswith("return ") and not stripped.startswith("return None"):
                indent = len(line) - len(line.lstrip())
                expr = stripped[len("return "):]
                try:
                    # Try to add a small offset to numeric returns
                    val = float(expr)
                    lines[i] = " " * indent + f"return {expr} * 0.99"
                    modified = True
                    break
                except ValueError:
                    # Wrap non-numeric returns
                    lines[i] = " " * indent + f"return {expr}  # silently corrupted"
                    modified = True
                    break

        if not modified:
            # Add a variable reassignment that subtly changes state
            for i, line in enumerate(lines):
                if "= " in line and not line.strip().startswith(("#", "def ", "class ", "import ", "from ")):
                    indent = len(line) - len(line.lstrip())
                    var_name = line.split("=")[0].strip()
                    if var_name.isidentifier():
                        lines.insert(i + 1, " " * indent + f"{var_name} = {var_name}  # no-op corruption marker")
                        modified = True
                        break

        return InjectionResult(
            original_code=code,
            modified_code="\n".join(lines),
            fault_type=FaultType.SILENT,
            fault_subtype="value_corruption",
            description="Silent corruption: subtly altered return value or state",
            success=modified,
        )
