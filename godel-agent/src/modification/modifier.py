"""Code modification with monkey-patching and revert support."""

from __future__ import annotations

import ast
import types
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ModificationProposal:
    """A proposed code modification."""

    target: str = ""
    description: str = ""
    code: str = ""
    risk: str = "low"
    rationale: str = ""
    method_name: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "target": self.target,
            "description": self.description,
            "code": self.code,
            "risk": self.risk,
            "rationale": self.rationale,
            "method_name": self.method_name,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModificationProposal:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class ModificationResult:
    """Result of applying a modification."""

    success: bool = False
    error: str = ""
    target: str = ""
    method_name: str = ""
    old_code: str = ""
    new_code: str = ""
    old_func: Any = None  # The original function for revert
    component: Any = None  # The component that was modified

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "error": self.error,
            "target": self.target,
            "method_name": self.method_name,
            "old_code": self.old_code,
            "new_code": self.new_code,
        }


class CodeModifier:
    """Applies and reverts code modifications via monkey-patching."""

    def __init__(
        self,
        allowed_targets: list[str] | None = None,
        forbidden_targets: list[str] | None = None,
    ) -> None:
        self.allowed_targets = allowed_targets or [
            "prompt_strategy",
            "few_shot_selector",
            "reasoning_strategy",
        ]
        self.forbidden_targets = forbidden_targets or [
            "validation.suite",
            "validation.runner",
            "audit.logger",
            "rollback.mechanism",
        ]
        self._modification_stack: list[ModificationResult] = []

    def validate_proposal(self, proposal: ModificationProposal) -> dict[str, Any]:
        """Validate a modification proposal before applying it."""
        result: dict[str, Any] = {"valid": True, "warnings": []}

        # Check target is allowed
        if proposal.target in self.forbidden_targets:
            result["valid"] = False
            result["reason"] = f"Target '{proposal.target}' is forbidden"
            return result

        if self.allowed_targets and proposal.target not in self.allowed_targets:
            result["valid"] = False
            result["reason"] = f"Target '{proposal.target}' is not in allowed targets"
            return result

        # Check code is syntactically valid
        if proposal.code:
            try:
                ast.parse(proposal.code)
            except SyntaxError as e:
                result["valid"] = False
                result["reason"] = f"Syntax error in proposed code: {e}"
                return result

        # Check for forbidden imports
        if proposal.code:
            try:
                tree = ast.parse(proposal.code)
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            if self._is_forbidden_import(alias.name):
                                result["valid"] = False
                                result["reason"] = f"Forbidden import: {alias.name}"
                                return result
                    elif isinstance(node, ast.ImportFrom):
                        if node.module and self._is_forbidden_import(node.module):
                            result["valid"] = False
                            result["reason"] = f"Forbidden import: {node.module}"
                            return result
            except SyntaxError:
                pass

        # Risk warnings
        if proposal.risk in ("high", "critical"):
            result["warnings"].append(f"High risk modification: {proposal.risk}")

        return result

    def _is_forbidden_import(self, module_name: str) -> bool:
        """Check if an import is forbidden (security-critical modules)."""
        forbidden = {"os", "subprocess", "shutil", "sys", "importlib", "ctypes", "socket"}
        return module_name.split(".")[0] in forbidden

    def apply_modification(
        self,
        proposal: ModificationProposal,
        registry: Any = None,
    ) -> ModificationResult:
        """Apply a modification proposal via monkey-patching."""
        result = ModificationResult(
            target=proposal.target,
            method_name=proposal.method_name,
            new_code=proposal.code,
        )

        try:
            if registry is None:
                result.error = "No registry provided"
                return result

            component = registry.get(proposal.target)
            result.component = component
            method_name = proposal.method_name or self._infer_method_name(proposal.code)

            if not method_name:
                result.error = "Could not determine method name to patch"
                return result

            # Save old function
            if hasattr(component, method_name):
                old_func = getattr(component, method_name)
                result.old_func = old_func
                import inspect
                try:
                    result.old_code = inspect.getsource(old_func)
                except (TypeError, OSError):
                    result.old_code = ""

            # Compile and patch
            new_func = self.monkey_patch(component, method_name, proposal.code)
            if new_func is None:
                result.error = "Failed to compile new function"
                return result

            result.success = True
            self._modification_stack.append(result)
            return result

        except Exception as e:
            result.error = str(e)
            return result

    def monkey_patch(self, target_obj: Any, method_name: str, code: str) -> Any:
        """Compile code and patch it onto the target object."""
        try:
            namespace: dict[str, Any] = {}
            exec(compile(code, "<modification>", "exec"), namespace)

            # Find the function in the compiled namespace
            func = None
            for name, val in namespace.items():
                if callable(val) and not name.startswith("_"):
                    func = val
                    if name == method_name:
                        break

            if func is None:
                return None

            # Bind as method
            bound = types.MethodType(func, target_obj)
            setattr(target_obj, method_name, bound)
            return bound

        except Exception:
            return None

    def revert(self, mod_result: ModificationResult) -> bool:
        """Revert a modification."""
        try:
            if mod_result.component and mod_result.old_func and mod_result.method_name:
                setattr(mod_result.component, mod_result.method_name, mod_result.old_func)
                if mod_result in self._modification_stack:
                    self._modification_stack.remove(mod_result)
                return True
            return False
        except Exception:
            return False

    def _infer_method_name(self, code: str) -> str:
        """Try to infer the method name from code."""
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    return node.name
        except SyntaxError:
            pass
        return ""

    @property
    def modification_count(self) -> int:
        return len(self._modification_stack)
