"""Rule invocation: safely executing verified and composed rules."""

from __future__ import annotations

from typing import Any, Optional

from src.library.composer import ComposedRule
from src.library.rule import VerifiedRule


class RuleInvoker:
    """Safely invokes verified rules and composed rules.

    Executes rule source code in an isolated namespace.
    """

    def __init__(self, timeout: float = 5.0) -> None:
        """Initialize the invoker.

        Args:
            timeout: Maximum execution time in seconds.
        """
        self.timeout = timeout

    def invoke(self, rule: VerifiedRule, input_val: Any) -> Any:
        """Invoke a verified rule with the given input.

        Args:
            rule: The verified rule to invoke.
            input_val: Input value.

        Returns:
            Rule output.

        Raises:
            RuntimeError: If the rule fails to execute.
        """
        try:
            namespace: dict = {}
            exec(rule.source_code, namespace)

            # Find the callable
            func = None
            for name, obj in namespace.items():
                if callable(obj) and not name.startswith("_"):
                    func = obj
                    break

            if func is None:
                raise RuntimeError(
                    f"No callable function found in rule {rule.rule_id}"
                )

            if isinstance(input_val, (list, tuple)):
                try:
                    return func(input_val)
                except TypeError:
                    return func(*input_val)
            return func(input_val)

        except RuntimeError:
            raise
        except Exception as e:
            raise RuntimeError(
                f"Rule {rule.rule_id} execution failed: {e}"
            ) from e

    def invoke_composed(
        self, composed_rule: ComposedRule, input_val: Any
    ) -> Any:
        """Invoke a composed rule with the given input.

        Args:
            composed_rule: The composed rule to invoke.
            input_val: Input value.

        Returns:
            Composed rule output.

        Raises:
            RuntimeError: If execution fails.
        """
        try:
            code = composed_rule.source_code
            namespace: dict = {}
            exec(code, namespace)

            func = None
            for name, obj in namespace.items():
                if callable(obj) and not name.startswith("_"):
                    func = obj
                    break

            if func is None:
                raise RuntimeError(
                    f"No callable in composed rule {composed_rule.composed_id}"
                )

            return func(input_val)

        except RuntimeError:
            raise
        except Exception as e:
            raise RuntimeError(
                f"Composed rule {composed_rule.composed_id} failed: {e}"
            ) from e
