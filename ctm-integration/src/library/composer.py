"""Rule composition: combining multiple rules for complex tasks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from src.library.rule import VerifiedRule
from src.synthesis.candidate_generator import IOExample
from src.synthesis.empirical_verifier import EmpiricalVerifier


@dataclass
class ComposedRule:
    """A rule composed from multiple verified rules."""

    composed_id: str
    component_rules: List[VerifiedRule]
    composition_type: str  # "sequential", "conditional", "parallel"
    description: str
    accuracy: float = 0.0

    @property
    def source_code(self) -> str:
        """Generate source code for the composed rule."""
        if self.composition_type == "sequential":
            return self._sequential_code()
        elif self.composition_type == "conditional":
            return self._conditional_code()
        else:
            return self._sequential_code()

    def _sequential_code(self) -> str:
        """Generate code for sequential composition (pipeline)."""
        lines = ["def composed_rule(x):"]
        lines.append("    result = x")
        for i, rule in enumerate(self.component_rules):
            func_name = f"_step_{i}"
            lines.append(f"    # Step {i}: {rule.description}")
            lines.append(f"    {func_name} = None")
            lines.append(f"    exec({rule.source_code!r}, globals(), locals())")
            # Extract function from rule
            lines.append(f"    for _n, _v in list(locals().items()):")
            lines.append(f"        if callable(_v) and not _n.startswith('_'):")
            lines.append(f"            {func_name} = _v")
            lines.append(f"            break")
            lines.append(f"    if {func_name} is not None:")
            lines.append(f"        result = {func_name}(result)")
        lines.append("    return result")
        return "\n".join(lines)

    def _conditional_code(self) -> str:
        """Generate code for conditional composition (dispatch)."""
        lines = ["def composed_rule(x):"]
        for i, rule in enumerate(self.component_rules):
            lines.append(f"    # Option {i}: {rule.description}")
            lines.append(f"    try:")
            lines.append(f"        ns = {{}}")
            lines.append(f"        exec({rule.source_code!r}, ns)")
            lines.append(f"        for _n, _v in ns.items():")
            lines.append(f"            if callable(_v) and not _n.startswith('_'):")
            lines.append(f"                result = _v(x)")
            lines.append(f"                if result is not None:")
            lines.append(f"                    return result")
            lines.append(f"    except Exception:")
            lines.append(f"        pass")
        lines.append("    return None")
        return "\n".join(lines)


class RuleComposer:
    """Composes multiple rules to solve complex tasks.

    Tries sequential composition (pipeline) and conditional composition
    (try rules in order until one works).
    """

    def __init__(self, verifier: Optional[EmpiricalVerifier] = None) -> None:
        self.verifier = verifier or EmpiricalVerifier()
        self._compose_counter = 0

    def compose(
        self,
        rules: List[VerifiedRule],
        examples: List[IOExample],
        max_rules: int = 3,
    ) -> Optional[ComposedRule]:
        """Try to compose rules to solve the given examples.

        Args:
            rules: Available verified rules.
            examples: I/O examples the composed rule should satisfy.
            max_rules: Maximum number of rules to compose.

        Returns:
            ComposedRule if a valid composition is found, None otherwise.
        """
        if not rules or not examples:
            return None

        # Try sequential composition
        seq = self._try_sequential(rules, examples, max_rules)
        if seq and seq.accuracy > 0.5:
            return seq

        # Try conditional composition
        cond = self._try_conditional(rules, examples, max_rules)
        if cond and cond.accuracy > 0.5:
            return cond

        # Return whichever is better
        if seq and cond:
            return seq if seq.accuracy >= cond.accuracy else cond
        return seq or cond

    def _try_sequential(
        self,
        rules: List[VerifiedRule],
        examples: List[IOExample],
        max_rules: int,
    ) -> Optional[ComposedRule]:
        """Try sequential composition of rules (pipeline).

        Tests pairs of rules as pipelines.
        """
        best: Optional[ComposedRule] = None
        best_accuracy = 0.0

        # Try individual rules first
        for rule in rules[:max_rules]:
            composed = self._make_composed([rule], "sequential", examples)
            if composed and composed.accuracy > best_accuracy:
                best = composed
                best_accuracy = composed.accuracy

        # Try pairs
        for i, r1 in enumerate(rules[:max_rules]):
            for j, r2 in enumerate(rules[:max_rules]):
                if i == j:
                    continue
                composed = self._make_composed([r1, r2], "sequential", examples)
                if composed and composed.accuracy > best_accuracy:
                    best = composed
                    best_accuracy = composed.accuracy

        return best

    def _try_conditional(
        self,
        rules: List[VerifiedRule],
        examples: List[IOExample],
        max_rules: int,
    ) -> Optional[ComposedRule]:
        """Try conditional composition (try rules in order).

        The first rule that produces a non-None result is used.
        """
        # Sort rules by accuracy (best first)
        sorted_rules = sorted(rules, key=lambda r: r.accuracy, reverse=True)
        subset = sorted_rules[:max_rules]

        composed = self._make_composed(subset, "conditional", examples)
        return composed

    def _make_composed(
        self,
        rules: List[VerifiedRule],
        composition_type: str,
        examples: List[IOExample],
    ) -> Optional[ComposedRule]:
        """Create and evaluate a composed rule."""
        self._compose_counter += 1
        composed = ComposedRule(
            composed_id=f"composed_{self._compose_counter:04d}",
            component_rules=rules,
            composition_type=composition_type,
            description=f"{composition_type} composition of {len(rules)} rules",
        )

        # Evaluate accuracy
        accuracy = self._evaluate_composed(composed, examples)
        composed.accuracy = accuracy

        return composed

    def _evaluate_composed(
        self, composed: ComposedRule, examples: List[IOExample]
    ) -> float:
        """Evaluate a composed rule against examples."""
        code = composed.source_code
        correct = 0

        try:
            namespace: dict = {}
            exec(code, namespace)

            func = None
            for name, obj in namespace.items():
                if callable(obj) and not name.startswith("_"):
                    func = obj
                    break

            if func is None:
                return 0.0

            for example in examples:
                try:
                    result = func(example.input)
                    if result == example.output:
                        correct += 1
                except Exception:
                    pass
        except Exception:
            return 0.0

        return correct / len(examples) if examples else 0.0
