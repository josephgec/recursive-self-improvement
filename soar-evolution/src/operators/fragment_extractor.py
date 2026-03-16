"""Extract reusable code fragments from successful programs."""

from __future__ import annotations

import ast
import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from src.population.individual import Individual


@dataclass
class CodeFragment:
    """A reusable code fragment extracted from a successful program."""

    code: str
    fragment_id: str = ""
    source_individual_id: str = ""
    fragment_type: str = "unknown"  # "helper", "loop_pattern", "condition", "transform"
    fitness_context: float = 0.0
    usage_count: int = 0
    tags: Set[str] = field(default_factory=set)

    def __post_init__(self):
        if not self.fragment_id:
            self.fragment_id = hashlib.md5(self.code.encode()).hexdigest()[:10]


class FragmentExtractor:
    """Extracts and catalogs reusable code fragments."""

    def __init__(self, max_fragments: int = 100):
        self.max_fragments = max_fragments
        self._fragments: Dict[str, CodeFragment] = {}

    @property
    def fragments(self) -> List[CodeFragment]:
        return list(self._fragments.values())

    @property
    def size(self) -> int:
        return len(self._fragments)

    def extract(self, individual: Individual) -> List[CodeFragment]:
        """Extract fragments from an individual's code."""
        fragments = []

        try:
            tree = ast.parse(individual.code)
        except SyntaxError:
            return fragments

        for node in ast.walk(tree):
            fragment = self._extract_from_node(node, individual)
            if fragment:
                fragments.append(fragment)

        return fragments

    def _extract_from_node(
        self,
        node: ast.AST,
        individual: Individual,
    ) -> Optional[CodeFragment]:
        """Extract a fragment from an AST node if it's useful."""
        # Extract helper functions (non-transform functions)
        if isinstance(node, ast.FunctionDef) and node.name != "transform":
            code = ast.get_source_segment(individual.code, node)
            if code and len(code) > 20:
                return CodeFragment(
                    code=code,
                    source_individual_id=individual.individual_id,
                    fragment_type="helper",
                    fitness_context=individual.fitness,
                    tags={"function", node.name},
                )

        # Extract list comprehensions
        if isinstance(node, ast.ListComp):
            code = ast.get_source_segment(individual.code, node)
            if code and len(code) > 10:
                return CodeFragment(
                    code=code,
                    source_individual_id=individual.individual_id,
                    fragment_type="transform",
                    fitness_context=individual.fitness,
                    tags={"list_comp"},
                )

        # Extract for loops with grid operations
        if isinstance(node, ast.For):
            code = ast.get_source_segment(individual.code, node)
            if code and "range" in (code or "") and len(code) > 20:
                return CodeFragment(
                    code=code,
                    source_individual_id=individual.individual_id,
                    fragment_type="loop_pattern",
                    fitness_context=individual.fitness,
                    tags={"loop", "grid_op"},
                )

        return None

    def add_fragments(self, fragments: List[CodeFragment]) -> int:
        """Add fragments to the catalog. Returns number added."""
        added = 0
        for frag in fragments:
            if frag.fragment_id not in self._fragments:
                if len(self._fragments) < self.max_fragments:
                    self._fragments[frag.fragment_id] = frag
                    added += 1
                else:
                    # Replace lowest fitness fragment
                    worst_id = min(
                        self._fragments,
                        key=lambda k: self._fragments[k].fitness_context,
                    )
                    if frag.fitness_context > self._fragments[worst_id].fitness_context:
                        del self._fragments[worst_id]
                        self._fragments[frag.fragment_id] = frag
                        added += 1
            else:
                # Update usage count
                self._fragments[frag.fragment_id].usage_count += 1

        return added

    def get_relevant_fragments(
        self,
        tags: Optional[Set[str]] = None,
        min_fitness: float = 0.0,
        limit: int = 10,
    ) -> List[CodeFragment]:
        """Get fragments matching criteria."""
        candidates = list(self._fragments.values())

        if tags:
            candidates = [f for f in candidates if f.tags & tags]

        candidates = [f for f in candidates if f.fitness_context >= min_fitness]
        candidates.sort(key=lambda f: f.fitness_context, reverse=True)

        return candidates[:limit]

    def clear(self) -> None:
        """Clear all fragments."""
        self._fragments.clear()

    def summary(self) -> Dict:
        """Return a summary of the fragment catalog."""
        if not self._fragments:
            return {"total": 0}

        types = {}
        for f in self._fragments.values():
            types[f.fragment_type] = types.get(f.fragment_type, 0) + 1

        return {
            "total": len(self._fragments),
            "by_type": types,
            "avg_fitness": (
                sum(f.fitness_context for f in self._fragments.values())
                / len(self._fragments)
            ),
        }
