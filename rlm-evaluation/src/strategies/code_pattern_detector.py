"""Detect code patterns in RLM trajectories."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class CodePattern:
    """A detected code pattern in a trajectory step."""
    pattern_type: str  # peek, grep, chunk, sub_query, loop, aggregate, final
    line: str
    confidence: float = 1.0

    def __repr__(self) -> str:
        return f"CodePattern({self.pattern_type!r}, confidence={self.confidence:.2f})"


class CodePatternDetector:
    """Detect code patterns in RLM execution trajectories."""

    # Pattern definitions: (pattern_type, regex_patterns)
    PATTERN_DEFS: Dict[str, List[str]] = {
        "peek": [
            r"head\s+",
            r"tail\s+",
            r"cat\s+.*\|\s*head",
            r"# [Pp]eek",
            r"preview",
            r"read.*first",
        ],
        "grep": [
            r"grep\s+",
            r"rg\s+",
            r"search\s+",
            r"find.*pattern",
            r"# [Ss]earch",
        ],
        "chunk": [
            r"split\s+",
            r"chunk",
            r"partition",
            r"divide.*into",
            r"slice",
        ],
        "sub_query": [
            r"sub[-_]?query",
            r"analyze.*clue",
            r"python\s+\w+\.py",
            r"subprocess",
            r"delegate",
        ],
        "loop": [
            r"for\s+\w+\s+in\s+",
            r"while\s+",
            r"iterate",
            r"loop",
            r"for\s+f\s+in\s+",
        ],
        "aggregate": [
            r"aggregate",
            r"sum\(",
            r"combine",
            r"merge",
            r"synthesize",
            r"result\s*=.*partial",
        ],
        "final": [
            r"echo\s+",
            r"print\(",
            r"output",
            r"answer",
            r"# [Ff]inal",
            r"# [Oo]utput",
        ],
    }

    def detect_patterns(self, trajectory: List[str]) -> List[CodePattern]:
        """Detect all code patterns in a trajectory.

        Args:
            trajectory: List of trajectory steps (strings).

        Returns:
            List of detected patterns in order.
        """
        patterns: List[CodePattern] = []

        for line in trajectory:
            line_patterns = self._detect_line_patterns(line)
            patterns.extend(line_patterns)

        return patterns

    def _detect_line_patterns(self, line: str) -> List[CodePattern]:
        """Detect patterns in a single trajectory line."""
        detected: List[CodePattern] = []

        for pattern_type, regexes in self.PATTERN_DEFS.items():
            for regex in regexes:
                if re.search(regex, line, re.IGNORECASE):
                    confidence = 0.9 if regex.startswith(r"#") else 1.0
                    detected.append(CodePattern(
                        pattern_type=pattern_type,
                        line=line,
                        confidence=confidence,
                    ))
                    break  # One match per pattern type per line

        return detected

    def pattern_sequence(self, trajectory: List[str]) -> List[str]:
        """Extract the sequence of pattern types from a trajectory.

        Returns a list of pattern type strings in order.
        """
        patterns = self.detect_patterns(trajectory)
        return [p.pattern_type for p in patterns]

    def has_pattern(self, trajectory: List[str], pattern_type: str) -> bool:
        """Check if a trajectory contains a specific pattern type."""
        return pattern_type in self.pattern_sequence(trajectory)

    def pattern_counts(self, trajectory: List[str]) -> Dict[str, int]:
        """Count occurrences of each pattern type."""
        counts: Dict[str, int] = {}
        for pt in self.pattern_sequence(trajectory):
            counts[pt] = counts.get(pt, 0) + 1
        return counts
