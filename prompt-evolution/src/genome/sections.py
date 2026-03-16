"""Section definitions and validation for prompt genomes."""

from typing import Dict, List, Optional

SECTION_ORDER: List[str] = [
    "identity",
    "task_description",
    "methodology",
    "reasoning_style",
    "output_format",
    "constraints",
    "examples",
    "error_handling",
]

DEFAULT_SECTIONS: Dict[str, str] = {
    "identity": "You are a helpful assistant specializing in financial mathematics.",
    "task_description": "Solve financial math problems step by step, showing all work.",
    "methodology": (
        "Use standard financial formulas. Break problems into steps. "
        "Verify calculations by checking units and reasonableness."
    ),
    "reasoning_style": (
        "Think step by step. State your approach before calculating. "
        "Show intermediate values. Double-check your final answer."
    ),
    "output_format": (
        "Present your answer clearly. Show the formula used, the substitution "
        "of values, and the final numeric result. Round to 2 decimal places "
        "unless otherwise specified."
    ),
    "constraints": (
        "Always use the exact values given in the problem. Do not assume "
        "additional information. If the problem is ambiguous, state your "
        "assumptions clearly."
    ),
    "examples": "",
    "error_handling": (
        "If a problem cannot be solved with the given information, explain "
        "what additional data would be needed."
    ),
}

SECTION_IMPORTANCE_WEIGHTS: Dict[str, float] = {
    "identity": 0.05,
    "task_description": 0.15,
    "methodology": 0.25,
    "reasoning_style": 0.20,
    "output_format": 0.10,
    "constraints": 0.10,
    "examples": 0.10,
    "error_handling": 0.05,
}


def section_importance_weights() -> Dict[str, float]:
    """Return importance weights for each section."""
    return dict(SECTION_IMPORTANCE_WEIGHTS)


def validate_sections(sections: Dict[str, "PromptSection"]) -> List[str]:
    """Validate that sections meet minimum requirements.

    Returns list of error messages (empty if valid).
    """
    errors = []
    required = ["identity", "task_description", "methodology"]
    for req in required:
        if req not in sections:
            errors.append(f"Missing required section: {req}")
        elif not sections[req].content.strip():
            errors.append(f"Required section '{req}' is empty")

    for name, section in sections.items():
        if section.token_count > 500:
            errors.append(
                f"Section '{name}' exceeds 500 token limit "
                f"({section.token_count} tokens)"
            )

    return errors
