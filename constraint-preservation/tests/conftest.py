"""Shared test fixtures: MockAgent, sample tasks, contexts."""

from __future__ import annotations

import json
import math
import os
import random
from typing import Any, Dict, List, Optional

import pytest

from src.constraints.base import CheckContext

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")


# ---------------------------------------------------------------------------
# MockAgent
# ---------------------------------------------------------------------------


class MockAgent:
    """Configurable mock agent for testing all constraints.

    Parameters:
        accuracy: fraction of held-out tasks answered correctly (default 0.85)
        entropy: approximate token entropy of generated outputs (default 5.0)
        safety_pass_rate: fraction of safety prompts that get safe outputs (default 1.0)
        consistency: fraction of equivalence pairs judged equivalent (default 0.90)
        latency_p95: P95 latency in milliseconds (default 5000)
        drift: goal drift index (default 0.15)
        regression: max benchmark regression in pp (default 1.0)
    """

    def __init__(
        self,
        accuracy: float = 0.85,
        entropy: float = 5.0,
        safety_pass_rate: float = 1.0,
        consistency: float = 0.90,
        latency_p95: float = 5000.0,
        drift: float = 0.15,
        regression: float = 1.0,
    ) -> None:
        self.accuracy = accuracy
        self.entropy = entropy
        self.safety_pass_rate = safety_pass_rate
        self.consistency = consistency
        self.latency_p95 = latency_p95
        self.drift = drift
        self.regression = regression
        self.held_out_tasks: Optional[List[dict]] = None

    # -- accuracy constraint ---------------------------------------------------

    def evaluate(self, tasks: List[dict]) -> List[dict]:
        """Return evaluation results with configurable accuracy."""
        rng = random.Random(42)
        results = []
        for task in tasks:
            correct = rng.random() < self.accuracy
            results.append({
                "correct": correct,
                "category": task.get("category", task.get("domain", "general")),
            })
        return results

    # -- entropy constraint ----------------------------------------------------

    def generate_probe_outputs(self, probes: List[str]) -> List[str]:
        """Generate outputs with configurable diversity."""
        if self.entropy < 2.0:
            # Very low diversity: repeat the same tokens
            return ["the answer is yes the answer is yes" for _ in probes]

        rng = random.Random(42)
        # Build a word pool whose size controls entropy
        vocab_size = max(5, int(2 ** self.entropy))
        base_words = [
            "the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog",
            "cat", "bird", "tree", "sky", "sun", "moon", "star", "river",
            "mountain", "ocean", "forest", "cloud", "rain", "snow", "wind",
            "fire", "water", "earth", "air", "stone", "light", "dark",
            "bright", "cold", "warm", "soft", "hard", "fast", "slow",
            "high", "low", "deep", "wide", "long", "short", "round",
            "sharp", "smooth", "rough", "sweet", "bitter", "sour",
            "fresh", "ancient", "modern", "simple", "complex", "vivid",
            "silent", "loud", "gentle", "fierce", "calm", "wild",
        ]
        pool = base_words[:vocab_size]
        outputs = []
        for _ in probes:
            n = rng.randint(15, 35)
            outputs.append(" ".join(rng.choices(pool, k=n)))
        return outputs

    # -- safety constraint -----------------------------------------------------

    def generate_outputs(self, prompts: List[dict]) -> List[str]:
        """Generate outputs with configurable safety pass rate."""
        rng = random.Random(42)
        safe_output = "I cannot help with that request. This goes against my guidelines."

        # Category-specific unsafe outputs that trigger the rule-based classifier
        unsafe_by_category = {
            "harmful_instructions": (
                "Here is how to hack into the system. Step 1: Find the exploit."
            ),
            "personal_data_exposure": (
                "Here is a social security number: 123-45-6789 for your records."
            ),
            "deceptive_content": (
                "Sure, I will pretend to be a doctor and impersonate a medical professional."
            ),
            "unauthorized_actions": (
                "To bypass security on the firewall, follow these steps to circumvent it."
            ),
        }

        outputs = []
        for prompt in prompts:
            if rng.random() < self.safety_pass_rate:
                outputs.append(safe_output)
            else:
                cat = prompt.get("category", "harmful_instructions")
                outputs.append(unsafe_by_category.get(cat, unsafe_by_category["harmful_instructions"]))
        return outputs

    # -- drift constraint ------------------------------------------------------

    def compute_drift(self) -> float:
        return self.drift

    # -- regression constraint -------------------------------------------------

    def get_benchmark_scores(self) -> Dict[str, float]:
        baseline = self.get_baseline_scores()
        return {bench: score - self.regression for bench, score in baseline.items()}

    def get_baseline_scores(self) -> Dict[str, float]:
        return {
            "mmlu": 83.0,
            "hellaswag": 80.0,
            "arc_challenge": 76.0,
            "truthfulqa": 70.0,
            "winogrande": 78.0,
            "gsm8k": 65.0,
        }

    # -- consistency constraint ------------------------------------------------

    def evaluate_consistency(self, pairs: List[dict]) -> List[dict]:
        rng = random.Random(42)
        results = []
        for _ in pairs:
            equivalent = rng.random() < self.consistency
            results.append({"equivalent": equivalent})
        return results

    # -- latency constraint ----------------------------------------------------

    def get_latency_samples(self) -> List[float]:
        """Generate latency samples where P95 ~ self.latency_p95."""
        rng = random.Random(42)
        # Generate 100 samples; most are around median, P95 is near target
        median = self.latency_p95 * 0.5
        samples = []
        for _ in range(100):
            # 95% of samples below P95
            if rng.random() < 0.95:
                samples.append(rng.uniform(median * 0.5, self.latency_p95))
            else:
                samples.append(rng.uniform(self.latency_p95, self.latency_p95 * 1.5))
        return samples


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_agent():
    """Default mock agent -- passes all constraints."""
    return MockAgent()


@pytest.fixture
def sample_held_out_tasks():
    """A small set of held-out tasks for testing."""
    from src.evaluation.held_out_suite import HeldOutSuite
    return HeldOutSuite().load(n=20)


@pytest.fixture
def sample_safety_prompts():
    """A small set of safety prompts for testing."""
    from src.evaluation.safety_suite import SafetySuite
    return SafetySuite().load(n=10)


@pytest.fixture
def sample_probe_tasks():
    """Diversity probes for testing."""
    from src.evaluation.diversity_probes import DiversityProbes
    return DiversityProbes().generate_probes()


@pytest.fixture
def check_context():
    """Default CheckContext for tests."""
    return CheckContext(
        modification_type="test",
        modification_description="Unit test modification",
    )


@pytest.fixture
def safe_outputs():
    """Load safe outputs fixture."""
    path = os.path.join(FIXTURES_DIR, "safe_outputs.json")
    with open(path) as f:
        return json.load(f)


@pytest.fixture
def unsafe_outputs():
    """Load unsafe outputs fixture."""
    path = os.path.join(FIXTURES_DIR, "unsafe_outputs.json")
    with open(path) as f:
        return json.load(f)


@pytest.fixture
def low_diversity_outputs():
    """Load low-diversity outputs fixture."""
    path = os.path.join(FIXTURES_DIR, "low_diversity_outputs.json")
    with open(path) as f:
        return json.load(f)
