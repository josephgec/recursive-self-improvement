"""Tests for the task router."""

from __future__ import annotations

from src.pipeline.router import RoutingDecision, TaskRouter, TaskType


class TestHeuristicRoute:
    """Test keyword + metadata-based routing."""

    def setup_method(self):
        self.router = TaskRouter()

    def test_algebra_keyword(self):
        decision = self.router.heuristic_route("Solve the equation x^2 - 4 = 0")
        assert decision.task_type == TaskType.ALGEBRA
        assert decision.use_symcode is True

    def test_geometry_keyword_prose(self):
        decision = self.router.heuristic_route(
            "Find the angle between two sides of a triangle"
        )
        assert decision.task_type == TaskType.GEOMETRY
        assert decision.use_symcode is False  # geometry defaults to prose

    def test_number_theory_keyword(self):
        decision = self.router.heuristic_route(
            "Find the remainder when 17^100 is divided modulo 5"
        )
        assert decision.task_type == TaskType.NUMBER_THEORY
        assert decision.use_symcode is True

    def test_calculus_keyword(self):
        decision = self.router.heuristic_route(
            "Find the derivative of sin(x)*x^2"
        )
        assert decision.task_type == TaskType.CALCULUS
        assert decision.use_symcode is True

    def test_probability_keyword(self):
        decision = self.router.heuristic_route(
            "What is the probability of rolling a 6 on a fair die?"
        )
        assert decision.task_type == TaskType.PROBABILITY
        assert decision.use_symcode is True

    def test_combinatorics_keyword(self):
        decision = self.router.heuristic_route(
            "How many ways can 5 people be arranged in a line?"
        )
        assert decision.task_type == TaskType.COMBINATORICS
        assert decision.use_symcode is True

    def test_general_fallback(self):
        decision = self.router.heuristic_route(
            "What is the meaning of life?"
        )
        assert decision.task_type == TaskType.GENERAL
        assert decision.use_symcode is True

    def test_metadata_subject_override(self):
        """MATH dataset subject field should override keywords."""
        decision = self.router.heuristic_route(
            "Find the area of the polygon",
            metadata={"subject": "geometry"},
        )
        assert decision.task_type == TaskType.GEOMETRY
        assert decision.use_symcode is False

    def test_metadata_task_type_override(self):
        """Explicit task_type in metadata takes priority."""
        decision = self.router.heuristic_route(
            "Solve the equation",
            metadata={"task_type": "number_theory"},
        )
        assert decision.task_type == TaskType.NUMBER_THEORY

    def test_metadata_use_symcode_override(self):
        """use_symcode in metadata overrides default for geometry."""
        decision = self.router.heuristic_route(
            "Find the angle of a triangle",
            metadata={"subject": "geometry", "use_symcode": True},
        )
        assert decision.task_type == TaskType.GEOMETRY
        assert decision.use_symcode is True

    def test_math_subject_mapping(self):
        """MATH dataset subjects map to correct TaskTypes."""
        cases = {
            "algebra": TaskType.ALGEBRA,
            "intermediate_algebra": TaskType.ALGEBRA,
            "prealgebra": TaskType.ALGEBRA,
            "geometry": TaskType.GEOMETRY,
            "number_theory": TaskType.NUMBER_THEORY,
            "counting_and_probability": TaskType.PROBABILITY,
            "precalculus": TaskType.CALCULUS,
        }
        for subject, expected_type in cases.items():
            decision = self.router.heuristic_route(
                "Some problem", metadata={"subject": subject}
            )
            assert decision.task_type == expected_type, (
                f"Subject '{subject}' mapped to {decision.task_type}, "
                f"expected {expected_type}"
            )

    def test_routing_decision_has_reasoning(self):
        decision = self.router.heuristic_route("Solve x + 1 = 2")
        assert decision.reasoning
        assert len(decision.reasoning) > 0

    def test_logic_keyword(self):
        decision = self.router.heuristic_route(
            "Prove by induction that the formula holds."
        )
        assert decision.task_type == TaskType.LOGIC
        assert decision.use_symcode is True


class TestLLMRoute:
    """Test LLM-based routing."""

    def setup_method(self):
        self.router = TaskRouter()

    def test_llm_route_no_client_fallback(self):
        """Without client, should fall back to heuristic."""
        decision = self.router.llm_route("Solve the equation x + 1 = 2")
        assert decision.task_type == TaskType.ALGEBRA
        assert decision.use_symcode is True

    def test_llm_route_with_mock_client(self):
        """Test LLM route with a mocked OpenAI client."""
        from unittest.mock import MagicMock
        import json

        mock_client = MagicMock()
        choice = MagicMock()
        choice.message.content = json.dumps({
            "task_type": "number_theory",
            "use_symcode": True,
            "reasoning": "This is a number theory problem.",
        })
        response = MagicMock()
        response.choices = [choice]
        mock_client.chat.completions.create.return_value = response

        decision = self.router.llm_route(
            "Find the remainder when 17^100 is divided by 5",
            llm_client=mock_client,
            router_prompt="Classify this math problem.",
        )
        assert decision.task_type == TaskType.NUMBER_THEORY
        assert decision.use_symcode is True
        assert decision.reasoning == "This is a number theory problem."

    def test_llm_route_invalid_json_fallback(self):
        """If LLM returns invalid JSON, should fall back to heuristic."""
        from unittest.mock import MagicMock

        mock_client = MagicMock()
        choice = MagicMock()
        choice.message.content = "this is not valid json"
        response = MagicMock()
        response.choices = [choice]
        mock_client.chat.completions.create.return_value = response

        decision = self.router.llm_route(
            "Find the derivative of x^2",
            llm_client=mock_client,
        )
        # Should fall back to heuristic
        assert decision.task_type == TaskType.CALCULUS

    def test_llm_route_api_error_fallback(self):
        """If LLM call raises exception, should fall back."""
        from unittest.mock import MagicMock

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception("API error")

        decision = self.router.llm_route(
            "Solve the equation x + 1 = 2",
            llm_client=mock_client,
        )
        assert decision.task_type == TaskType.ALGEBRA

    def test_llm_route_unknown_task_type(self):
        """If LLM returns unknown task type, should default to GENERAL."""
        from unittest.mock import MagicMock
        import json

        mock_client = MagicMock()
        choice = MagicMock()
        choice.message.content = json.dumps({
            "task_type": "quantum_physics",
            "use_symcode": False,
            "reasoning": "This is something else.",
        })
        response = MagicMock()
        response.choices = [choice]
        mock_client.chat.completions.create.return_value = response

        decision = self.router.llm_route(
            "Describe a quantum state",
            llm_client=mock_client,
        )
        assert decision.task_type == TaskType.GENERAL
        assert decision.use_symcode is False
