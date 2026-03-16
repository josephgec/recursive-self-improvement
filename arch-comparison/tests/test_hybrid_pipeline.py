"""Tests for the hybrid pipeline: tool calling, chain logging, multi-step."""

from __future__ import annotations

import pytest

from src.hybrid.pipeline import HybridPipeline, HybridResult, ReasoningStep
from src.hybrid.tool_dispatcher import ToolDispatcher, ToolResult
from src.hybrid.result_integrator import ResultIntegrator, IntegrationContext
from src.hybrid.chain_logger import ChainLogger, LogEntry


# ── ToolDispatcher tests ──

class TestToolDispatcher:
    def test_dispatch_sympy_addition(self):
        d = ToolDispatcher()
        result = d.dispatch("sympy_solve", "3 + 4")
        assert result.success
        assert result.output == "7"

    def test_dispatch_sympy_multiplication(self):
        d = ToolDispatcher()
        result = d.dispatch("sympy_solve", "5 * 6")
        assert result.success
        assert result.output == "30"

    def test_dispatch_sympy_subtraction(self):
        d = ToolDispatcher()
        result = d.dispatch("sympy_solve", "10 - 3")
        assert result.success
        assert result.output == "7"

    def test_dispatch_sympy_equation(self):
        d = ToolDispatcher()
        result = d.dispatch("sympy_solve", "x + 3 = 7")
        assert result.success
        assert "4" in result.output

    def test_dispatch_sympy_mul_equation(self):
        d = ToolDispatcher()
        result = d.dispatch("sympy_solve", "2*x = 10")
        assert result.success
        assert "5" in result.output

    def test_dispatch_z3_sat(self):
        d = ToolDispatcher()
        result = d.dispatch("z3_check", "P or not P is true")
        assert result.success
        assert result.output == "sat"

    def test_dispatch_z3_unsat(self):
        d = ToolDispatcher()
        result = d.dispatch("z3_check", "false and contradiction")
        assert result.success
        assert result.output == "unsat"

    def test_dispatch_z3_range_contradiction(self):
        d = ToolDispatcher()
        result = d.dispatch("z3_check", "x > 5 and x < 3")
        assert result.success
        assert result.output == "unsat"

    def test_dispatch_simplify(self):
        d = ToolDispatcher()
        result = d.dispatch("simplify", "3 + 4")
        assert result.success
        assert result.output == "7"

    def test_dispatch_simplify_non_arithmetic(self):
        d = ToolDispatcher()
        result = d.dispatch("simplify", "x + y")
        assert result.success
        assert "simplified" in result.output

    def test_dispatch_factor(self):
        d = ToolDispatcher()
        result = d.dispatch("factor", "x^2 - 1")
        assert result.success
        assert "factored" in result.output

    def test_dispatch_expand(self):
        d = ToolDispatcher()
        result = d.dispatch("expand", "(x+1)^2")
        assert result.success
        assert "expanded" in result.output

    def test_dispatch_unknown_tool(self):
        d = ToolDispatcher()
        result = d.dispatch("nonexistent_tool", "input")
        assert not result.success
        assert "Unknown tool" in result.error

    def test_get_tool_definitions(self):
        d = ToolDispatcher()
        defs = d.get_tool_definitions()
        assert len(defs) >= 5
        names = [d["name"] for d in defs]
        assert "sympy_solve" in names
        assert "z3_check" in names

    def test_available_tools(self):
        d = ToolDispatcher()
        tools = d.available_tools
        assert "sympy_solve" in tools
        assert "z3_check" in tools

    def test_execution_time_recorded(self):
        d = ToolDispatcher()
        result = d.dispatch("sympy_solve", "3 + 4")
        assert result.execution_time >= 0

    def test_sympy_solve_fallback(self):
        d = ToolDispatcher()
        result = d.dispatch("sympy_solve", "some complex thing")
        assert result.success
        assert "solved" in result.output


# ── ResultIntegrator tests ──

class TestResultIntegrator:
    def test_integrate_numeric(self):
        ri = ResultIntegrator()
        result = ri.integrate("42")
        assert "42" in result
        assert "answer" in result.lower()

    def test_integrate_float(self):
        ri = ResultIntegrator()
        result = ri.integrate("3.14")
        assert "3.14" in result

    def test_integrate_sat(self):
        ri = ResultIntegrator()
        result = ri.integrate("sat")
        assert "satisfiable" in result.lower()

    def test_integrate_unsat(self):
        ri = ResultIntegrator()
        result = ri.integrate("unsat")
        assert "contradiction" in result.lower() or "unsatisfiable" in result.lower()

    def test_integrate_equation(self):
        ri = ResultIntegrator()
        result = ri.integrate("x = 4")
        assert "4" in result

    def test_integrate_empty(self):
        ri = ResultIntegrator()
        result = ri.integrate("")
        assert "no output" in result.lower()

    def test_integrate_generic(self):
        ri = ResultIntegrator()
        result = ri.integrate("some arbitrary output")
        assert "some arbitrary output" in result

    def test_integrate_with_context(self):
        ri = ResultIntegrator()
        ctx = IntegrationContext(original_problem="What is x?", step_number=2)
        result = ri.integrate("42", ctx)
        assert "42" in result

    def test_integrate_integer_float(self):
        ri = ResultIntegrator()
        result = ri.integrate("7.0")
        # Should display as integer since 7.0 == 7
        assert "7" in result


# ── ChainLogger tests ──

class TestChainLogger:
    def test_log_step(self):
        cl = ChainLogger()
        entry = cl.log_step("reasoning", "Thinking about the problem")
        assert entry.step_number == 1
        assert entry.step_type == "reasoning"
        assert len(cl) == 1

    def test_log_multiple_steps(self):
        cl = ChainLogger()
        cl.log_step("reasoning", "Step 1")
        cl.log_step("tool_call", "Calling tool", tool_name="sympy_solve", tool_input="3+4")
        cl.log_step("tool_result", "Got 7", tool_output="7")
        assert len(cl) == 3

    def test_get_chain(self):
        cl = ChainLogger()
        cl.log_step("reasoning", "Step 1")
        cl.log_step("reasoning", "Step 2")
        chain = cl.get_chain()
        assert len(chain) == 2
        assert chain[0].step_number == 1
        assert chain[1].step_number == 2

    def test_clear(self):
        cl = ChainLogger()
        cl.log_step("reasoning", "Step 1")
        cl.clear()
        assert len(cl) == 0
        cl.log_step("reasoning", "After clear")
        assert cl.get_chain()[0].step_number == 1

    def test_format_for_display_empty(self):
        cl = ChainLogger()
        assert cl.format_for_display() == "(empty chain)"

    def test_format_for_display(self):
        cl = ChainLogger()
        cl.log_step("reasoning", "Think")
        cl.log_step("tool_call", "Call solver", tool_name="sympy_solve",
                     tool_input="3+4", tool_output="7")
        display = cl.format_for_display()
        assert "Step 1" in display
        assert "Step 2" in display
        assert "sympy_solve" in display

    def test_log_step_with_metadata(self):
        cl = ChainLogger()
        entry = cl.log_step("reasoning", "Step", metadata={"key": "value"})
        assert entry.metadata == {"key": "value"}


# ── HybridPipeline tests ──

class TestHybridPipeline:
    def test_solve_basic_arithmetic(self, mock_llm):
        pipeline = HybridPipeline(llm=mock_llm, max_tool_calls=3)
        result = pipeline.solve("What is 3 + 4?")
        assert isinstance(result, HybridResult)
        assert result.answer != ""
        assert len(result.reasoning_chain) > 0
        assert result.total_time >= 0

    def test_solve_produces_reasoning_chain(self, mock_llm):
        pipeline = HybridPipeline(llm=mock_llm, max_tool_calls=3)
        result = pipeline.solve("What is 5 * 6?")
        assert len(result.reasoning_chain) >= 2  # at least reasoning + conclusion
        types = [s.step_type for s in result.reasoning_chain]
        assert "reasoning" in types
        assert "conclusion" in types

    def test_solve_with_tool_calls(self):
        """Test that the pipeline makes tool calls when the LLM requests them."""
        def tool_llm(prompt):
            if "tool result" in prompt.lower():
                return "ANSWER: 7"
            return "TOOL_CALL: sympy_solve(3 + 4)"

        pipeline = HybridPipeline(llm=tool_llm, max_tool_calls=3)
        result = pipeline.solve("What is 3 + 4?")
        assert result.tool_calls_made >= 1
        chain_types = [s.step_type for s in result.reasoning_chain]
        assert "tool_call" in chain_types

    def test_max_tool_calls_respected(self):
        """Test that max_tool_calls limits the number of tool calls."""
        call_count = 0

        def always_calls_tool(prompt):
            nonlocal call_count
            call_count += 1
            return "TOOL_CALL: sympy_solve(1 + 1)"

        pipeline = HybridPipeline(llm=always_calls_tool, max_tool_calls=2)
        result = pipeline.solve("Loop test")
        assert result.tool_calls_made <= 2

    def test_solve_no_tool_call(self):
        """Test pipeline when LLM doesn't request any tools."""
        def no_tool_llm(prompt):
            return "The answer is 42.\nANSWER: 42"

        pipeline = HybridPipeline(llm=no_tool_llm, max_tool_calls=5)
        result = pipeline.solve("What is 42?")
        assert result.tool_calls_made == 0
        assert result.answer == "42"

    def test_chain_logger_populated(self, mock_llm):
        pipeline = HybridPipeline(llm=mock_llm, max_tool_calls=3)
        pipeline.solve("What is 2 + 3?")
        chain = pipeline.logger.get_chain()
        assert len(chain) > 0

    def test_extract_answer_with_answer_tag(self):
        pipeline = HybridPipeline()
        answer = pipeline._extract_answer("Some reasoning\nANSWER: 42")
        assert answer == "42"

    def test_extract_answer_with_prose(self):
        pipeline = HybridPipeline()
        answer = pipeline._extract_answer("therefore the answer is 42.")
        assert answer == "42"

    def test_extract_answer_fallback(self):
        pipeline = HybridPipeline()
        answer = pipeline._extract_answer("just some text\nfinal line")
        assert answer == "final line"

    def test_extract_tool_call_pattern1(self):
        pipeline = HybridPipeline()
        tc = pipeline._extract_tool_call("TOOL_CALL: sympy_solve(3 + 4)")
        assert tc is not None
        assert tc[0] == "sympy_solve"
        assert tc[1] == "3 + 4"

    def test_extract_tool_call_pattern2(self):
        pipeline = HybridPipeline()
        tc = pipeline._extract_tool_call("[sympy_solve]: 3 + 4")
        assert tc is not None
        assert tc[0] == "sympy_solve"

    def test_extract_tool_call_none(self):
        pipeline = HybridPipeline()
        tc = pipeline._extract_tool_call("No tool call here")
        assert tc is None

    def test_default_mock_llm_arithmetic(self):
        result = HybridPipeline._default_mock_llm("What is 5 + 3?")
        assert "TOOL_CALL" in result or "ANSWER" in result

    def test_default_mock_llm_logic(self):
        result = HybridPipeline._default_mock_llm("Is this satisfiable?")
        assert "TOOL_CALL" in result or "ANSWER" in result

    def test_default_mock_llm_default(self):
        result = HybridPipeline._default_mock_llm("something random")
        assert "ANSWER" in result

    def test_default_mock_llm_solve(self):
        result = HybridPipeline._default_mock_llm("Solve for x: x + 3 = 7")
        assert "TOOL_CALL" in result

    def test_default_mock_llm_tool_result_followup(self):
        result = HybridPipeline._default_mock_llm(
            "Problem\n\nTool result: The computation yields 7, therefore the answer is 7."
        )
        assert "7" in result
