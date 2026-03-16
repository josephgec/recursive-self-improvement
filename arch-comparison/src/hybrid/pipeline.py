"""Hybrid pipeline: LLM + external solver via agentic tool-calling loop."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from src.hybrid.chain_logger import ChainLogger
from src.hybrid.result_integrator import IntegrationContext, ResultIntegrator
from src.hybrid.tool_dispatcher import ToolDispatcher, ToolResult


@dataclass
class ReasoningStep:
    """A single step in the hybrid reasoning chain."""
    step_type: str  # "reasoning", "tool_call", "tool_result", "conclusion"
    content: str
    tool_name: Optional[str] = None
    tool_input: Optional[str] = None
    tool_output: Optional[str] = None


@dataclass
class HybridResult:
    """Result from the hybrid pipeline."""
    answer: str
    correct: bool = False
    reasoning_chain: List[ReasoningStep] = field(default_factory=list)
    tool_calls_made: int = 0
    total_time: float = 0.0
    metadata: dict = field(default_factory=dict)


class HybridPipeline:
    """Agentic tool-calling pipeline: LLM reasons and invokes external solvers.

    The LLM (mock) decides when to call tools, receives results, and
    continues reasoning until it produces a final answer.
    """

    def __init__(
        self,
        llm: Optional[Callable] = None,
        max_tool_calls: int = 5,
        dispatcher: Optional[ToolDispatcher] = None,
    ) -> None:
        self.llm = llm or self._default_mock_llm
        self.max_tool_calls = max_tool_calls
        self.dispatcher = dispatcher or ToolDispatcher()
        self.integrator = ResultIntegrator()
        self.logger = ChainLogger()

    def solve(self, problem: str) -> HybridResult:
        """Solve a problem using the agentic tool-calling loop."""
        self.logger.clear()
        start = time.monotonic()
        reasoning_chain: List[ReasoningStep] = []
        tool_calls_made = 0

        # Step 1: Initial LLM reasoning
        llm_response = self.llm(problem)
        step = ReasoningStep(step_type="reasoning", content=llm_response)
        reasoning_chain.append(step)
        self.logger.log_step("reasoning", llm_response)

        # Step 2: Agentic loop — detect tool calls and execute them
        current_context = llm_response
        while tool_calls_made < self.max_tool_calls:
            tool_call = self._extract_tool_call(current_context)
            if tool_call is None:
                break

            tool_name, tool_input = tool_call
            tool_calls_made += 1

            # Execute tool
            result: ToolResult = self.dispatcher.dispatch(tool_name, tool_input)

            # Log tool call
            tool_step = ReasoningStep(
                step_type="tool_call",
                content=f"Calling {tool_name}",
                tool_name=tool_name,
                tool_input=tool_input,
                tool_output=result.output if result.success else result.error,
            )
            reasoning_chain.append(tool_step)
            self.logger.log_step(
                "tool_call",
                f"Calling {tool_name}",
                tool_name=tool_name,
                tool_input=tool_input,
                tool_output=result.output if result.success else result.error,
            )

            # Integrate result
            ctx = IntegrationContext(
                original_problem=problem,
                reasoning_so_far=current_context,
                step_number=len(reasoning_chain),
            )
            integrated = self.integrator.integrate(
                result.output if result.success else (result.error or ""),
                ctx,
            )

            result_step = ReasoningStep(
                step_type="tool_result",
                content=integrated,
                tool_name=tool_name,
                tool_output=result.output if result.success else result.error,
            )
            reasoning_chain.append(result_step)
            self.logger.log_step("tool_result", integrated)

            # Continue reasoning with integrated result
            followup_prompt = f"{problem}\n\nPrevious reasoning: {current_context}\n\nTool result: {integrated}"
            current_context = self.llm(followup_prompt)
            cont_step = ReasoningStep(step_type="reasoning", content=current_context)
            reasoning_chain.append(cont_step)
            self.logger.log_step("reasoning", current_context)

        # Step 3: Extract final answer
        answer = self._extract_answer(current_context)
        conclusion = ReasoningStep(step_type="conclusion", content=answer)
        reasoning_chain.append(conclusion)
        self.logger.log_step("conclusion", answer)

        elapsed = time.monotonic() - start
        return HybridResult(
            answer=answer,
            reasoning_chain=reasoning_chain,
            tool_calls_made=tool_calls_made,
            total_time=elapsed,
        )

    def _extract_tool_call(self, text: str) -> Optional[tuple]:
        """Extract a tool call from LLM output.

        Looks for patterns like: TOOL_CALL: tool_name(input)
        or [tool_name]: input
        """
        # Pattern 1: TOOL_CALL: name(input)
        match = re.search(r"TOOL_CALL:\s*(\w+)\(([^)]*)\)", text)
        if match:
            return match.group(1), match.group(2)

        # Pattern 2: [tool_name]: input
        match = re.search(r"\[(\w+)\]:\s*(.+?)(?:\n|$)", text)
        if match:
            name = match.group(1)
            if name in self.dispatcher.available_tools:
                return name, match.group(2).strip()

        return None

    def _extract_answer(self, text: str) -> str:
        """Extract the final answer from LLM output."""
        # Look for "ANSWER: ..." pattern
        match = re.search(r"ANSWER:\s*(.+?)(?:\n|$)", text)
        if match:
            return match.group(1).strip()

        # Look for "the answer is ..." pattern
        match = re.search(r"the answer is\s+(.+?)(?:\.|$)", text, re.IGNORECASE)
        if match:
            return match.group(1).strip()

        # Fallback: return last line
        lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
        return lines[-1] if lines else text.strip()

    @staticmethod
    def _default_mock_llm(prompt: str) -> str:
        """Default mock LLM: deterministic responses based on content."""
        prompt_lower = prompt.lower()

        # Arithmetic: detect "a + b" or "a * b" patterns
        arith = re.search(r"(\d+)\s*([\+\-\*\/])\s*(\d+)", prompt)
        if arith and ("what" in prompt_lower or "compute" in prompt_lower or "calculate" in prompt_lower):
            a, op, b = int(arith.group(1)), arith.group(2), int(arith.group(3))
            expr = f"{a} {op} {b}"
            return f"I need to compute {expr}.\nTOOL_CALL: sympy_solve({expr})\nANSWER: {expr}"

        # Equation solving
        if "solve" in prompt_lower:
            eq_match = re.search(r"(\w)\s*\+\s*(\d+)\s*=\s*(\d+)", prompt)
            if eq_match:
                expr = eq_match.group(0)
                return f"Let me solve this equation.\nTOOL_CALL: sympy_solve({expr})\nANSWER: solving"

        # Logic / satisfiability
        if "satisfi" in prompt_lower or "contradict" in prompt_lower or "logic" in prompt_lower:
            formula = re.search(r"formula[:\s]+(.+?)(?:\n|$)", prompt, re.IGNORECASE)
            f_text = formula.group(1) if formula else "formula"
            return f"Let me check satisfiability.\nTOOL_CALL: z3_check({f_text})\nANSWER: checking"

        # Tool result follow-up: extract the answer from tool result
        if "tool result" in prompt_lower:
            ans_match = re.search(r"the answer is\s+(.+?)[\.\n]", prompt, re.IGNORECASE)
            if ans_match:
                return f"Based on the solver, the answer is {ans_match.group(1).strip()}.\nANSWER: {ans_match.group(1).strip()}"
            result_match = re.search(r"(?:yields|gives|returned:?)\s+(.+?)[\.\n,]", prompt, re.IGNORECASE)
            if result_match:
                return f"ANSWER: {result_match.group(1).strip()}"

        # Default
        return f"I'll reason about this directly.\nANSWER: unknown"
