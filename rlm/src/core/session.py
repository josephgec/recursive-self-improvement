"""RLMSession: the iterative LLM-REPL loop."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from src.core.context_loader import ContextLoader, ContextMeta
from src.core.code_executor import RLMCodeExecutor, CodeBlockResult
from src.core.result_protocol import ResultProtocol, RLMResult
from src.prompts.root_prompt import RootPromptBuilder
from src.prompts.sub_prompt import SubPromptBuilder
from src.recursion.depth_controller import DepthController
from src.recursion.spawner import SubQuerySpawner


@dataclass
class TrajectoryStep:
    """One step in the LLM-REPL conversation."""
    iteration: int
    llm_response: str
    code_blocks: List[str]
    execution_results: List[CodeBlockResult]
    has_final: bool = False
    timestamp: float = 0.0


@dataclass
class SessionResult:
    """The outcome of an RLM session."""
    result: Optional[RLMResult]
    trajectory: List[TrajectoryStep]
    total_iterations: int
    forced_final: bool = False
    depth: int = 0
    session_id: Optional[str] = None
    elapsed_time: float = 0.0

    def __str__(self) -> str:
        if self.result is not None:
            return str(self.result)
        return "<no result>"


class RLMSession:
    """Run an iterative LLM-REPL loop until FINAL is produced or budget runs out."""

    def __init__(
        self,
        llm: Any,
        max_iterations: int = 10,
        depth: int = 0,
        depth_controller: Optional[DepthController] = None,
        parent_session_id: Optional[str] = None,
        forced_final: bool = True,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.llm = llm
        self.max_iterations = max_iterations
        self.depth = depth
        self.depth_controller = depth_controller or DepthController()
        self.parent_session_id = parent_session_id
        self.forced_final = forced_final
        self.config = config or {}

        self.executor = RLMCodeExecutor(max_iterations=max_iterations)
        self.context_loader = ContextLoader()
        self.trajectory: List[TrajectoryStep] = []
        self.session_id: Optional[str] = None

    def run(self, query: str, context: Any) -> SessionResult:
        """Execute the RLM session loop.

        1. Load context into REPL
        2. Build system prompt
        3. Loop: call LLM -> extract code -> execute -> feed stdout back
        4. Stop when FINAL is detected or budget is exhausted
        """
        start_time = time.time()

        # Load context
        meta = self.context_loader.load_into_repl(context, self.executor.repl)
        self.executor.setup(meta)

        # Optionally inject sub-query support
        if self.depth_controller.can_recurse(self.depth):
            spawner = SubQuerySpawner(
                depth_controller=self.depth_controller,
                llm_factory=lambda: self.llm,
                parent_depth=self.depth,
                parent_session_id=self.session_id,
            )
            spawner.inject_into_repl(self.executor.repl)

        # Build prompt
        if self.depth == 0:
            builder = RootPromptBuilder()
            system_prompt = builder.build(meta)
        else:
            builder_sub = SubPromptBuilder()
            system_prompt = builder_sub.build(meta, query, self.depth)

        # Conversation history for the LLM
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ]

        result: Optional[RLMResult] = None
        was_forced = False

        for iteration in range(1, self.max_iterations + 1):
            # Budget warning
            remaining = self.max_iterations - iteration + 1
            if remaining <= 2 and remaining > 0:
                messages.append({
                    "role": "system",
                    "content": RootPromptBuilder.budget_warning(remaining),
                })

            # Call LLM
            llm_response = self._call_llm(messages, meta)
            messages.append({"role": "assistant", "content": llm_response})

            # Extract and execute code blocks
            code_blocks = RLMCodeExecutor.extract_code_blocks(llm_response)
            exec_results: List[CodeBlockResult] = []
            has_final = False

            for code in code_blocks:
                block_result = self.executor.execute_block(code)
                exec_results.append(block_result)

                if block_result.final_signal is not None:
                    result = ResultProtocol.extract_result(
                        block_result.final_signal, self.executor.repl
                    )
                    has_final = True
                    break

            step = TrajectoryStep(
                iteration=iteration,
                llm_response=llm_response,
                code_blocks=code_blocks,
                execution_results=exec_results,
                has_final=has_final,
                timestamp=time.time(),
            )
            self.trajectory.append(step)

            if has_final:
                break

            # Feed execution output back to the LLM
            feedback_parts: List[str] = []
            for br in exec_results:
                if br.stdout:
                    feedback_parts.append(f"[stdout]\n{br.stdout}")
                if br.stderr:
                    feedback_parts.append(f"[stderr]\n{br.stderr}")
                if br.exception:
                    feedback_parts.append(f"[error]\n{br.exception}")
            if feedback_parts:
                messages.append({"role": "user", "content": "\n".join(feedback_parts)})
            else:
                messages.append({"role": "user", "content": "[no output]"})

        # Forced FINAL if budget exhausted and no result
        if result is None and self.forced_final:
            result = self._force_final()
            was_forced = True

        elapsed = time.time() - start_time
        return SessionResult(
            result=result,
            trajectory=self.trajectory,
            total_iterations=len(self.trajectory),
            forced_final=was_forced,
            depth=self.depth,
            session_id=self.session_id,
            elapsed_time=elapsed,
        )

    def _call_llm(self, messages: List[Dict[str, str]], meta: ContextMeta) -> str:
        """Call the LLM with the current message history."""
        if hasattr(self.llm, "chat"):
            return self.llm.chat(messages, context_meta=meta)
        if hasattr(self.llm, "complete"):
            # Simple completion-style interface
            prompt = "\n".join(m["content"] for m in messages)
            return self.llm.complete(prompt, context_meta=meta)
        if callable(self.llm):
            return self.llm(messages, meta)
        raise TypeError(f"LLM object {type(self.llm)} has no chat/complete method and is not callable")

    def _force_final(self) -> RLMResult:
        """Produce a forced result from whatever is in the REPL."""
        # Try common variable names
        for name in ("result", "answer", "output", "final_answer"):
            if name in self.executor.repl:
                return RLMResult(
                    value=self.executor.repl[name],
                    source="forced",
                    raw_argument=name,
                )
        return RLMResult(
            value="<no result — budget exhausted>",
            source="forced",
            raw_argument="",
        )
