"""Functional API: rlm.completion() and RLMCompletionAPI."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.core.config import load_config, merge_configs, DEFAULT_CONFIG
from src.core.session import RLMSession, SessionResult
from src.recursion.depth_controller import DepthController


class RLMCompletionAPI:
    """Object-oriented API for RLM completion."""

    def __init__(
        self,
        llm: Any = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.config = merge_configs(DEFAULT_CONFIG, config or {})
        self.llm = llm

    def complete(
        self,
        prompt: str,
        context: Any,
        model: Optional[str] = None,
        llm: Optional[Any] = None,
        **kwargs: Any,
    ) -> str:
        """Run an RLM session and return the result as a string."""
        session = self.create_session(llm=llm or self.llm, **kwargs)
        result = session.run(query=prompt, context=context)
        return str(result)

    def create_session(
        self,
        llm: Optional[Any] = None,
        max_iterations: Optional[int] = None,
        depth: int = 0,
        **kwargs: Any,
    ) -> RLMSession:
        """Create and return a new RLMSession."""
        used_llm = llm or self.llm
        if used_llm is None:
            raise ValueError("No LLM provided. Pass llm= to complete() or to RLMCompletionAPI().")
        iters = max_iterations or self.config.get("max_iterations", 10)
        dc = DepthController(
            max_depth=self.config.get("max_depth", 3),
            max_iterations=iters,
            max_sub_queries=self.config.get("recursion", {}).get("max_sub_queries", 5),
            budget_fraction=self.config.get("recursion", {}).get("budget_fraction", 0.5),
        )
        return RLMSession(
            llm=used_llm,
            max_iterations=iters,
            depth=depth,
            depth_controller=dc,
            forced_final=self.config.get("forced_final", True),
            config=self.config,
        )

    def complete_batch(
        self,
        prompts: List[Dict[str, Any]],
        llm: Optional[Any] = None,
    ) -> List[str]:
        """Run multiple completions sequentially.

        Each item in *prompts* must have ``prompt`` and ``context`` keys.
        """
        results: List[str] = []
        for item in prompts:
            r = self.complete(
                prompt=item["prompt"],
                context=item["context"],
                llm=llm,
            )
            results.append(r)
        return results


def completion(
    prompt: str,
    context: Any,
    model: str = "mock",
    llm: Any = None,
    config: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> str:
    """Drop-in replacement for ``llm.completion()``.

    Loads *context* into a REPL and lets the LLM interact with it via code.

    Args:
        prompt: The query / instruction.
        context: The context to load (str, list, dict, or Path).
        model: Model identifier (default "mock").
        llm: An LLM object with a ``.chat()`` or ``.complete()`` method.
        config: Optional configuration overrides.

    Returns:
        The final answer as a string.
    """
    api = RLMCompletionAPI(llm=llm, config=config)
    return api.complete(prompt=prompt, context=context, model=model, **kwargs)
