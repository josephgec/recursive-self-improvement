"""SubQuerySpawner: inject rlm_sub_query into the REPL."""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, TYPE_CHECKING

from src.recursion.depth_controller import DepthController, BudgetExhaustedError

if TYPE_CHECKING:
    from src.core.session import RLMSession


class SubQuerySpawner:
    """Create and inject sub-query functions into a parent session's REPL."""

    def __init__(
        self,
        depth_controller: DepthController,
        llm_factory: Callable,
        parent_depth: int = 0,
        parent_session_id: Optional[str] = None,
        registry: Optional[Any] = None,
    ) -> None:
        self.depth_controller = depth_controller
        self.llm_factory = llm_factory
        self.parent_depth = parent_depth
        self.parent_session_id = parent_session_id
        self.registry = registry
        self.sub_sessions: list = []

    def create_sub_query_function(self, parent_session: Optional[Any] = None) -> Callable:
        """Return an ``rlm_sub_query(query, context)`` callable.

        When invoked, it spawns a child RLMSession at depth+1.
        """
        spawner = self

        def rlm_sub_query(query: str, context: str) -> str:
            return spawner._spawn_session(query, context)

        return rlm_sub_query

    def _spawn_session(self, query: str, context: str) -> str:
        """Spawn a child RLM session and return its result as a string."""
        from src.core.session import RLMSession

        child_depth = self.depth_controller.register_sub_query(self.parent_depth)
        child_budget = self.depth_controller.allocate_sub_budget(
            self.depth_controller.max_iterations
        )

        llm = self.llm_factory()
        child_session = RLMSession(
            llm=llm,
            max_iterations=child_budget,
            depth=child_depth,
            depth_controller=self.depth_controller,
            parent_session_id=self.parent_session_id,
        )
        result = child_session.run(query=query, context=context)
        self.sub_sessions.append(child_session)
        return str(result.result) if result.result is not None else ""

    def inject_into_repl(self, repl: Dict[str, Any], parent_session: Optional[Any] = None) -> None:
        """Inject rlm_sub_query into the given REPL namespace."""
        fn = self.create_sub_query_function(parent_session)
        repl["rlm_sub_query"] = fn
