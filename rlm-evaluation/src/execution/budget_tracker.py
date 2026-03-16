"""Budget tracking for token usage and costs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class UsageRecord:
    """A single usage record."""
    system: str
    task_id: str
    input_tokens: int
    output_tokens: int
    cost: float
    num_calls: int = 1


class BudgetTracker:
    """Track token usage, cost, and call counts across evaluation runs."""

    def __init__(self, budget_limit: float = float("inf")) -> None:
        self.budget_limit = budget_limit
        self._records: List[UsageRecord] = []
        self._total_cost: float = 0.0
        self._total_input_tokens: int = 0
        self._total_output_tokens: int = 0
        self._total_calls: int = 0

    def track(
        self,
        system: str,
        task_id: str,
        input_tokens: int,
        output_tokens: int,
        cost: float,
        num_calls: int = 1,
    ) -> None:
        """Record a usage entry."""
        record = UsageRecord(
            system=system,
            task_id=task_id,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=cost,
            num_calls=num_calls,
        )
        self._records.append(record)
        self._total_cost += cost
        self._total_input_tokens += input_tokens
        self._total_output_tokens += output_tokens
        self._total_calls += num_calls

    @property
    def total_cost(self) -> float:
        return self._total_cost

    @property
    def total_input_tokens(self) -> int:
        return self._total_input_tokens

    @property
    def total_output_tokens(self) -> int:
        return self._total_output_tokens

    @property
    def total_tokens(self) -> int:
        return self._total_input_tokens + self._total_output_tokens

    @property
    def total_calls(self) -> int:
        return self._total_calls

    @property
    def budget_remaining(self) -> float:
        return self.budget_limit - self._total_cost

    @property
    def over_budget(self) -> bool:
        return self._total_cost > self.budget_limit

    def cost_by_system(self) -> Dict[str, float]:
        """Break down costs by system."""
        costs: Dict[str, float] = {}
        for record in self._records:
            costs[record.system] = costs.get(record.system, 0.0) + record.cost
        return costs

    def tokens_by_system(self) -> Dict[str, int]:
        """Break down total tokens by system."""
        tokens: Dict[str, int] = {}
        for record in self._records:
            total = record.input_tokens + record.output_tokens
            tokens[record.system] = tokens.get(record.system, 0) + total
        return tokens

    def calls_by_system(self) -> Dict[str, int]:
        """Break down call counts by system."""
        calls: Dict[str, int] = {}
        for record in self._records:
            calls[record.system] = calls.get(record.system, 0) + record.num_calls
        return calls

    def summary(self) -> Dict[str, object]:
        """Return a summary of all tracked usage."""
        return {
            "total_cost": self._total_cost,
            "total_input_tokens": self._total_input_tokens,
            "total_output_tokens": self._total_output_tokens,
            "total_tokens": self.total_tokens,
            "total_calls": self._total_calls,
            "budget_limit": self.budget_limit,
            "budget_remaining": self.budget_remaining,
            "over_budget": self.over_budget,
            "cost_by_system": self.cost_by_system(),
            "tokens_by_system": self.tokens_by_system(),
            "num_records": len(self._records),
        }

    @property
    def records(self) -> List[UsageRecord]:
        return list(self._records)
