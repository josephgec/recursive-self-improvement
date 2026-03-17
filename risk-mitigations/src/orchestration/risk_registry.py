"""Risk registry with 6 risk domains.

Central registry for all operational risks, providing unified status checking.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional
from enum import Enum


class RiskSeverity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class RiskStatus:
    """Status of a single risk."""
    risk_id: str
    name: str
    domain: str
    severity: str
    score: float  # 0-1, higher = more concerning
    details: Dict[str, Any] = field(default_factory=dict)
    mitigations_active: List[str] = field(default_factory=list)

    @property
    def is_critical(self) -> bool:
        return self.severity == "critical"

    @property
    def needs_attention(self) -> bool:
        return self.severity in ("high", "critical")


@dataclass
class Risk:
    """Definition of a risk domain."""
    risk_id: str
    name: str
    domain: str
    description: str
    checker: Optional[Callable[[], RiskStatus]] = None

    def check(self) -> RiskStatus:
        """Check this risk's current status."""
        if self.checker:
            return self.checker()
        return RiskStatus(
            risk_id=self.risk_id,
            name=self.name,
            domain=self.domain,
            severity="low",
            score=0.0,
            details={"message": "No checker configured"},
        )


@dataclass
class RiskDashboard:
    """Dashboard summarizing all risk statuses."""
    statuses: List[RiskStatus] = field(default_factory=list)
    overall_severity: str = "low"
    overall_score: float = 0.0
    critical_count: int = 0
    high_count: int = 0
    timestamp: str = ""

    @property
    def needs_immediate_action(self) -> bool:
        return self.critical_count > 0

    @property
    def total_risks(self) -> int:
        return len(self.statuses)


# Default 6 risk domains
DEFAULT_RISKS = [
    Risk(
        risk_id="R1",
        name="Model Collapse",
        domain="collapse",
        description="Risk of model quality degradation through iterative training on synthetic data",
    ),
    Risk(
        risk_id="R2",
        name="Self-Modification Safety",
        domain="self_mod",
        description="Risk from unchecked self-modifications causing regressions or safety issues",
    ),
    Risk(
        risk_id="R3",
        name="Reward Hacking",
        domain="reward",
        description="Risk of the agent gaming its reward signal rather than genuinely improving",
    ),
    Risk(
        risk_id="R4",
        name="Cost Overrun",
        domain="cost",
        description="Risk of exceeding computational and financial budgets",
    ),
    Risk(
        risk_id="R5",
        name="Constraint Violations",
        domain="constraints",
        description="Risk of constraint violations or overly restrictive constraints blocking progress",
    ),
    Risk(
        risk_id="R6",
        name="Publication Readiness",
        domain="publication",
        description="Risk of missing publication deadlines or submitting incomplete work",
    ),
]


class RiskRegistry:
    """Central registry for all 6 risk domains.

    Provides unified risk checking across all domains.
    """

    def __init__(self, risks: Optional[List[Risk]] = None):
        import copy
        self._risks: Dict[str, Risk] = {}
        for risk in (risks or DEFAULT_RISKS):
            self._risks[risk.risk_id] = copy.copy(risk)

    def register_checker(self, risk_id: str, checker: Callable[[], RiskStatus]) -> None:
        """Register a checker function for a risk.

        Args:
            risk_id: The risk ID (e.g., "R1").
            checker: A callable returning RiskStatus.
        """
        if risk_id not in self._risks:
            raise KeyError(f"Unknown risk: {risk_id}")
        self._risks[risk_id].checker = checker

    def check(self, risk_id: str) -> RiskStatus:
        """Check a single risk.

        Args:
            risk_id: The risk ID.

        Returns:
            RiskStatus for the risk.
        """
        if risk_id not in self._risks:
            raise KeyError(f"Unknown risk: {risk_id}")
        return self._risks[risk_id].check()

    def check_all(self) -> RiskDashboard:
        """Check all registered risks.

        Returns:
            RiskDashboard with all statuses.
        """
        from datetime import datetime

        statuses = []
        for risk in self._risks.values():
            status = risk.check()
            statuses.append(status)

        critical = sum(1 for s in statuses if s.severity == "critical")
        high = sum(1 for s in statuses if s.severity == "high")

        if critical > 0:
            overall_severity = "critical"
        elif high > 0:
            overall_severity = "high"
        elif any(s.severity == "medium" for s in statuses):
            overall_severity = "medium"
        else:
            overall_severity = "low"

        scores = [s.score for s in statuses]
        overall_score = max(scores) if scores else 0.0

        return RiskDashboard(
            statuses=statuses,
            overall_severity=overall_severity,
            overall_score=overall_score,
            critical_count=critical,
            high_count=high,
            timestamp=datetime.now().isoformat(),
        )

    def get_risk(self, risk_id: str) -> Risk:
        """Get a risk definition."""
        if risk_id not in self._risks:
            raise KeyError(f"Unknown risk: {risk_id}")
        return self._risks[risk_id]

    def get_all_risks(self) -> List[Risk]:
        """Return all registered risks."""
        return list(self._risks.values())

    @property
    def risk_count(self) -> int:
        return len(self._risks)
