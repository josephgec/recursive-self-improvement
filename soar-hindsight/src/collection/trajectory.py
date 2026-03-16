"""Core dataclasses for representing evolutionary search trajectories."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class TaskSpec:
    """Specification for a programming task."""

    task_id: str
    description: str
    test_cases: List[Dict[str, Any]] = field(default_factory=list)
    difficulty: str = "medium"
    tags: List[str] = field(default_factory=list)
    constraints: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "description": self.description,
            "test_cases": self.test_cases,
            "difficulty": self.difficulty,
            "tags": list(self.tags),
            "constraints": dict(self.constraints),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TaskSpec":
        return cls(
            task_id=data["task_id"],
            description=data["description"],
            test_cases=data.get("test_cases", []),
            difficulty=data.get("difficulty", "medium"),
            tags=data.get("tags", []),
            constraints=data.get("constraints", {}),
        )


@dataclass
class ImprovementStep:
    """A single step in an improvement chain."""

    step_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    operator: str = ""
    code_before: str = ""
    code_after: str = ""
    fitness_before: float = 0.0
    fitness_after: float = 0.0
    error_before: Optional[str] = None
    error_after: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def fitness_delta(self) -> float:
        return self.fitness_after - self.fitness_before

    @property
    def is_improvement(self) -> bool:
        return self.fitness_after > self.fitness_before

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_id": self.step_id,
            "operator": self.operator,
            "code_before": self.code_before,
            "code_after": self.code_after,
            "fitness_before": self.fitness_before,
            "fitness_after": self.fitness_after,
            "error_before": self.error_before,
            "error_after": self.error_after,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ImprovementStep":
        return cls(
            step_id=data.get("step_id", str(uuid.uuid4())[:8]),
            operator=data.get("operator", ""),
            code_before=data.get("code_before", ""),
            code_after=data.get("code_after", ""),
            fitness_before=data.get("fitness_before", 0.0),
            fitness_after=data.get("fitness_after", 0.0),
            error_before=data.get("error_before"),
            error_after=data.get("error_after"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class IndividualRecord:
    """Record of a single individual in the evolutionary population."""

    individual_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    generation: int = 0
    code: str = ""
    fitness: float = 0.0
    parent_ids: List[str] = field(default_factory=list)
    operator: str = ""
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_solved(self) -> bool:
        return self.fitness >= 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "individual_id": self.individual_id,
            "generation": self.generation,
            "code": self.code,
            "fitness": self.fitness,
            "parent_ids": list(self.parent_ids),
            "operator": self.operator,
            "error": self.error,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "IndividualRecord":
        return cls(
            individual_id=data.get("individual_id", str(uuid.uuid4())[:8]),
            generation=data.get("generation", 0),
            code=data.get("code", ""),
            fitness=data.get("fitness", 0.0),
            parent_ids=data.get("parent_ids", []),
            operator=data.get("operator", ""),
            error=data.get("error"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class SearchTrajectory:
    """Complete trajectory of an evolutionary search run."""

    trajectory_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    task: Optional[TaskSpec] = None
    individuals: List[IndividualRecord] = field(default_factory=list)
    improvement_steps: List[ImprovementStep] = field(default_factory=list)
    best_fitness: float = 0.0
    total_generations: int = 0
    solved: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def best_individual(self) -> Optional[IndividualRecord]:
        if not self.individuals:
            return None
        return max(self.individuals, key=lambda ind: ind.fitness)

    @property
    def solved_individuals(self) -> List[IndividualRecord]:
        return [ind for ind in self.individuals if ind.is_solved]

    @property
    def failed_individuals(self) -> List[IndividualRecord]:
        return [ind for ind in self.individuals if ind.error is not None]

    @property
    def generations(self) -> List[List[IndividualRecord]]:
        if not self.individuals:
            return []
        max_gen = max(ind.generation for ind in self.individuals)
        result = []
        for g in range(max_gen + 1):
            gen_individuals = [
                ind for ind in self.individuals if ind.generation == g
            ]
            if gen_individuals:
                result.append(gen_individuals)
        return result

    def add_individual(self, individual: IndividualRecord) -> None:
        self.individuals.append(individual)
        if individual.fitness > self.best_fitness:
            self.best_fitness = individual.fitness
        if individual.is_solved:
            self.solved = True
        if individual.generation >= self.total_generations:
            self.total_generations = individual.generation + 1

    def extract_improvement_chain(self) -> List[ImprovementStep]:
        """Extract sequential improvement steps from the trajectory."""
        if len(self.individuals) < 2:
            return []

        sorted_inds = sorted(self.individuals, key=lambda x: (x.generation, -x.fitness))
        steps = []
        prev = sorted_inds[0]

        for curr in sorted_inds[1:]:
            if curr.fitness > prev.fitness:
                step = ImprovementStep(
                    operator=curr.operator,
                    code_before=prev.code,
                    code_after=curr.code,
                    fitness_before=prev.fitness,
                    fitness_after=curr.fitness,
                    error_before=prev.error,
                    error_after=curr.error,
                )
                steps.append(step)
                prev = curr

        self.improvement_steps = steps
        return steps

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trajectory_id": self.trajectory_id,
            "task": self.task.to_dict() if self.task else None,
            "individuals": [ind.to_dict() for ind in self.individuals],
            "improvement_steps": [s.to_dict() for s in self.improvement_steps],
            "best_fitness": self.best_fitness,
            "total_generations": self.total_generations,
            "solved": self.solved,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SearchTrajectory":
        task = TaskSpec.from_dict(data["task"]) if data.get("task") else None
        individuals = [
            IndividualRecord.from_dict(ind) for ind in data.get("individuals", [])
        ]
        steps = [
            ImprovementStep.from_dict(s) for s in data.get("improvement_steps", [])
        ]
        traj = cls(
            trajectory_id=data.get("trajectory_id", str(uuid.uuid4())[:8]),
            task=task,
            individuals=individuals,
            improvement_steps=steps,
            best_fitness=data.get("best_fitness", 0.0),
            total_generations=data.get("total_generations", 0),
            solved=data.get("solved", False),
            metadata=data.get("metadata", {}),
        )
        return traj
