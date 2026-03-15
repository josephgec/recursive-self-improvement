"""BDM scoring for data, programs, and rules."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from src.bdm.block_decomposer import BlockDecomposer
from src.bdm.compressor import CompressionBaseline
from src.bdm.ctm_table import CTMTable
from src.utils.encoding import data_to_binary
from src.utils.program_length import measure_program_length


@dataclass
class BDMScore:
    """Result of BDM scoring."""

    bdm_value: float
    num_blocks: int
    num_unique_blocks: int
    block_details: List[Dict[str, Any]] = field(default_factory=list)
    normalized_bdm: float = 0.0  # BDM / length

    def __post_init__(self) -> None:
        if self.block_details:
            total_chars = sum(
                len(d.get("content", "")) * d.get("multiplicity", 1)
                for d in self.block_details
            )
            if total_chars > 0:
                self.normalized_bdm = self.bdm_value / total_chars


@dataclass
class RuleScore:
    """Score for a synthesized rule combining accuracy and complexity."""

    bdm_score: float
    program_length: float
    mdl_score: float  # Minimum Description Length
    accuracy: float = 0.0
    residual_complexity: float = 0.0

    @property
    def fitness(self) -> float:
        """Combined fitness: higher accuracy, lower complexity is better."""
        if self.accuracy <= 0:
            return float("inf")
        return self.mdl_score / self.accuracy


class BDMScorer:
    """Scorer that computes BDM complexity for data and programs.

    BDM formula: BDM(s) = sum(K_CTM(b_i) + log2(n_i))
    where b_i are unique blocks and n_i is their multiplicity.
    """

    def __init__(
        self,
        ctm_table: Optional[CTMTable] = None,
        block_size: int = 12,
    ) -> None:
        self.ctm_table = ctm_table or CTMTable.with_fallback_only()
        self.decomposer = BlockDecomposer(
            ctm_table=self.ctm_table, block_size=block_size
        )
        self.compressor = CompressionBaseline()

    def score(self, data: Union[str, list, int, float]) -> BDMScore:
        """Compute BDM score for arbitrary data.

        Args:
            data: Input data (string, list, number).

        Returns:
            BDMScore with the computed complexity.
        """
        blocks = self.decomposer.decompose(data)

        bdm_value = 0.0
        block_details = []
        total_blocks = 0

        for block in blocks:
            # BDM formula: K_CTM(b_i) + log2(n_i)
            contribution = block.ctm_complexity + math.log2(block.multiplicity)
            bdm_value += contribution
            total_blocks += block.multiplicity
            block_details.append(
                {
                    "content": block.content,
                    "position": block.position,
                    "ctm_complexity": block.ctm_complexity,
                    "multiplicity": block.multiplicity,
                    "contribution": contribution,
                }
            )

        return BDMScore(
            bdm_value=bdm_value,
            num_blocks=total_blocks,
            num_unique_blocks=len(blocks),
            block_details=block_details,
        )

    def score_program(self, code: str) -> BDMScore:
        """Compute BDM score for a program (source code).

        Args:
            code: Python source code string.

        Returns:
            BDMScore for the program text.
        """
        return self.score(code)

    def score_rule(
        self,
        code: str,
        inputs: List[Any],
        outputs: List[Any],
    ) -> RuleScore:
        """Score a rule combining program complexity and fit quality.

        MDL = K(program) + K(residuals)
        where residuals capture what the program fails to explain.

        Args:
            code: Python source code of the rule.
            inputs: Input examples.
            outputs: Expected output examples.

        Returns:
            RuleScore with BDM, program length, MDL, and accuracy.
        """
        # Program complexity via BDM
        program_bdm = self.score(code)

        # Program length via AST
        prog_length = measure_program_length(code)

        # Try to compute accuracy and residuals
        correct = 0
        residuals = []
        try:
            namespace: dict = {}
            exec(code, namespace)
            # Find the callable
            func = None
            for name, obj in namespace.items():
                if callable(obj) and not name.startswith("_"):
                    func = obj
                    break

            if func is not None:
                for inp, expected in zip(inputs, outputs):
                    try:
                        if isinstance(inp, (list, tuple)):
                            result = func(*inp)
                        else:
                            result = func(inp)
                        if result == expected:
                            correct += 1
                        else:
                            residuals.append(str(result))
                    except Exception:
                        residuals.append("ERROR")
        except Exception:
            pass

        accuracy = correct / len(inputs) if inputs else 0.0

        # Residual complexity
        residual_str = "|".join(residuals) if residuals else ""
        residual_complexity = self.score(residual_str).bdm_value if residual_str else 0.0

        # MDL = program complexity + residual complexity
        mdl = program_bdm.bdm_value + residual_complexity

        return RuleScore(
            bdm_score=program_bdm.bdm_value,
            program_length=prog_length.total_score,
            mdl_score=mdl,
            accuracy=accuracy,
            residual_complexity=residual_complexity,
        )

    def compare_to_baselines(
        self, data: Union[str, list]
    ) -> Dict[str, float]:
        """Compare BDM score against compression baselines.

        Args:
            data: Input data.

        Returns:
            Dictionary of complexity measurements.
        """
        bdm = self.score(data)
        binary = self.decomposer.encode_to_binary(data)

        baselines = self.compressor.compare_all(binary)
        baselines["bdm"] = bdm.bdm_value
        baselines["bdm_normalized"] = bdm.normalized_bdm

        return baselines
