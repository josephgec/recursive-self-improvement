"""Block decomposition for the Block Decomposition Method (BDM)."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import List, Optional, Union

from src.bdm.ctm_table import CTMTable
from src.utils.encoding import data_to_binary


@dataclass
class Block:
    """A block from the decomposition of a binary string."""

    content: str
    position: int
    ctm_complexity: float = 0.0
    multiplicity: int = 1


class BlockDecomposer:
    """Decomposes data into blocks for BDM computation.

    The BDM formula is: BDM(s) = sum(K_CTM(b_i) + log2(n_i))
    where b_i are unique blocks and n_i is their multiplicity.
    """

    def __init__(
        self,
        ctm_table: Optional[CTMTable] = None,
        block_size: int = 12,
        overlap: bool = False,
    ) -> None:
        """Initialize the block decomposer.

        Args:
            ctm_table: CTM lookup table. If None, uses fallback-only table.
            block_size: Size of blocks to decompose into.
            overlap: Whether to use overlapping blocks.
        """
        self.ctm_table = ctm_table or CTMTable.with_fallback_only()
        self.block_size = block_size
        self.overlap = overlap

    def decompose(self, data: Union[str, list, int, float]) -> List[Block]:
        """Decompose data into blocks with CTM complexity and multiplicity.

        Args:
            data: Input data to decompose. Will be converted to binary.

        Returns:
            List of Block objects with complexity and multiplicity computed.
        """
        binary = self.encode_to_binary(data)
        raw_blocks = self._split_into_blocks(binary)

        # Count multiplicities
        content_counts: Counter = Counter()
        content_first_pos: dict = {}
        for content, pos in raw_blocks:
            content_counts[content] += 1
            if content not in content_first_pos:
                content_first_pos[content] = pos

        # Build block list with complexity and multiplicity
        blocks = []
        for content, count in content_counts.items():
            complexity = self.ctm_table.lookup(content)
            blocks.append(
                Block(
                    content=content,
                    position=content_first_pos[content],
                    ctm_complexity=complexity,
                    multiplicity=count,
                )
            )

        # Sort by position
        blocks.sort(key=lambda b: b.position)
        return blocks

    def encode_to_binary(self, data: Union[str, list, int, float]) -> str:
        """Convert input data to binary string representation.

        If data is already a binary string (only 0s and 1s), return as-is.

        Args:
            data: Input data.

        Returns:
            Binary string.
        """
        if isinstance(data, str) and all(c in "01" for c in data) and len(data) > 0:
            return data
        return data_to_binary(data)

    def _split_into_blocks(self, binary: str) -> List[tuple]:
        """Split binary string into fixed-size blocks.

        Args:
            binary: Binary string to split.

        Returns:
            List of (content, position) tuples.
        """
        blocks = []
        if self.overlap:
            step = 1
        else:
            step = self.block_size

        for i in range(0, len(binary), step):
            block_content = binary[i : i + self.block_size]
            if len(block_content) > 0:
                blocks.append((block_content, i))
            if self.overlap and i + self.block_size >= len(binary):
                break

        return blocks
