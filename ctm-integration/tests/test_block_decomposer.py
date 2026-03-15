"""Tests for block decomposition."""

import pytest

from src.bdm.block_decomposer import BlockDecomposer, Block
from src.bdm.ctm_table import CTMTable


class TestBlockDecomposition:
    """Tests for decomposing data into blocks."""

    def test_decompose_binary_string(self, decomposer):
        """Decomposing a binary string should produce blocks."""
        blocks = decomposer.decompose("01010101")
        assert len(blocks) > 0
        assert all(isinstance(b, Block) for b in blocks)

    def test_block_content_is_binary(self, decomposer):
        """All block content should be binary strings."""
        blocks = decomposer.decompose("01100110")
        for block in blocks:
            assert all(c in "01" for c in block.content)

    def test_block_sizes_respect_max(self, decomposer):
        """No block should exceed the configured block size."""
        blocks = decomposer.decompose("0110011001100110")
        for block in blocks:
            assert len(block.content) <= decomposer.block_size

    def test_decompose_covers_full_input(self, decomposer):
        """Blocks should cover the entire input."""
        data = "01010101"
        blocks = decomposer.decompose(data)
        total_chars = sum(len(b.content) * b.multiplicity for b in blocks)
        assert total_chars >= len(data)

    def test_encode_binary_passthrough(self, decomposer):
        """Binary strings should pass through encoding unchanged."""
        binary = "01101001"
        result = decomposer.encode_to_binary(binary)
        assert result == binary

    def test_encode_non_binary_string(self, decomposer):
        """Non-binary strings should be encoded to binary."""
        result = decomposer.encode_to_binary("AB")
        assert all(c in "01" for c in result)
        assert len(result) == 16  # 2 chars * 8 bits

    def test_encode_integer(self, decomposer):
        """Integers should be encoded to binary."""
        result = decomposer.encode_to_binary(42)
        assert all(c in "01" for c in result)


class TestMultiplicity:
    """Tests for block multiplicity counting."""

    def test_repeated_blocks_have_multiplicity(self, decomposer):
        """Repeated identical blocks should have multiplicity > 1."""
        # With block_size=4: "0101" "0101" "0101" "0101"
        data = "0101010101010101"
        blocks = decomposer.decompose(data)
        # All blocks should be "0101" with multiplicity = 4
        for block in blocks:
            if block.content == "0101":
                assert block.multiplicity == 4
                break
        else:
            pytest.fail("Expected to find block '0101' with multiplicity")

    def test_unique_blocks_have_multiplicity_one(self, decomposer):
        """Unique blocks should have multiplicity 1."""
        # With block_size=4: "0001" "0010" "0100" "1000"
        data = "0001001001001000"
        blocks = decomposer.decompose(data)
        # Check that some blocks have multiplicity 1
        some_unique = any(b.multiplicity == 1 for b in blocks)
        assert some_unique

    def test_ctm_complexity_assigned(self, decomposer):
        """Each block should have a CTM complexity value."""
        blocks = decomposer.decompose("01010101")
        for block in blocks:
            assert block.ctm_complexity >= 0

    def test_position_tracking(self, decomposer):
        """Blocks should track their first position."""
        blocks = decomposer.decompose("01010101")
        positions = [b.position for b in blocks]
        # Positions should be non-negative and sorted
        assert all(p >= 0 for p in positions)
        assert positions == sorted(positions)


class TestBlockDecomposerConfig:
    """Tests for different decomposer configurations."""

    def test_different_block_sizes(self, ctm_table):
        """Different block sizes should produce different decompositions."""
        data = "0110011001100110"

        d4 = BlockDecomposer(ctm_table=ctm_table, block_size=4)
        d8 = BlockDecomposer(ctm_table=ctm_table, block_size=8)

        blocks4 = d4.decompose(data)
        blocks8 = d8.decompose(data)

        # More blocks with smaller block size
        total4 = sum(b.multiplicity for b in blocks4)
        total8 = sum(b.multiplicity for b in blocks8)
        assert total4 >= total8
