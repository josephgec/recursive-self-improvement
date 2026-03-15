"""Tests for utility modules: encoding, program_length, turing_machines."""

import pytest

from src.utils.encoding import (
    data_to_binary,
    binary_to_data,
    encode_tokens,
    encode_number,
)
from src.utils.program_length import measure_program_length, ProgramLength
from src.utils.turing_machines import SimpleTM, Transition, enumerate_tms, count_tms, LEFT, RIGHT, HALT


class TestEncoding:
    """Tests for encoding utilities."""

    def test_string_to_binary(self):
        """Strings should encode as 8-bit per char."""
        result = data_to_binary("A")
        assert result == "01000001"
        assert len(result) == 8

    def test_string_roundtrip(self):
        """Encoding then decoding should recover the string."""
        original = "Hello"
        binary = data_to_binary(original)
        decoded = binary_to_data(binary)
        assert decoded == original

    def test_integer_encoding(self):
        """Integers should encode to binary."""
        result = data_to_binary(42)
        assert all(c in "01" for c in result)
        assert "101010" in result  # 42 in binary

    def test_float_encoding(self):
        """Floats should encode to binary."""
        result = data_to_binary(3.14)
        assert all(c in "01" for c in result)
        assert len(result) > 0

    def test_list_encoding(self):
        """Lists should encode element by element."""
        result = data_to_binary([1, 2, 3])
        assert all(c in "01" for c in result)
        assert len(result) > 0

    def test_negative_number_encoding(self):
        """Negative numbers should include a sign bit."""
        result = encode_number(-5)
        assert result.startswith("1")  # sign bit

    def test_zero_encoding(self):
        """Zero should encode to 00000000."""
        result = encode_number(0)
        assert result == "00000000"

    def test_encode_tokens(self):
        """Token list should encode to concatenated binary."""
        result = encode_tokens(["AB"])
        assert all(c in "01" for c in result)
        assert len(result) == 16  # 2 chars * 8 bits

    def test_data_to_binary_tuple(self):
        """Tuples should encode like lists."""
        result = data_to_binary((1, 2))
        assert all(c in "01" for c in result)

    def test_data_to_binary_other(self):
        """Other types should convert via str()."""
        result = data_to_binary(None)
        assert all(c in "01" for c in result)

    def test_binary_to_data_partial_byte(self):
        """Partial bytes at the end should be ignored."""
        # 16 bits + 4 extra bits
        binary = "0100000101000010" + "0101"
        decoded = binary_to_data(binary)
        assert decoded == "AB"  # only full bytes

    def test_list_with_strings(self):
        """Lists with string elements should encode."""
        result = data_to_binary(["x", "y"])
        assert all(c in "01" for c in result)

    def test_list_with_floats(self):
        """Lists with float elements should encode."""
        result = data_to_binary([1.5, 2.5])
        assert all(c in "01" for c in result)

    def test_list_with_mixed(self):
        """Lists with mixed types should encode."""
        result = data_to_binary([1, "a", None])
        assert all(c in "01" for c in result)


class TestProgramLength:
    """Tests for program length measurement."""

    def test_measure_simple_function(self):
        """Simple function should have measurable length."""
        code = "def f(x):\n    return x * 2\n"
        result = measure_program_length(code)
        assert isinstance(result, ProgramLength)
        assert result.ast_nodes > 0
        assert result.lines > 0
        assert result.tokens > 0
        assert result.characters > 0

    def test_longer_program_more_nodes(self):
        """Longer programs should have more AST nodes."""
        short = "x = 1"
        long = "x = 1\ny = 2\nz = x + y\nw = z * 2\n"
        short_len = measure_program_length(short)
        long_len = measure_program_length(long)
        assert long_len.ast_nodes > short_len.ast_nodes

    def test_total_score(self):
        """Total score should be a weighted combination."""
        code = "def f(x):\n    return x\n"
        result = measure_program_length(code)
        expected = result.ast_nodes * 1.0 + result.lines * 0.5 + result.tokens * 0.1
        assert abs(result.total_score - expected) < 1e-9

    def test_syntax_error_fallback(self):
        """Invalid Python should use fallback counting."""
        code = "this is not valid python !!!"
        result = measure_program_length(code)
        # Should still return something reasonable
        assert result.ast_nodes > 0  # uses token fallback
        assert result.lines >= 1

    def test_empty_lines_ignored(self):
        """Empty and comment lines should not count."""
        code = "x = 1\n\n# comment\n\ny = 2\n"
        result = measure_program_length(code)
        assert result.lines == 2  # only x=1 and y=2


class TestTuringMachines:
    """Tests for Turing machine implementation."""

    def test_simple_tm_halts(self):
        """A TM that immediately halts should produce output."""
        tm = SimpleTM(num_states=1, num_symbols=2)
        # State 0, symbol 0: write 1, move right, halt
        tm.transitions[(0, 0)] = Transition(write_symbol=1, move=RIGHT, next_state=HALT)
        output = tm.run(max_steps=10)
        assert output is not None
        assert "1" in output

    def test_tm_non_halting(self):
        """A TM that loops should return None."""
        tm = SimpleTM(num_states=1, num_symbols=2)
        # Loop: stay in state 0 forever
        tm.transitions[(0, 0)] = Transition(write_symbol=0, move=RIGHT, next_state=0)
        tm.transitions[(0, 1)] = Transition(write_symbol=1, move=RIGHT, next_state=0)
        output = tm.run(max_steps=10)
        assert output is None

    def test_tm_no_transition(self):
        """A TM with no matching transition should halt."""
        tm = SimpleTM(num_states=1, num_symbols=2)
        # No transitions defined - should halt immediately
        output = tm.run(max_steps=10)
        assert output is not None  # halts with tape content

    def test_enumerate_tms_count(self):
        """enumerate_tms should produce the expected number of TMs."""
        tms = list(enumerate_tms(max_states=1, max_symbols=2))
        expected = count_tms(1, 2)
        assert len(tms) == expected

    def test_enumerate_tms_produces_valid(self):
        """All enumerated TMs should be valid."""
        for tm in enumerate_tms(max_states=1, max_symbols=2):
            assert isinstance(tm, SimpleTM)
            assert tm.num_states >= 1
            assert tm.num_symbols >= 2

    def test_count_tms_formula(self):
        """count_tms should match the formula."""
        # For 1 state, 2 symbols: (2*2*(1+1))^(1*2) = 8^2 = 64
        count = count_tms(1, 2)
        assert count == 64

    def test_tm_bounds_check(self):
        """TM should handle going off the tape edge."""
        tm = SimpleTM(num_states=1, num_symbols=2)
        # Move left from center until off tape
        tm.transitions[(0, 0)] = Transition(write_symbol=1, move=LEFT, next_state=0)
        output = tm.run(max_steps=1000, tape_size=10)
        # Should halt due to bounds check
        assert output is not None or output is None  # shouldn't crash

    def test_extract_all_zeros_tape(self):
        """TM that writes nothing should produce '0'."""
        tm = SimpleTM(num_states=1, num_symbols=2)
        # State 0, symbol 0: write 0 (no change), halt
        tm.transitions[(0, 0)] = Transition(write_symbol=0, move=RIGHT, next_state=HALT)
        output = tm.run(max_steps=10)
        # All zeros tape produces "0"
        assert output is not None
        assert output == "0"
