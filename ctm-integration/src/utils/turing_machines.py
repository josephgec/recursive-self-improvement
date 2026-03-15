"""Simple Turing machine implementation and enumeration for CTM table building."""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Dict, Generator, List, Optional, Tuple


# Directions
LEFT = -1
RIGHT = 1

# Special halt state
HALT = -1


@dataclass
class Transition:
    """A single TM transition: (write_symbol, move_direction, next_state)."""

    write_symbol: int
    move: int  # LEFT (-1) or RIGHT (1)
    next_state: int  # -1 means HALT


@dataclass
class SimpleTM:
    """A simple deterministic Turing machine.

    States are numbered 0..num_states-1, with HALT=-1.
    Symbols are numbered 0..num_symbols-1.
    The tape is initialized to all zeros (blank = 0).
    """

    num_states: int
    num_symbols: int
    transitions: Dict[Tuple[int, int], Transition] = field(default_factory=dict)

    def run(self, max_steps: int = 100, tape_size: int = 100) -> Optional[str]:
        """Run the TM and return the tape output as a binary string.

        Args:
            max_steps: Maximum steps before declaring non-halting.
            tape_size: Size of the tape.

        Returns:
            Binary string of the tape contents if the TM halts, None otherwise.
        """
        tape = [0] * tape_size
        head = tape_size // 2
        state = 0
        steps = 0

        while steps < max_steps:
            if state == HALT:
                break

            symbol = tape[head]
            key = (state, symbol)

            if key not in self.transitions:
                # No transition defined: halt
                break

            t = self.transitions[key]
            tape[head] = t.write_symbol
            head += t.move
            state = t.next_state

            # Bounds check
            if head < 0 or head >= tape_size:
                break

            steps += 1
        else:
            # Did not halt within max_steps
            return None

        # Extract the non-trivial part of the tape
        return self._extract_output(tape)

    def _extract_output(self, tape: List[int]) -> str:
        """Extract the meaningful portion of the tape as a binary string."""
        # Find first and last non-zero positions
        first_nonzero = None
        last_nonzero = None
        for i, v in enumerate(tape):
            if v != 0:
                if first_nonzero is None:
                    first_nonzero = i
                last_nonzero = i

        if first_nonzero is None:
            # All zeros - return empty string (the TM produced nothing)
            return "0"

        output = tape[first_nonzero : last_nonzero + 1]
        return "".join(str(s) for s in output)


def enumerate_tms(
    max_states: int, max_symbols: int
) -> Generator[SimpleTM, None, None]:
    """Enumerate all possible Turing machines with given bounds.

    For each (num_states, num_symbols) pair, generates all possible
    transition tables.

    Args:
        max_states: Maximum number of states (e.g., 2 or 3).
        max_symbols: Maximum number of symbols (e.g., 2).

    Yields:
        SimpleTM instances.
    """
    for num_states in range(1, max_states + 1):
        for num_symbols in range(2, max_symbols + 1):
            yield from _enumerate_tms_for(num_states, num_symbols)


def _enumerate_tms_for(
    num_states: int, num_symbols: int
) -> Generator[SimpleTM, None, None]:
    """Enumerate all TMs with exactly num_states states and num_symbols symbols."""
    # Each cell in the transition table is (write_symbol, direction, next_state)
    # Possible values for each:
    write_options = list(range(num_symbols))
    direction_options = [LEFT, RIGHT]
    # next_state can be any state or HALT
    state_options = list(range(num_states)) + [HALT]

    # All possible single transitions
    single_transitions = list(
        itertools.product(write_options, direction_options, state_options)
    )

    # All cells in the transition table: (state, symbol) pairs
    cells = list(itertools.product(range(num_states), range(num_symbols)))
    num_cells = len(cells)

    # Enumerate all possible transition tables
    for combo in itertools.product(single_transitions, repeat=num_cells):
        tm = SimpleTM(num_states=num_states, num_symbols=num_symbols)
        for cell, (write_sym, direction, next_state) in zip(cells, combo):
            tm.transitions[cell] = Transition(
                write_symbol=write_sym,
                move=direction,
                next_state=next_state,
            )
        yield tm


def count_tms(max_states: int, max_symbols: int) -> int:
    """Count the total number of TMs that would be enumerated.

    For n states, k symbols: each of the n*k cells has k * 2 * (n+1) options.
    Total = (2k(n+1))^(nk)
    """
    total = 0
    for n in range(1, max_states + 1):
        for k in range(2, max_symbols + 1):
            options_per_cell = k * 2 * (n + 1)
            num_cells = n * k
            total += options_per_cell ** num_cells
    return total
