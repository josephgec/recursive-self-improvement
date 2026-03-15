#!/usr/bin/env python3
"""Build the CTM lookup table by enumerating small Turing machines."""

import os
import sys
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.bdm.ctm_table import CTMTable


def main():
    print("Building CTM table...")
    print("  max_states=2, max_symbols=2, max_steps=50, block_size=12")

    table = CTMTable()
    start = time.time()
    table.build(max_states=2, max_symbols=2, max_steps=50, block_size=12)
    elapsed = time.time() - start

    output_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "ctm_tables", "ctm_2state_2symbol.json"
    )
    table.save(output_path)

    print(f"  Table size: {table.size} entries")
    print(f"  Time: {elapsed:.2f}s")
    print(f"  Saved to: {output_path}")

    # Verify
    loaded = CTMTable()
    loaded.load(output_path)
    print(f"  Verified load: {loaded.size} entries")

    # Sample lookups
    for s in ["0", "1", "00", "01", "0000", "0101", "1111"]:
        k = loaded.lookup(s)
        print(f"  K({s}) = {k:.2f}")


if __name__ == "__main__":
    main()
