"""Encoding utilities for converting data to binary representations."""

from __future__ import annotations

from typing import List, Union


def data_to_binary(data: Union[str, list, int, float]) -> str:
    """Convert various data types to binary string representation.

    Args:
        data: Input data (string, list of ints/floats, int, or float).

    Returns:
        Binary string (e.g., '01101001').
    """
    if isinstance(data, str):
        return _string_to_binary(data)
    elif isinstance(data, (list, tuple)):
        return _list_to_binary(data)
    elif isinstance(data, int):
        return encode_number(data)
    elif isinstance(data, float):
        return encode_number(data)
    else:
        return _string_to_binary(str(data))


def binary_to_data(binary: str) -> str:
    """Convert binary string back to character string.

    Args:
        binary: Binary string of 0s and 1s.

    Returns:
        Decoded string.
    """
    chars = []
    for i in range(0, len(binary), 8):
        byte = binary[i : i + 8]
        if len(byte) == 8:
            chars.append(chr(int(byte, 2)))
    return "".join(chars)


def encode_tokens(tokens: List[str]) -> str:
    """Encode a list of tokens to binary.

    Each token is converted to its binary representation and concatenated.

    Args:
        tokens: List of string tokens.

    Returns:
        Concatenated binary string.
    """
    parts = []
    for token in tokens:
        parts.append(_string_to_binary(token))
    return "".join(parts)


def encode_number(n: Union[int, float]) -> str:
    """Encode a number to binary string.

    For integers: standard binary representation (minimum 8 bits).
    For floats: encode as string then to binary.

    Args:
        n: Number to encode.

    Returns:
        Binary string representation.
    """
    if isinstance(n, float):
        return _string_to_binary(f"{n:.6g}")
    if n < 0:
        # Two's complement for negative: prefix with sign bit
        magnitude = encode_number(-n)
        return "1" + magnitude
    if n == 0:
        return "00000000"
    bits = bin(n)[2:]
    # Pad to at least 8 bits
    while len(bits) < 8:
        bits = "0" + bits
    return bits


def _string_to_binary(s: str) -> str:
    """Convert a string to binary (8-bit per character)."""
    return "".join(format(ord(c), "08b") for c in s)


def _list_to_binary(data: list) -> str:
    """Convert a list of values to binary."""
    parts = []
    for item in data:
        if isinstance(item, int):
            parts.append(encode_number(item))
        elif isinstance(item, float):
            parts.append(encode_number(item))
        elif isinstance(item, str):
            parts.append(_string_to_binary(item))
        else:
            parts.append(_string_to_binary(str(item)))
    return "".join(parts)
