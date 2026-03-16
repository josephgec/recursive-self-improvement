"""Output size limiting for sandboxed code execution."""


class OutputLimiter:
    """Limits output size from sandboxed code execution.

    Truncates output that exceeds the maximum character limit and appends
    a truncation notice.
    """

    TRUNCATION_MESSAGE = "\n... [OUTPUT TRUNCATED: exceeded {limit} character limit, showing first {limit} chars of {total}] ..."

    def __init__(self, max_chars: int = 100000):
        self.max_chars = max_chars

    def truncate(self, output: str, max_chars: int = None) -> str:
        """Truncate output if it exceeds the character limit.

        Args:
            output: The output string to potentially truncate.
            max_chars: Override the default character limit.

        Returns:
            The original output if within limits, or truncated output
            with a notification message appended.
        """
        limit = max_chars if max_chars is not None else self.max_chars

        if len(output) <= limit:
            return output

        truncated = output[:limit]
        truncated += self.TRUNCATION_MESSAGE.format(
            limit=limit,
            total=len(output),
        )
        return truncated

    def check(self, output: str, max_chars: int = None) -> bool:
        """Check if output is within limits.

        Args:
            output: The output string to check.
            max_chars: Override the default character limit.

        Returns:
            True if the output is within limits.
        """
        limit = max_chars if max_chars is not None else self.max_chars
        return len(output) <= limit
