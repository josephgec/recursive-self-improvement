"""Modal utility functions (stub implementation)."""


def is_modal_available() -> bool:
    """Check if Modal is available and configured."""
    try:
        import modal
        return True
    except ImportError:
        return False


def create_modal_sandbox() -> None:
    """Create a Modal sandbox function."""
    raise NotImplementedError("Modal sandbox not yet implemented")
