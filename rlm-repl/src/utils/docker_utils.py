"""Docker utility functions (stub implementation)."""


def is_docker_available() -> bool:
    """Check if Docker daemon is available."""
    try:
        import docker
        client = docker.from_env()
        client.ping()
        return True
    except Exception:
        return False


def build_sandbox_image(tag: str = "rlm-sandbox:latest") -> bool:
    """Build the sandbox Docker image."""
    raise NotImplementedError("Docker image building not yet implemented")


def get_container_stats(container_id: str) -> dict:
    """Get stats for a running container."""
    raise NotImplementedError("Container stats not yet implemented")
