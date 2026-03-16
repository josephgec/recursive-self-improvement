"""Factory for creating REPL backends."""

from typing import Optional

import yaml

from src.interface.base import SandboxREPL
from src.safety.policy import SafetyPolicy
from src.backends.local import LocalREPL
from src.backends.docker import DockerREPL
from src.backends.modal_repl import ModalREPL


class REPLFactory:
    """Factory for creating REPL backend instances.

    Supports creation by backend name or from a YAML configuration file.
    """

    BACKENDS = {
        "local": LocalREPL,
        "docker": DockerREPL,
        "modal": ModalREPL,
    }

    @classmethod
    def create(
        cls,
        backend: str = "local",
        policy: Optional[SafetyPolicy] = None,
    ) -> SandboxREPL:
        """Create a REPL backend instance.

        Args:
            backend: Backend name ('local', 'docker', 'modal').
            policy: Safety policy to use.

        Returns:
            A SandboxREPL instance.

        Raises:
            ValueError: If the backend name is unknown.
        """
        if backend not in cls.BACKENDS:
            raise ValueError(
                f"Unknown backend '{backend}'. "
                f"Available: {list(cls.BACKENDS.keys())}"
            )

        backend_class = cls.BACKENDS[backend]
        policy = policy or SafetyPolicy()
        return backend_class(policy=policy)

    @classmethod
    def from_config(cls, config_path: str) -> SandboxREPL:
        """Create a REPL from a YAML configuration file.

        Args:
            config_path: Path to the YAML configuration file.

        Returns:
            A configured SandboxREPL instance.
        """
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        backend = config.get("backend", "local")

        # Load safety policy
        policy_path = config.get("safety", {}).get("policy")
        if policy_path:
            try:
                policy = SafetyPolicy.from_yaml(policy_path)
            except FileNotFoundError:
                policy = SafetyPolicy()
        else:
            policy = SafetyPolicy()

        # Apply execution overrides
        execution = config.get("execution", {})
        if "timeout_seconds" in execution:
            policy.timeout_seconds = execution["timeout_seconds"]
        if "max_memory_mb" in execution:
            policy.max_memory_mb = execution["max_memory_mb"]
        if "max_output_chars" in execution:
            policy.max_output_chars = execution["max_output_chars"]
        if "max_recursion_depth" in execution:
            policy.max_spawn_depth = execution["max_recursion_depth"]

        return cls.create(backend=backend, policy=policy)

    @classmethod
    def auto_detect(cls, policy: Optional[SafetyPolicy] = None) -> SandboxREPL:
        """Auto-detect the best available backend.

        Tries Docker first, then falls back to local.

        Args:
            policy: Safety policy to use.

        Returns:
            A SandboxREPL instance.
        """
        policy = policy or SafetyPolicy()

        # Try Docker
        try:
            repl = DockerREPL(policy=policy, use_fallback=False)
            if repl.is_docker:
                return repl
        except (RuntimeError, Exception):
            pass

        # Fall back to local
        return LocalREPL(policy=policy)
