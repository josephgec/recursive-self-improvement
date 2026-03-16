#!/usr/bin/env python3
"""Build the Docker sandbox image."""

import subprocess
import sys


def main():
    """Build the sandbox Docker image."""
    print("Building rlm-sandbox Docker image...")
    try:
        result = subprocess.run(
            ["docker", "build", "-f", "docker/Dockerfile.sandbox", "-t", "rlm-sandbox:latest", "."],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            print("Build successful!")
        else:
            print(f"Build failed:\n{result.stderr}")
            sys.exit(1)
    except FileNotFoundError:
        print("Docker not found. Please install Docker first.")
        sys.exit(1)


if __name__ == "__main__":
    main()
