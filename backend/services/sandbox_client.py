"""Ports used by the sandbox orchestration layer."""

from __future__ import annotations

from typing import Any, Protocol


class SandboxClient(Protocol):
    """Port for a sandbox implementation used by the application service."""

    def ensure_image(self) -> None:
        """Ensure the configured sandbox image exists."""

    def create_container(self, urls: list[str], models: list[str], mode: str, dataset: str) -> str:
        """Start a sandbox analysis and return its identifier."""

    def wait_for_result(self, container_id: str, timeout: int = 300) -> dict[str, Any]:
        """Wait for and return a sandbox result."""

    def cleanup_container(self, container_id: str) -> None:
        """Stop and remove one sandbox."""

    def cleanup_all(self) -> None:
        """Stop and remove all sandboxes owned by the client."""
