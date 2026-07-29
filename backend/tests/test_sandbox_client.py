"""Tests for the sandbox port and orchestrator delegation."""

from services.container_orchestrator import ContainerOrchestrator


class FakeSandboxClient:
    """Minimal sandbox fake used to verify orchestration boundaries."""

    def __init__(self):
        self.calls = []

    def ensure_image(self):
        """Record image preparation without requiring Docker."""
        self.calls.append(("ensure",))

    def create_container(self, urls, models, mode, dataset):
        """Record creation and return a fake identifier."""
        self.calls.append(("create", urls, models, mode, dataset))
        return "fake-1"

    def wait_for_result(self, container_id, timeout=300):
        """Return a deterministic fake result."""
        self.calls.append(("wait", container_id, timeout))
        return {"success": True, "results": []}

    def cleanup_container(self, container_id):
        """Record one cleanup call."""
        self.calls.append(("cleanup", container_id))

    def cleanup_all(self):
        """Record global cleanup."""
        self.calls.append(("cleanup_all",))


def test_orchestrator_delegates_to_injected_client():
    """The application orchestrator does not require Docker in tests."""
    fake = FakeSandboxClient()
    orchestrator = ContainerOrchestrator(fake)
    assert orchestrator.create_container(["https://example.com"], ["svm"]) == "fake-1"
    assert orchestrator.wait_for_result("fake-1")["success"] is True
    orchestrator.cleanup_container("fake-1")
    assert [call[0] for call in fake.calls] == ["create", "wait", "cleanup"]
