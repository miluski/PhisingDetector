from threading import Lock
from types import SimpleNamespace

from services.container_orchestrator import DockerSandboxClient, ContainerOrchestrator


class _WarmContainer:
    def __init__(self):
        self.id = "warm-id"
        self.short_id = "warm"
        self.name = "phishing-sandbox-warm"
        self.status = "running"
        self.exec_calls = []
        self.stop_calls = 0
        self.remove_calls = 0

    def reload(self):
        return None

    def exec_run(self, command, environment, workdir):
        self.exec_calls.append((command, environment, workdir))
        return SimpleNamespace(
            exit_code=0,
            output=b'{"success": true, "results": []}\n',
        )

    def stop(self, timeout):
        self.stop_calls += 1

    def remove(self, force, v):
        self.remove_calls += 1


class _WarmContainers:
    def __init__(self):
        self.container = _WarmContainer()
        self.run_calls = 0

    def run(self, *args, **kwargs):
        self.run_calls += 1
        return self.container


def _make_warm_client():
    docker_client = SimpleNamespace(containers=_WarmContainers())
    client = DockerSandboxClient.__new__(DockerSandboxClient)
    client.client = docker_client
    client.image_name = "sandbox"
    client.host_backend_path = "/app"
    client.containers = {}
    client._request_environment = {}
    client._warm_container = None
    client._warm_lock = Lock()
    client._execution_lock = Lock()
    client.ensure_image_exists = lambda: None
    return client, docker_client


def test_container_log_parser_reads_last_json_line_without_trailing_newline():
    parser = ContainerOrchestrator.__new__(ContainerOrchestrator)
    result = parser._parse_container_logs('INFO startup\n{"success": true, "results": []}')

    assert result == {"success": True, "results": []}


def test_docker_client_reuses_warm_container_between_requests():
    client, docker_client = _make_warm_client()

    first = client.create_container(["https://one.example"], ["svm"])
    second = client.create_container(["https://two.example"], ["svm"])

    assert first != second
    assert docker_client.containers.run_calls == 1
    assert client.wait_for_result(first)["success"] is True
    assert client.wait_for_result(second)["success"] is True
    assert len(docker_client.containers.container.exec_calls) == 2

    client.cleanup_container(first)
    client.cleanup_container(second)
    assert docker_client.containers.container.stop_calls == 0
    client.cleanup_all()
    assert docker_client.containers.container.stop_calls == 1
    assert docker_client.containers.container.remove_calls == 1
