import argparse
import subprocess

import pytest

from scripts import setup


def create_required_artifacts(backend_root):
    paths = [*setup.required_model_paths(backend_root), *setup.required_dataset_paths(backend_root)]
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"artifact")
    return paths


def test_required_artifact_manifest_contains_expected_matrix(tmp_path):
    backend = tmp_path / "backend"

    assert len(setup.required_model_paths(backend)) == 32
    assert len(setup.required_dataset_paths(backend)) == 4


def test_artifact_validation_lists_missing_paths(tmp_path):
    backend = tmp_path / "backend"

    with pytest.raises(setup.SetupError, match="Missing required project artifacts"):
        setup.validate_artifacts(backend)


def test_artifact_validation_accepts_complete_bundle(tmp_path):
    backend = tmp_path / "backend"
    create_required_artifacts(backend)

    setup.validate_artifacts(backend)


def test_certificate_preparation_is_idempotent(tmp_path, monkeypatch):
    monkeypatch.setattr(setup, "CERTIFICATE_ROOT", tmp_path)
    (tmp_path / "cert.pem").write_bytes(b"cert")
    (tmp_path / "key.pem").write_bytes(b"key")

    setup.prepare_certificates(lambda command, check, capture: pytest.fail(str(command)))


def test_check_only_stops_before_preparation(monkeypatch):
    calls = []
    monkeypatch.setattr(setup, "validate_artifacts", lambda: None)
    monkeypatch.setattr(setup, "prepare_certificates", lambda runner: calls.append("prepare"))
    runner = lambda command, check, capture: subprocess.CompletedProcess(command, 0, "ok", "")

    setup.execute(argparse.Namespace(check_only=True, prepare_only=False), runner)

    assert calls == []


def test_start_failure_includes_compose_logs():
    def runner(command, check, capture):
        if "logs" in command:
            return subprocess.CompletedProcess(command, 0, "backend failed", "")
        return subprocess.CompletedProcess(command, 1, "", "build error")

    with pytest.raises(setup.SetupError, match="backend failed"):
        setup.start_services(runner)


def test_health_failure_includes_compose_logs():
    def runner(command, check, capture):
        return subprocess.CompletedProcess(command, 0, "health logs", "")

    def waiter():
        raise setup.SetupError("health timeout")

    with pytest.raises(setup.SetupError, match="health logs"):
        setup.wait_with_logs(waiter, runner)
