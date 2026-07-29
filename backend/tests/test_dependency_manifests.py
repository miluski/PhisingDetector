from pathlib import Path


BACKEND_ROOT = Path(__file__).resolve().parents[1]
LOCK_FILES = ("requirements.txt", "requirements-dev.txt")
PYTHON_IMAGE = "python:3.13.14-slim-bookworm@sha256:"


def _requirements(filename):
    content = (BACKEND_ROOT / filename).read_text()
    return {line for line in content.splitlines() if line}


def _locked_versions(filename):
    requirements = _requirements(filename)
    return {requirement.split(" ; ", 1)[0] for requirement in requirements}


def test_lock_files_contain_only_exact_versions():
    for filename in LOCK_FILES:
        requirements = _requirements(filename)
        assert requirements
        assert all("==" in requirement for requirement in requirements)


def test_development_lock_contains_runtime_lock():
    runtime = _locked_versions("requirements.txt")
    development = _locked_versions("requirements-dev.txt")
    assert runtime <= development


def test_docker_images_use_pinned_python_313_runtime_lock():
    for filename in ("Dockerfile", "Dockerfile.sandbox"):
        content = (BACKEND_ROOT / filename).read_text()
        assert content.startswith(f"FROM {PYTHON_IMAGE}")
        assert "requirements.txt" in content
        assert "requirements-dev.txt" not in content
