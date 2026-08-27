from pathlib import Path


BACKEND_ROOT = Path(__file__).resolve().parents[1]
MANIFESTS = ("requirements.txt", "requirements-dev.txt")
PYTHON_IMAGE = "python:3.13.14-slim-bookworm@sha256:"


def requirements(filename: str) -> set[str]:
    content = (BACKEND_ROOT / filename).read_text()
    return {line for line in content.splitlines() if line}


def package_name(requirement: str) -> str:
    return requirement.split("==", 1)[0].lower().replace("_", "-")


def test_only_runtime_and_development_manifests_exist():
    manifests = {path.name for path in BACKEND_ROOT.glob("requirements*")}
    assert manifests == set(MANIFESTS)


def test_manifests_contain_only_exact_versions():
    for filename in MANIFESTS:
        entries = requirements(filename)
        assert entries
        assert all("==" in entry for entry in entries)


def test_development_manifest_does_not_repeat_runtime_packages():
    runtime = {package_name(entry) for entry in requirements("requirements.txt")}
    development = {package_name(entry) for entry in requirements("requirements-dev.txt")}
    assert runtime.isdisjoint(development)


def test_dockerfile_uses_pinned_python_runtime_manifest():
    content = (BACKEND_ROOT / "Dockerfile").read_text()
    assert content.startswith(f"FROM {PYTHON_IMAGE}")
    assert "COPY requirements.txt" in content
    assert "requirements-dev.txt" not in content
    assert "AS sandbox" in content
    assert "AS app" in content
    assert "AS certificates" in content
