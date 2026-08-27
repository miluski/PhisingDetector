import subprocess
from pathlib import Path

from backend.scripts.maintenance.audit_duplicates import PROJECT_ROOT, unexpected_duplicates


def run_git(root: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=root, check=True, capture_output=True)


def test_duplicate_audit_reports_unlisted_group(tmp_path):
    run_git(tmp_path, "init")
    (tmp_path / "first.txt").write_text("same")
    (tmp_path / "second.txt").write_text("same")
    run_git(tmp_path, "add", "first.txt", "second.txt")

    groups = unexpected_duplicates(tmp_path)

    assert groups == [frozenset({"first.txt", "second.txt"})]


def test_repository_has_no_unexpected_exact_duplicates():
    assert unexpected_duplicates(PROJECT_ROOT) == []
