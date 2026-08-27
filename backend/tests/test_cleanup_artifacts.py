from backend.scripts.maintenance.cleanup_artifacts import ArtifactCleaner


def write(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def test_cleanup_dry_run_preserves_candidates(tmp_path):
    backend = tmp_path / "backend"
    results = tmp_path / "results"
    archive = backend / "archive" / "project_versions_2026-01-01"
    source = backend / "notebooks" / "original" / "model.ipynb"
    output = results / "notebook_runs" / "model.ipynb"
    write(archive / "artifact.csv", b"old")
    write(source, b"same")
    write(output, b"same")

    candidates = ArtifactCleaner(backend, results).run(False)

    assert {candidate.path for candidate in candidates} == {archive, output}
    assert archive.exists()
    assert output.exists()


def test_cleanup_apply_removes_only_safe_candidates(tmp_path):
    backend = tmp_path / "backend"
    results = tmp_path / "results"
    archive = backend / "archive" / "project_versions_2026-01-01"
    source = backend / "notebooks" / "original" / "model.ipynb"
    duplicate = results / "notebook_runs" / "model.ipynb"
    distinct = results / "notebook_runs" / "distinct.ipynb"
    write(archive / "artifact.csv", b"old")
    write(source, b"same")
    write(duplicate, b"same")
    write(distinct, b"different")

    ArtifactCleaner(backend, results).run(True)

    assert not archive.exists()
    assert not duplicate.exists()
    assert distinct.exists()
