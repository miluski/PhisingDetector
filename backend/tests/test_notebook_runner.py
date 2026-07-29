"""Tests for durable isolated notebook execution."""

from __future__ import annotations

import nbformat
import pytest

from scripts import run_model_notebooks_parallel as runner


def test_failed_run_persists_completed_cell_outputs(tmp_path, monkeypatch) -> None:
    """Keep successful earlier outputs when a later notebook cell fails."""
    notebook_path = tmp_path / "experiment.ipynb"
    output_dir = tmp_path / "runs"
    notebook = nbformat.v4.new_notebook(
        cells=[nbformat.v4.new_code_cell("print('completed')")]
    )
    nbformat.write(notebook, notebook_path)
    monkeypatch.setattr(runner, "_execute_client", _partially_execute_and_fail)

    with pytest.raises(RuntimeError, match="later cell failed"):
        runner.execute_one(notebook_path, output_dir, 60, True)

    _assert_completed_output(notebook_path)
    _assert_completed_output(output_dir / notebook_path.name)


def _partially_execute_and_fail(notebook, _timeout) -> None:
    """Simulate one completed cell followed by a later execution failure."""
    cell = notebook.cells[0]
    cell.execution_count = 1
    cell.outputs = [nbformat.v4.new_output("stream", name="stdout", text="completed\n")]
    raise RuntimeError("later cell failed")


def _assert_completed_output(path) -> None:
    """Assert that one persisted notebook contains the completed output."""
    notebook = nbformat.read(path, as_version=4)
    cell = notebook.cells[0]
    assert cell.execution_count == 1
    assert cell.outputs[0].text == "completed\n"
