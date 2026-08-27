"""Execute model notebooks in independent Jupyter kernel processes."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[2]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from config.project_paths import RESULTS_ROOT

DEFAULT_NOTEBOOKS = sorted(
    [
        *(
            BACKEND_ROOT / "notebooks" / "original"
        ).glob("[0-7]_*.ipynb"),
        *(
            BACKEND_ROOT / "notebooks" / "url_corpus"
        ).glob("[0-7]_*.ipynb"),
    ],
    key=lambda path: (
        path.parent.name,
        int(path.name.split("_", 1)[0]),
        path.name,
    ),
)


def execute_one(
    notebook: Path, output_dir: Path, timeout: int, write_back: bool,
) -> int:
    """Execute one notebook and persist completed cells even after failure."""
    import nbformat
    notebook_data = nbformat.read(notebook, as_version=4)
    try:
        _execute_client(notebook_data, timeout)
    finally:
        _write_notebook(notebook_data, notebook, output_dir, write_back)
    return 0


def _execute_client(notebook_data, timeout):
    """Execute notebook cells without allowing failed cells."""
    from nbclient import NotebookClient
    client = NotebookClient(
        notebook_data,
        timeout=timeout,
        kernel_name="python3",
        resources={"metadata": {"path": str(BACKEND_ROOT)}},
        allow_errors=False,
    )
    client.execute()


def _write_notebook(notebook_data, notebook, output_dir, write_back):
    """Persist executed notebook copy and optionally update its source."""
    import nbformat
    output_dir.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook_data, output_dir / notebook.name)
    if write_back:
        nbformat.write(notebook_data, notebook)


def parse_args() -> argparse.Namespace:
    """Parse notebook runner options."""
    parser = argparse.ArgumentParser(description="Uruchamia notebooki modeli w niezależnych procesach.")
    _add_user_arguments(parser)
    _add_worker_arguments(parser)
    return parser.parse_args()


def _add_user_arguments(parser):
    """Register options used by the public runner command."""
    parser.add_argument("notebooks", nargs="*", type=Path, help="Notebooki; domyślnie original i url_corpus.")
    parser.add_argument("--timeout", type=int, default=7200, help="Maksymalny czas jednej komórki.")
    parser.add_argument(
        "--output-dir", type=Path,
        default=RESULTS_ROOT / "notebook_runs",
    )
    parser.add_argument("--parallel", action="store_true", help="Uruchom procesy równolegle.")
    parser.add_argument("--no-write-back", action="store_true", help="Nie modyfikuj źródeł.")


def _add_worker_arguments(parser):
    """Register hidden options used by isolated worker processes."""
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--worker-notebook", type=Path, help=argparse.SUPPRESS)


def worker_command(
    notebook: Path, output_dir: Path, timeout: int, write_back: bool,
) -> list[str]:
    """Build the command used to launch an isolated worker."""
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--worker-notebook",
        str(notebook),
        "--timeout",
        str(timeout),
        "--output-dir",
        str(output_dir),
    ]
    if not write_back:
        command.append("--no-write-back")
    return command


def run_parallel(
    notebooks: list[Path], output_dir: Path, timeout: int, write_back: bool,
) -> int:
    """Run workers concurrently and return the first non-zero status."""
    output_dir.mkdir(parents=True, exist_ok=True)
    processes = [_spawn_worker(path, output_dir, timeout, write_back) for path in notebooks]
    return _wait_workers(processes)


def _spawn_worker(notebook, output_dir, timeout, write_back):
    """Start one worker and return its process bookkeeping tuple."""
    notebook = notebook.resolve()
    log_file = (output_dir / f"{notebook.stem}.log").open("w", encoding="utf-8")
    process = subprocess.Popen(worker_command(notebook, output_dir, timeout, write_back), cwd=BACKEND_ROOT,
                               stdout=log_file, stderr=subprocess.STDOUT, env=os.environ.copy())
    print(f"Uruchomiono niezależny proces: {notebook.name} (PID {process.pid})")
    return notebook, process, log_file


def _wait_workers(processes):
    """Wait for workers and report their statuses."""
    status = 0
    for notebook, process, log_file in processes:
        result = process.wait()
        log_file.close()
        status = result or status
        print(f"{'BŁĄD' if result else 'OK'}: {notebook.name}" + (f" ({result})" if result else ""))
    return status


def run_sequential(
    notebooks: list[Path], output_dir: Path, timeout: int, write_back: bool,
) -> int:
    """Run workers one at a time to isolate CPU/RAM measurements."""
    output_dir.mkdir(parents=True, exist_ok=True)
    exit_code = 0
    for notebook in notebooks:
        result = _run_worker(notebook, output_dir, timeout, write_back)
        exit_code = result or exit_code
    return exit_code


def _run_worker(notebook, output_dir, timeout, write_back):
    """Execute one worker synchronously and close its log."""
    notebook = notebook.resolve()
    log_path = output_dir / f"{notebook.stem}.log"
    print(f"Uruchomiono izolowany proces: {notebook.name}")
    with log_path.open("w", encoding="utf-8") as log_file:
        result = subprocess.run(worker_command(notebook, output_dir, timeout, write_back), cwd=BACKEND_ROOT,
                                stdout=log_file, stderr=subprocess.STDOUT, env=os.environ.copy(), check=False)
    print(f"{'BŁĄD' if result.returncode else 'OK'}: {notebook.name}")
    return result.returncode


def main() -> int:
    """Coordinate worker mode and public sequential/parallel modes."""
    args = parse_args()
    if args.worker:
        if args.worker_notebook is None:
            raise ValueError("Brak notebooka procesu roboczego")
        return execute_one(args.worker_notebook, args.output_dir, args.timeout, not args.no_write_back)

    notebooks = [path.resolve() for path in args.notebooks] or DEFAULT_NOTEBOOKS
    if not notebooks:
        raise FileNotFoundError("Nie znaleziono notebooków modeli URL-corpus")
    if args.parallel:
        return run_parallel(notebooks, args.output_dir, args.timeout, not args.no_write_back)
    return run_sequential(notebooks, args.output_dir, args.timeout, not args.no_write_back)


if __name__ == "__main__":
    raise SystemExit(main())
