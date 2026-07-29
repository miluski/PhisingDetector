"""Execute one serialized model fit inside a disposable process."""

from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path

from joblib.externals import cloudpickle


def _load(path: Path):
    """Load a cloudpickle payload from disk."""
    with path.open("rb") as stream:
        return cloudpickle.load(stream)


def _save(path: Path, payload) -> None:
    """Persist a cloudpickle payload to disk."""
    with path.open("wb") as stream:
        cloudpickle.dump(payload, stream)


def _wait(path: Path) -> None:
    """Wait for a parent-process synchronization marker."""
    while not path.exists():
        time.sleep(0.01)


def _history_payload(fit_result):
    """Extract serializable Keras history without retaining its model."""
    values = getattr(fit_result, "history", None)
    return dict(values) if values is not None else None


def _fit(workspace: Path):
    """Fit the requested estimator after the parent starts monitoring."""
    request = _load(workspace / "request.pkl")
    (workspace / "ready").touch()
    _wait(workspace / "start")
    started = time.perf_counter()
    fit_result = request["model"].fit(
        request["features"], request["target"], **request["fit_kwargs"]
    )
    duration = time.perf_counter() - started
    history = _history_payload(fit_result)
    (workspace / "training_done").touch()
    _wait(workspace / "end_ack")
    return request["model"], history, duration


def _run(workspace: Path) -> None:
    """Write either a fitted model response or a complete traceback."""
    try:
        model, history, duration = _fit(workspace)
        _save(
            workspace / "response.pkl",
            {"model": model, "history": history, "duration": duration},
        )
    except Exception:
        (workspace / "error.txt").write_text(
            traceback.format_exc(), encoding="utf-8"
        )
        (workspace / "ready").touch()
        (workspace / "training_done").touch()
        _wait(workspace / "end_ack")


def main() -> None:
    """Run the isolated worker for the workspace passed on the command line."""
    if len(sys.argv) != 2:
        raise SystemExit("Usage: isolated_training_worker.py WORKSPACE")
    _run(Path(sys.argv[1]))


if __name__ == "__main__":
    main()
