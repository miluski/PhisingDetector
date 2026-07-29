"""Run model fitting in a process whose resources are fully reclaimable."""

from __future__ import annotations

import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from joblib.externals import cloudpickle

from .resource_monitor import ResourceMonitor


@dataclass
class IsolatedFitResult:
    """Contain a restored model and measurements from its exited worker."""

    model: Any
    history: SimpleNamespace | None
    training_time: float
    monitor: ResourceMonitor
    stats: dict


class IsolatedModelTrainer:
    """Coordinate fitting, monitoring, process exit and model restoration."""

    def __init__(self, interval: float, baseline_seconds: float):
        """Configure measurement frequency and idle baseline duration."""
        self.interval = interval
        self.baseline_seconds = baseline_seconds

    def fit(
        self, model, features, target, fit_kwargs=None,
    ) -> IsolatedFitResult:
        """Fit an estimator in a disposable worker and restore its artifact."""
        with tempfile.TemporaryDirectory(prefix="phishing-fit-") as directory:
            workspace = Path(directory)
            self._write_request(
                workspace, model, features, target, fit_kwargs or {}
            )
            process = self._start_worker(workspace)
            try:
                return self._measure(process, workspace)
            finally:
                self._terminate_if_running(process)

    @staticmethod
    def _write_request(
        workspace, model, features, target, fit_kwargs,
    ) -> None:
        """Serialize all worker inputs before resource measurement starts."""
        payload = {
            "model": model, "features": features,
            "target": target, "fit_kwargs": fit_kwargs,
        }
        with (workspace / "request.pkl").open("wb") as stream:
            cloudpickle.dump(payload, stream)

    @staticmethod
    def _start_worker(workspace: Path) -> subprocess.Popen:
        """Start the disposable Python worker."""
        worker = Path(__file__).with_name("isolated_training_worker.py")
        return subprocess.Popen(
            [sys.executable, str(worker), str(workspace)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )

    def _measure(self, process, workspace) -> IsolatedFitResult:
        """Measure only the worker from idle baseline through process exit."""
        self._wait_for_marker(process, workspace / "ready")
        monitor = ResourceMonitor(
            interval=self.interval, process_id=process.pid
        )
        monitor.start()
        time.sleep(self.baseline_seconds)
        monitor.mark_training_start()
        (workspace / "start").touch()
        self._wait_for_marker(process, workspace / "training_done")
        monitor.mark_training_end()
        (workspace / "end_ack").touch()
        process.wait()
        monitor.wait_for_post_training_idle()
        monitor.stop()
        return self._build_result(workspace, monitor)

    @staticmethod
    def _wait_for_marker(process, marker: Path) -> None:
        """Wait for a worker marker or raise when the worker exits early."""
        while not marker.exists():
            if process.poll() is not None:
                raise RuntimeError("Isolated training worker exited early")
            time.sleep(0.01)

    @staticmethod
    def _build_result(workspace, monitor) -> IsolatedFitResult:
        """Restore the model only after its training process has exited."""
        error_path = workspace / "error.txt"
        if error_path.exists():
            raise RuntimeError(error_path.read_text(encoding="utf-8"))
        with (workspace / "response.pkl").open("rb") as stream:
            payload = cloudpickle.load(stream)
        history = payload["history"]
        return IsolatedFitResult(
            model=payload["model"],
            history=SimpleNamespace(history=history) if history else None,
            training_time=float(payload["duration"]),
            monitor=monitor,
            stats=monitor.get_stats(),
        )

    @staticmethod
    def _terminate_if_running(process) -> None:
        """Prevent an orphan worker when notebook execution is interrupted."""
        if process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def fit_model_isolated(
    model, features, target, *, fit_kwargs=None,
    interval: float = 0.05, baseline_seconds: float = 0.5,
) -> IsolatedFitResult:
    """Fit a model in a process that releases all resources when it exits."""
    trainer = IsolatedModelTrainer(interval, baseline_seconds)
    return trainer.fit(model, features, target, fit_kwargs)
