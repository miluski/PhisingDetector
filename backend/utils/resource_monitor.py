import csv
import threading
import time
from typing import Optional

import numpy as np
import psutil
from matplotlib.ticker import FuncFormatter


MAX_TWO_DECIMALS = FuncFormatter(
    lambda value, _: f"{value:.2f}".rstrip("0").rstrip(".")
)


def _data_limits(values, padding_ratio: float = 0.08):
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0, 1.0

    low = float(finite.min())
    high = float(finite.max())
    span = high - low
    if span == 0:
        span = max(abs(high), 1.0) * 0.1
    padding = span * padding_ratio
    return max(0.0, low - padding), high + padding


class ProcessResourceSampler:
    """Sample the current process and all descendants as one workload."""

    def __init__(self, process: psutil.Process, num_cores: int):
        """Initialize a sampler for one root process."""
        self.process = process
        self.num_cores = num_cores
        self.cpu_times: dict[int, float] = {}
        self.last_sample_time: float | None = None

    def processes(self) -> list[psutil.Process]:
        """Return unique live root and child processes."""
        candidates = [self.process]
        try:
            candidates.extend(self.process.children(recursive=True))
        except (psutil.NoSuchProcess, psutil.AccessDenied, PermissionError):
            return [self.process]
        return list({item.pid: item for item in candidates}.values())

    def prime(self) -> None:
        """Store cumulative CPU counters without creating a sample."""
        counters: dict[int, float] = {}
        for process in self.processes():
            try:
                counters[process.pid] = self._cpu_time(process)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        self.cpu_times = counters
        self.last_sample_time = time.monotonic()

    def sample(self) -> tuple[float, float]:
        """Return interval CPU utilization and resident RAM in gigabytes."""
        now = time.monotonic()
        elapsed = max(now - (self.last_sample_time or now), 1e-9)
        current_times: dict[int, float] = {}
        cpu_seconds = 0.0
        ram_bytes = 0
        for process in self.processes():
            try:
                current = self._cpu_time(process)
                previous = self.cpu_times.get(process.pid, current)
                cpu_seconds += max(current - previous, 0.0)
                current_times[process.pid] = current
                ram_bytes += process.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        self.cpu_times, self.last_sample_time = current_times, now
        cpu_percent = cpu_seconds / elapsed * 100.0 / self.num_cores
        return cpu_percent, ram_bytes / (1024 ** 3)

    @staticmethod
    def _cpu_time(process: psutil.Process) -> float:
        """Return cumulative user and system CPU time for one process."""
        counters = process.cpu_times()
        return float(counters.user + counters.system)

    def reset(self) -> None:
        """Clear counters before monitoring another workload run."""
        self.cpu_times.clear()
        self.last_sample_time = None


class ResourceTrace:
    """Store synchronized resource samples and their phase labels."""

    def __init__(self):
        """Initialize empty trace arrays."""
        self.timestamps: list[float] = []
        self.phases: list[str] = []
        self.cpu_percent: list[float] = []
        self.ram_gb: list[float] = []

    def clear(self) -> None:
        """Remove all samples before a new monitoring run."""
        self.timestamps.clear()
        self.phases.clear()
        self.cpu_percent.clear()
        self.ram_gb.clear()

    def append(self, elapsed: float, cpu: float, ram: float, phase: str) -> None:
        """Append one synchronized sample."""
        self.timestamps.append(elapsed)
        self.cpu_percent.append(cpu)
        self.ram_gb.append(ram)
        self.phases.append(phase)


class ResourcePhaseTracker:
    """Track relative boundaries and the currently sampled workload phase."""

    def __init__(self):
        """Initialize an empty baseline/training/post-training state."""
        self.training_start: float | None = None
        self.training_end: float | None = None
        self.phase = "baseline"

    def reset(self) -> None:
        """Reset phase boundaries before a new monitoring run."""
        self.training_start = None
        self.training_end = None
        self.phase = "baseline"

    def start_training(self, elapsed: float) -> None:
        """Mark the beginning of the measured training phase."""
        self.training_start = elapsed
        self.phase = "training"

    def finish_training(self, elapsed: float) -> float:
        """Mark training completion and return its non-negative duration."""
        self.training_end = elapsed
        self.phase = "post_training"
        return max(0.0, elapsed - (self.training_start or 0.0))


class ResourceStatsCalculator:
    """Calculate statistics from a resource trace and optional training window."""

    @staticmethod
    def calculate(trace: ResourceTrace, duration: float,
                  training_start: float | None, training_end: float | None):
        """Return min, max, mean, deviation, duration and sample count."""
        if not trace.timestamps:
            return None
        values = ResourceStatsCalculator._window(trace, training_start, training_end)
        return ResourceStatsCalculator._summary(values, duration)

    @staticmethod
    def _window(trace, start, end):
        """Select training samples when both phase boundaries are present."""
        arrays = np.asarray(trace.timestamps), np.asarray(trace.cpu_percent), np.asarray(trace.ram_gb)
        if start is None or end is None:
            return arrays
        mask = (arrays[0] >= start) & (arrays[0] <= end)
        if not mask.any():
            return arrays
        return tuple(array[mask] for array in arrays)

    @staticmethod
    def _summary(values, duration):
        """Build the public statistics dictionary."""
        timestamps, cpu, ram = values
        return {
            "cpu": {"min": cpu.min(), "max": cpu.max(), "mean": cpu.mean(), "std": cpu.std()},
            "ram": {"min": ram.min(), "max": ram.max(), "mean": ram.mean(), "std": ram.std()},
            "duration": duration,
            "samples": len(timestamps),
        }


class ResourceMonitor:
    """Collect process and descendant CPU/RAM samples across training phases."""

    def __init__(
        self, interval: float = 0.1, debug: bool = False,
        process_id: int | None = None,
    ):
        """Initialize a thread-based monitor for one workload."""
        self.interval, self.debug = interval, debug
        self.process_id = process_id
        self.trace = ResourceTrace()
        self._bind_trace()
        self._init_runtime()
        self._init_process_sampler()
        self._init_thread_state()

    def _bind_trace(self):
        """Expose trace arrays under the legacy monitor attribute names."""
        self.timestamps, self.phases = self.trace.timestamps, self.trace.phases
        self.cpu_percent, self.ram_gb = self.trace.cpu_percent, self.trace.ram_gb

    def _init_runtime(self):
        """Initialize elapsed-time and phase state."""
        self.start_time = None
        self.duration = self.monitoring_duration = 0.0
        self.training_start_time = self.training_end_time = None
        self.training_duration = None
        self.phase_tracker = ResourcePhaseTracker()

    def _init_process_sampler(self):
        """Initialize process sampler using all available CPU cores."""
        self.process = psutil.Process(self.process_id)
        self.num_cores = psutil.cpu_count() or 1
        self.sampler = ProcessResourceSampler(self.process, self.num_cores)

    def _init_thread_state(self):
        """Initialize synchronization and worker-thread state."""
        self._monitoring = False
        self._monitor_thread = None
        self._stop_event = threading.Event()
        self._started = False
        self._sample_lock = threading.Lock()
        self._phase = self.phase_tracker.phase

    def start(self):
        """Start sampling without recording an artificial zero-time sample."""
        if self.debug:
            print(f"[INFO] ResourceMonitor uruchomiony z interwałem={self.interval}s")
        self._reset_run_state()
        self.start_time = time.time()
        self._prime_cpu_counters()
        self._monitoring = True
        self._stop_event.clear()
        self._monitor_thread = threading.Thread(target=self._monitor_resources, daemon=True)
        self._monitor_thread.start()

    def _reset_run_state(self):
        """Clear trace and phase state before starting a new run."""
        self._started = True
        self.trace.clear()
        self.phase_tracker.reset()
        self.sampler.reset()
        self.duration = self.monitoring_duration = 0.0
        self.training_start_time = self.training_end_time = None
        self.training_duration = None
        self._phase = self.phase_tracker.phase

    def mark_training_start(self):
        """Mark training start and reset process CPU counters."""
        if not self._started or self.start_time is None:
            raise RuntimeError("ResourceMonitor.start() must be called before training markers")
        self.training_start_time = time.time() - self.start_time
        self.phase_tracker.start_training(self.training_start_time)
        self._phase = self.phase_tracker.phase
        self._prime_cpu_counters()

    def mark_training_end(self):
        """Mark training completion and capture a boundary sample if needed."""
        if self.start_time is None:
            raise RuntimeError("ResourceMonitor.start() must be called before training markers")
        started = self.start_time or time.time()
        elapsed = time.time() - started
        active_sample_exists = bool(self.timestamps) and any(
            timestamp >= (self.training_start_time or 0.0)
            for timestamp in self.timestamps
        )
        time_since_last_sample = elapsed - self.timestamps[-1] if self.timestamps else elapsed
        if (
            not active_sample_exists
            or time_since_last_sample >= self.interval * 0.8
        ):
            self._collect_sample(elapsed_override=elapsed)
        self.training_duration = self.phase_tracker.finish_training(elapsed)
        self.training_end_time = self.phase_tracker.training_end
        self._phase = self.phase_tracker.phase

    def wait_for_post_training_idle(
        self, minimum: float = 0.75, maximum: float = 3.0,
        stable_samples: int = 5,
    ) -> bool:
        """Keep sampling until post-training CPU stabilizes or time expires."""
        if self.training_end_time is None:
            raise RuntimeError("mark_training_end() must be called before waiting")
        started = time.monotonic()
        while time.monotonic() - started < maximum:
            time.sleep(min(self.interval, 0.1))
            elapsed = time.monotonic() - started
            if elapsed >= minimum and self._post_training_cpu_is_idle(stable_samples):
                return True
        return False

    def _post_training_cpu_is_idle(self, sample_count: int) -> bool:
        """Return whether recent CPU samples match the measured baseline."""
        pairs = zip(self.phases, self.cpu_percent)
        baseline = [value for phase, value in pairs if phase == "baseline"]
        pairs = zip(self.phases, self.cpu_percent)
        post = [value for phase, value in pairs if phase == "post_training"]
        if not baseline or len(post) < sample_count:
            return False
        tolerance = max(0.5, float(np.std(baseline)) * 2.0)
        threshold = float(np.median(baseline)) + tolerance
        return bool(np.max(post[-sample_count:]) <= threshold)

    def stop(self):
        """Stop sampling, close the trace and print optional diagnostics."""
        self._monitoring = False
        self._stop_event.set()
        if self._monitor_thread:
            self._monitor_thread.join(timeout=2)
        self._finalize_duration()
        if self.debug:
            self._print_summary()

    def _finalize_duration(self):
        """Capture the final boundary and choose the public duration."""
        elapsed = time.time() - self.start_time
        self.monitoring_duration = self.duration = elapsed
        last = self.timestamps[-1] if self.timestamps else 0.0
        if self.training_end_time is None and (not self.timestamps or elapsed - last >= self.interval * 0.8):
            self._collect_sample()
        if self.training_duration is not None:
            self.duration = self.training_duration
        elif self.timestamps:
            self.duration = max(self.duration, self.timestamps[-1])

    def _print_summary(self):
        """Print compact diagnostics for interactive notebook users."""
        stats = self.get_stats()
        if not stats:
            return
        print(f"\n[INFO] Monitorowanie zakończone; czas: {stats['duration']:.2f}s")
        print(f"[INFO] Próbki: {stats['samples']}; CPU średnio: {stats['cpu']['mean']:.1f}%")
        print(f"[INFO] RAM średnio: {stats['ram']['mean']:.2f} GB")

    def _monitor_resources(self):
        while not self._stop_event.wait(self.interval):
            self._collect_sample()

    def _processes(self):
        """Return processes from the delegated sampler."""
        return self.sampler.processes()

    def _prime_cpu_counters(self):
        """Prime CPU counters without recording a measurement."""
        with self._sample_lock:
            self.sampler.prime()

    def _collect_sample(self, elapsed_override=None):
        """Collect one sample and append it to the current phase trace."""
        try:
            with self._sample_lock:
                cpu, ram_gb = self.sampler.sample()
            started = self.start_time or time.time()
            elapsed = elapsed_override if elapsed_override is not None else time.time() - started
            self.trace.append(elapsed, cpu, ram_gb, self._phase)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return

    def plot(self, title: str = "Resource Usage", save_path: Optional[str] = None):
        """Render the trace through the dedicated plotting component."""
        from .resource_plotter import plot_monitor
        return plot_monitor(self, title, save_path)

    def save_trace(self, path):
        """Save the training-window trace as a compact CSV file."""
        times = np.asarray(self.timestamps, dtype=float)
        cpu_data = np.asarray(self.cpu_percent, dtype=float)
        ram_data = np.asarray(self.ram_gb, dtype=float)
        if self.training_start_time is not None and self.training_end_time is not None:
            active = (
                (times >= self.training_start_time)
                & (times <= self.training_end_time)
            )
            times = times[active] - self.training_start_time
            cpu_data = cpu_data[active]
            ram_data = ram_data[active]
        path = str(path)
        with open(path, "w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["czas_s", "cpu_procent", "ram_gb"])
            writer.writerows(zip(times, cpu_data, ram_data))

    def save_phase_trace(self, path):
        """Save the complete trace with baseline and phase labels."""
        times = np.asarray(self.timestamps, dtype=float)
        cpu_data = np.asarray(self.cpu_percent, dtype=float)
        ram_data = np.asarray(self.ram_gb, dtype=float)
        phases = list(self.phases)
        if len(phases) != len(times):
            phases = ["training"] * len(times)
        path = str(path)
        with open(path, "w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["czas_s", "faza", "cpu_procent", "ram_gb"])
            writer.writerows(zip(times, phases, cpu_data, ram_data))

    def get_stats(self):
        """Return statistics for the current trace and training window."""
        return ResourceStatsCalculator.calculate(
            self.trace, self.duration, self.training_start_time, self.training_end_time
        )
