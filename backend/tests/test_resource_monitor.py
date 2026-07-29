from utils.resource_monitor import (
    ProcessResourceSampler,
    ResourceMonitor,
    ResourceStatsCalculator,
    ResourceTrace,
)


class FakeProcess:
    """Expose deterministic process counters for sampler tests."""

    pid = 123

    def __init__(self):
        self.cpu_seconds = 0.0
        self.rss = 1024 ** 3

    def children(self, recursive=True):
        """Return no child processes."""
        return []

    def cpu_times(self):
        """Return cumulative CPU time."""
        from types import SimpleNamespace
        return SimpleNamespace(user=self.cpu_seconds, system=0.0)

    def memory_info(self):
        """Return deterministic resident memory."""
        from types import SimpleNamespace
        return SimpleNamespace(rss=self.rss)


def test_process_sampler_uses_cumulative_cpu_time_per_interval(monkeypatch):
    process = FakeProcess()
    sampler = ProcessResourceSampler(process, num_cores=4)
    clock = iter([10.0, 11.0])
    monkeypatch.setattr(
        "utils.resource_monitor.time.monotonic", lambda: next(clock)
    )

    sampler.prime()
    process.cpu_seconds = 1.0
    cpu_percent, ram_gb = sampler.sample()

    assert cpu_percent == 25.0
    assert ram_gb == 1.0


def test_process_sampler_priming_resets_cpu_measurement_window(monkeypatch):
    process = FakeProcess()
    sampler = ProcessResourceSampler(process, num_cores=2)
    clock = iter([10.0, 11.0, 12.0])
    monkeypatch.setattr(
        "utils.resource_monitor.time.monotonic", lambda: next(clock)
    )

    sampler.prime()
    process.cpu_seconds = 8.0
    sampler.prime()
    process.cpu_seconds = 9.0

    assert sampler.sample()[0] == 50.0


def test_stop_skips_noisy_boundary_sample_after_recent_measurement(monkeypatch):
    events = []
    monitor = ResourceMonitor(interval=0.1)
    monitor._monitoring = True
    monitor.start_time = 10.0
    monitor.timestamps = [10.95]
    monitor._collect_sample = lambda: events.append("sample")
    monkeypatch.setattr("utils.resource_monitor.time.time", lambda: 21.0)

    class MonitorThread:
        def join(self, timeout):
            events.append(("join", timeout))

    monitor._monitor_thread = MonitorThread()

    monitor.stop()

    assert events == [("join", 2)]
    assert monitor._stop_event.is_set()
    assert monitor._monitoring is False


def test_stop_collects_one_sample_for_short_run(monkeypatch):
    events = []
    monitor = ResourceMonitor(interval=0.1)
    monitor._monitoring = True
    monitor.start_time = 10.0
    monitor._collect_sample = lambda: events.append("sample")
    monkeypatch.setattr("utils.resource_monitor.time.time", lambda: 10.03)

    class MonitorThread:
        def join(self, timeout):
            events.append(("join", timeout))

    monitor._monitor_thread = MonitorThread()
    monitor.stop()

    assert events == [("join", 2), "sample"]


def test_stop_does_not_collect_post_training_sample(monkeypatch):
    events = []
    monitor = ResourceMonitor(interval=0.1)
    monitor._monitoring = True
    monitor._started = True
    monitor.start_time = 10.0
    monitor.training_start_time = 0.2
    monitor.training_end_time = 0.8
    monitor.training_duration = 0.6
    monitor.timestamps = [0.2, 0.8]
    monitor._collect_sample = lambda: events.append("sample")
    monkeypatch.setattr("utils.resource_monitor.time.time", lambda: 11.0)

    class MonitorThread:
        def join(self, timeout):
            events.append(("join", timeout))

    monitor._monitor_thread = MonitorThread()
    monitor.stop()

    assert events == [("join", 2)]
    assert monitor.duration == 0.6


def test_post_training_idle_requires_cpu_to_return_to_baseline():
    monitor = ResourceMonitor(interval=0.1)
    for cpu in (1.0, 1.2, 0.8):
        monitor.trace.append(0.0, cpu, 1.0, "baseline")
    monitor.trace.append(0.5, 8.0, 1.2, "training")
    for cpu in (1.3, 1.1, 1.0):
        monitor.trace.append(1.0, cpu, 1.2, "post_training")

    assert monitor._post_training_cpu_is_idle(3)

    monitor.trace.append(1.1, 3.0, 1.2, "post_training")

    assert not monitor._post_training_cpu_is_idle(3)


def test_training_statistics_exclude_post_training_resource_usage():
    trace = ResourceTrace()
    trace.append(0.0, 1.0, 1.0, "baseline")
    trace.append(1.0, 20.0, 2.0, "training")
    trace.append(2.0, 30.0, 3.0, "training")
    trace.append(3.0, 99.0, 9.0, "post_training")

    stats = ResourceStatsCalculator.calculate(trace, 1.0, 1.0, 2.0)

    assert stats["cpu"]["max"] == 30.0
    assert stats["ram"]["max"] == 3.0
    assert stats["samples"] == 2


def test_resource_plotter_represents_training_mean_as_a_line():
    import numpy as np
    from matplotlib import pyplot as plt

    from utils.resource_plotter import draw_axis

    figure, axis = plt.subplots()
    draw_axis(
        axis, np.array([0.0, 0.1, 0.2]), np.array([1.0, 2.0, 3.0]),
        "Użycie CPU (%)", "#4477AA",
    )

    mean_line = next(
        line for line in axis.lines
        if line.get_label() == "Średnia podczas uczenia"
    )
    assert np.allclose(mean_line.get_ydata(), [2.0, 2.0])
    assert mean_line.get_color() == "#228833"
    assert axis.texts[0].get_text() == "Średnia: 2%"
    plt.close(figure)
