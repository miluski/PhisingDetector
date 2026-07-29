import pandas as pd

from scripts.regenerate_resource_plots import regenerate_scope


def test_cached_resource_trace_regenerates_bilingual_cpu_and_ram(tmp_path):
    root = tmp_path / "results"
    trace_dir = root / "original" / "resource_usage"
    trace_dir.mkdir(parents=True)
    _trace_frame().to_csv(
        trace_dir / "logistic_regression_full.csv", index=False
    )

    paths = regenerate_scope(root, "original")

    assert len(paths) == 2
    assert all(path.is_file() and path.parent.name == "pl" for path in paths)
    assert all((path.parent.parent / "eng" / path.name).is_file()
               for path in paths)


def _trace_frame() -> pd.DataFrame:
    """Return a trace whose final phase never reaches an exact zero."""
    return pd.DataFrame({
        "czas_s": [0.0, 0.1, 0.2, 0.3, 0.4],
        "faza": [
            "baseline", "baseline", "training",
            "post_training", "post_training",
        ],
        "cpu_procent": [1.0, 1.0, 8.0, 3.0, 1.2],
        "ram_gb": [0.2, 0.2, 0.5, 0.3, 0.21],
    })
