import ast
import json
from pathlib import Path


BACKEND_ROOT = Path(__file__).resolve().parents[1]
URL_NOTEBOOK_ROOT = BACKEND_ROOT / "notebooks" / "url_corpus"
ORIGINAL_NOTEBOOK_ROOT = BACKEND_ROOT / "notebooks" / "original"
URL_NOTEBOOKS = tuple(URL_NOTEBOOK_ROOT / name for name in (
    "8_algorithm_runtime_complexity_benchmark_url_corpus.ipynb",
    "9_external_generalization_benchmark_url_corpus.ipynb",
    "10_realtime_application_benchmark_url_corpus.ipynb",
))
ORIGINAL_NOTEBOOKS = tuple(ORIGINAL_NOTEBOOK_ROOT / name for name in (
    "8_algorithm_runtime_complexity_benchmark.ipynb",
    "9_held_out_performance_uncertainty_benchmark.ipynb",
    "10_realtime_application_benchmark.ipynb",
))
NOTEBOOKS = URL_NOTEBOOKS + ORIGINAL_NOTEBOOKS


def source(path: Path) -> str:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    return "\n".join(
        "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )


def test_supplementary_notebooks_exist_and_never_train():
    for path in NOTEBOOKS:
        code = source(path)
        assert ".fit(" not in code
        assert "RESULTS_ROOT /" in code
        assert "'original'" in code or "'url_corpus'" in code


def test_runtime_notebook_has_exact_measurement_contract():
    for path in (URL_NOTEBOOKS[0], ORIGINAL_NOTEBOOKS[0]):
        code = source(path)
        assert "BATCH_SIZES = (1, 32, 256, 1000)" in code
        assert "WARMUPS = 10" in code
        assert "REPETITIONS = 100" in code
        assert "LOAD_REPETITIONS = 30" in code
        assert "classifier_only" in code
        assert "scaler_and_classifier" in code
        assert ".melt(" in code
        assert ".dropna(subset=['value'])" in code


def test_runtime_notebook_outputs_do_not_show_nan():
    for path in (URL_NOTEBOOKS[0], ORIGINAL_NOTEBOOKS[0]):
        notebook = json.loads(path.read_text(encoding="utf-8"))
        text_outputs = []
        for cell in notebook["cells"]:
            for output in cell.get("outputs", []):
                text_outputs.append(output.get("text", ""))
                data = output.get("data", {})
                text_outputs.extend(
                    [data.get("text/plain", ""), data.get("text/html", "")]
                )
        assert "NaN" not in "".join(map(str, text_outputs))


def test_generalization_notebook_has_bootstrap_and_quality_gate_contract():
    code = source(URL_NOTEBOOKS[1])
    assert "BOOTSTRAP_REPETITIONS = 1000" in code
    assert "RANDOM_STATE = 42" in code
    assert "recheck_v4_domain_fix" in code
    assert "quality_gate(metrics)" in code
    assert "'overlap_domains': len(overlap)" in code


def test_realtime_notebook_uses_only_safe_url_and_always_cleans_up():
    for path in (URL_NOTEBOOKS[2], ORIGINAL_NOTEBOOKS[2]):
        code = source(path)
        assert "SAFE_URL = 'https://example.com/account/verify?token=benchmark'" in code
        assert "http://" not in code
        assert "finally:" in code
        assert "phishing_service.orchestrator.cleanup_all()" in code
        assert "ContainerConfig.NETWORK_MODE" in code


def test_warmups_are_declared_separately_from_measurements():
    for path in (URL_NOTEBOOKS[0], ORIGINAL_NOTEBOOKS[0]):
        assert "measure_ns(operation, WARMUPS, REPETITIONS)" in source(path)
    for path in (URL_NOTEBOOKS[2], ORIGINAL_NOTEBOOKS[2]):
        realtime = source(path)
        assert "for _ in range(API_WARMUPS):" in realtime
        assert "for repetition in range(1, API_REPETITIONS + 1):" in realtime


def test_original_quality_notebook_reports_honest_scope():
    code = source(ORIGINAL_NOTEBOOKS[1])
    assert "BOOTSTRAP_REPETITIONS = 1000" in code
    assert "'external_generalization_available': False" in code
    assert "Brak niezależnego zbioru zewnętrznego" in code
    assert "optimized_minus_full" in code


def test_supplementary_notebooks_use_readable_shared_plots():
    for path in (URL_NOTEBOOKS[0], ORIGINAL_NOTEBOOKS[0]):
        assert "plot_runtime_results(" in source(path)
    assert "plot_generalization_results(" in source(URL_NOTEBOOKS[1])
    assert "plot_heldout_results(" in source(ORIGINAL_NOTEBOOKS[1])
    for path in (URL_NOTEBOOKS[2], ORIGINAL_NOTEBOOKS[2]):
        assert "plot_realtime_results(" in source(path)


def test_notebook_functions_respect_twenty_line_limit():
    for path in NOTEBOOKS:
        tree = ast.parse(source(path))
        functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
        assert all(node.end_lineno - node.lineno + 1 <= 20 for node in functions), path.name
