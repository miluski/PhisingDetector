import ast
import io
import json
import subprocess
import sys
import tokenize
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest

from scripts.prepare_url_corpus_notebooks import (
    SOURCE_NOTEBOOKS,
    preserve_matching_runtime_state,
    render_notebook,
)
from utils.notebook_plot_style import scientific_scale


BACKEND_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_ROOT = BACKEND_ROOT / "notebooks"
PHASE_PLOT_PATH = BACKEND_ROOT / "utils" / "phase_testing_plots.py"


def model_notebooks():
    originals = sorted((NOTEBOOK_ROOT / "original").glob("[1-6]*.ipynb"))
    url_corpus = sorted((NOTEBOOK_ROOT / "url_corpus").glob("[1-6]*.ipynb"))
    return originals + url_corpus


def all_notebooks():
    originals = sorted((NOTEBOOK_ROOT / "original").glob("[0-7]*.ipynb"))
    url_corpus = sorted((NOTEBOOK_ROOT / "url_corpus").glob("[0-7]*.ipynb"))
    return originals + url_corpus


def test_url_corpus_notebooks_use_current_feature_directory():
    for path in sorted((NOTEBOOK_ROOT / "url_corpus").glob("[0-7]*.ipynb")):
        _, code = notebook_source(path)
        assert "quality_large_splits" not in code, path.name
        if "load_split_frames(" in code:
            assert "quality_v4_repaired_features" in code, path.name


def notebook_source(path: Path) -> tuple[dict, str]:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    code = "\n".join(
        "".join(cell.get("source", []))
        if isinstance(cell.get("source", []), list)
        else cell.get("source", "")
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )
    return notebook, code


def distribution_geometry_function():
    """Load distribution geometry directly from the source notebook."""
    path = NOTEBOOK_ROOT / "original" / "0_comprehensive_statistical_analysis.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    cell = next(item for item in notebook["cells"] if item.get("id") == "48d9040b")
    tree = ast.parse("".join(cell["source"]))
    required = {
        "_discrete_feature_values", "_discrete_edges", "_adaptive_bin_count",
        "_discrete_ticks", "_integer_interval_edges",
        "_continuous_geometry", "_distribution_geometry",
    }
    nodes = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in required
    ]
    namespace = {"np": np, "scientific_scale": scientific_scale}
    exec(compile(ast.Module(nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace["_distribution_geometry"]


def notebook_markdown(notebook: dict) -> str:
    """Join all Markdown cell sources from a notebook."""
    return "\n".join(
        "".join(cell.get("source", []))
        if isinstance(cell.get("source", []), list)
        else cell.get("source", "")
        for cell in notebook["cells"]
        if cell.get("cell_type") == "markdown"
    )


def compiled_notebook_function(source: str, name: str):
    """Compile one named function from a notebook code cell."""
    tree = ast.parse(source)
    node = next(
        item for item in tree.body
        if isinstance(item, ast.FunctionDef) and item.name == name
    )
    namespace = {"np": np}
    exec(compile(ast.Module([node], []), "<notebook>", "exec"), namespace)
    return namespace[name]


def compiled_notebook_functions(source: str, names: tuple[str, ...]):
    """Compile cooperating named functions from one notebook code cell."""
    tree = ast.parse(source)
    selected = set(names)
    nodes = [
        item for item in tree.body
        if isinstance(item, ast.FunctionDef) and item.name in selected
    ]
    namespace = {"np": np}
    exec(compile(ast.Module(nodes, []), "<notebook>", "exec"), namespace)
    return tuple(namespace[name] for name in names)


def unused_import_bindings(trees: list[ast.AST]) -> set[str]:
    imported = set()
    loaded = set()
    exported = set()

    for tree in trees:
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                loaded.add(node.id)
            elif isinstance(node, ast.Import):
                imported.update(
                    alias.asname or alias.name.split(".")[0] for alias in node.names
                )
            elif isinstance(node, ast.ImportFrom):
                if node.module != "__future__":
                    imported.update(
                        alias.asname or alias.name
                        for alias in node.names
                        if alias.name != "*"
                    )
            elif (
                isinstance(node, ast.Assign)
                and any(
                    isinstance(target, ast.Name) and target.id == "__all__"
                    for target in node.targets
                )
                and isinstance(node.value, (ast.List, ast.Tuple))
            ):
                exported.update(
                    element.value
                    for element in node.value.elts
                    if isinstance(element, ast.Constant)
                    and isinstance(element.value, str)
                )

    return imported - loaded - exported


@pytest.mark.parametrize("path", model_notebooks(), ids=lambda path: path.name)
def test_model_notebook_contract(path: Path):
    notebook, code = notebook_source(path)

    for cell in notebook["cells"]:
        if cell.get("cell_type") != "code":
            continue
        source = cell.get("source", [])
        source = "".join(source) if isinstance(source, list) else source
        ast.parse(source)

    assert "from utils.notebook_plot_style import" in code
    assert "nested_backend = candidate / 'backend'" in code
    assert "os.chdir(notebooks_dir)" in code
    assert "save_dataset_figure" in code
    assert "plt.savefig(" not in code
    assert "validation_split" not in code
    assert "label='Błąd testowania'" not in code
    assert "Wynik testowania" not in code
    assert "reduced'" not in code
    assert ".set_title(" not in code
    assert ".suptitle(" not in code
    assert "title=axis_relation_title(" not in code
    assert "title_text=axis_relation_title(" not in code
    assert "subplot_titles=" not in code
    assert "renamed_evaluation_results" not in code
    if "remove_dataset_figure(" in code:
        assert "    remove_dataset_figure," in code
    if "CLASS_COLORS[" in code:
        assert "    CLASS_COLORS," in code


def test_mlp_uses_explicit_validation_and_inference_metrics():
    for path in [
        NOTEBOOK_ROOT / "original" / "5_mlp_phishing_detection.ipynb",
        NOTEBOOK_ROOT / "url_corpus" / "5_mlp_phishing_detection_url_corpus.ipynb",
    ]:
        _, code = notebook_source(path)
        assert "'validation_data': (" in code
        assert "data['X_val_scaled'], data['y_val']" in code
        assert "EpochEndEvaluationCallback" not in code
        assert "epoch_end_train_loss" in code
        assert "epoch_end_validation_accuracy" in code
        assert "label='Zbiór walidacyjny'" in code
        assert "Zbiór walidacyjny — binarna entropia krzyżowa" not in code


@pytest.mark.parametrize("path", all_notebooks(), ids=lambda path: path.name)
def test_every_notebook_has_portable_backend_bootstrap(path: Path):
    _, code = notebook_source(path)
    assert "nested_backend = candidate / 'backend'" in code
    assert "os.chdir(notebooks_dir)" in code
    assert "sys.path.insert(0, str(backend_root))" in code


@pytest.mark.parametrize("path", all_notebooks(), ids=lambda path: path.name)
def test_every_notebook_is_clean_valid_python(path: Path):
    notebook, _ = notebook_source(path)
    for cell in notebook["cells"]:
        if cell.get("cell_type") != "code":
            continue
        source = cell.get("source", [])
        source = "".join(source) if isinstance(source, list) else source
        ast.parse(source)
        tokens = tokenize.generate_tokens(io.StringIO(source).readline)
        ignored = {
            tokenize.COMMENT,
            tokenize.NL,
            tokenize.NEWLINE,
            tokenize.INDENT,
            tokenize.DEDENT,
            tokenize.ENDMARKER,
            tokenize.ENCODING,
        }
        assert any(
            token.type not in ignored and token.string.strip() for token in tokens
        ), f"Empty/comment-only code cell in {path.name}"


@pytest.mark.parametrize("path", all_notebooks(), ids=lambda path: path.name)
def test_every_notebook_reloads_plot_style(path: Path):
    _, code = notebook_source(path)

    assert "if 'utils.notebook_plot_style' in sys.modules:" in code
    assert "importlib.reload(sys.modules['utils.notebook_plot_style'])" in code


@pytest.mark.parametrize("path", all_notebooks(), ids=lambda path: path.name)
def test_every_notebook_has_no_unused_imports(path: Path):
    notebook, _ = notebook_source(path)
    trees = []
    for cell in notebook["cells"]:
        if cell.get("cell_type") != "code":
            continue
        source = cell.get("source", [])
        source = "".join(source) if isinstance(source, list) else source
        trees.append(ast.parse(source))

    assert unused_import_bindings(trees) == set()


def test_python_modules_have_no_unused_imports():
    intentional_side_effect_imports = {
        BACKEND_ROOT / "run.py": {"app"},
    }

    failures = {}
    for path in sorted(BACKEND_ROOT.rglob("*.py")):
        unused = unused_import_bindings([ast.parse(path.read_text(encoding="utf-8"))])
        unused -= intentional_side_effect_imports.get(path, set())
        if unused:
            failures[path.relative_to(BACKEND_ROOT).as_posix()] = sorted(unused)

    assert failures == {}


def test_statistical_analysis_excludes_url_metadata():
    for path in [
        NOTEBOOK_ROOT / "original" / "0_comprehensive_statistical_analysis.ipynb",
        NOTEBOOK_ROOT / "url_corpus" / "0_comprehensive_statistical_analysis_url_corpus.ipynb",
    ]:
        _, code = notebook_source(path)
        assert "FEATURE_COLUMNS = feature_columns(df)" in code
        assert "feature_cols = feature_columns(df)" in code
        assert "df.select_dtypes(include=[np.number]).columns[:-1]" not in code
        assert "df.columns[-1]" not in code


def test_model_testing_never_passes_metadata_to_models():
    original_path = (
        NOTEBOOK_ROOT / "original" / "6_comprehensive_model_testing.ipynb"
    )
    url_path = (
        NOTEBOOK_ROOT / "url_corpus" /
        "6_comprehensive_model_testing_url_corpus.ipynb"
    )
    _, original_code = notebook_source(original_path)
    _, url_code = notebook_source(url_path)
    assert "columns = feature_columns(df)" in original_code
    assert "X = df[columns]" in original_code
    assert "return df[feature_names], df['phishing']" in url_code
    for code in (original_code, url_code):
        assert "X = df.iloc[:, :-1]" not in code
        assert "X_test = test_df.iloc[:, :-1]" not in code


def test_model_testing_notebooks_show_representative_details_only():
    paths = (
        NOTEBOOK_ROOT / "original" / "6_comprehensive_model_testing.ipynb",
        NOTEBOOK_ROOT
        / "url_corpus"
        / "6_comprehensive_model_testing_url_corpus.ipynb",
    )
    for path in paths:
        notebook, code = notebook_source(path)
        markdown = "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
            if cell.get("cell_type") == "markdown"
        )
        assert "display_in_notebook=show_inline" in code
        assert "phase_mapping[phase_name] == 5" in code
        assert "Rozkład klas: 50/50" in code
        assert "### Mapa wyników" in markdown
        assert "reprezentatywną część szczegółowych wykresów" in markdown


def test_url_model_testing_uses_matching_quality_v4_payloads():
    path = (
        NOTEBOOK_ROOT / "url_corpus" /
        "6_comprehensive_model_testing_url_corpus.ipynb"
    )
    _, code = notebook_source(path)
    assert "models' / 'staged_url_corpus_v4'" in code
    assert "dataset_selected_test.csv" in code
    assert "artifact_modes = {'full': 'full', 'optimized': 'selected'}" in code
    assert "model_data['feature_names']" in code
    assert "predict_with_model(model_data, X_test)" in code
    assert ".fit(" not in code


def test_url_mlp_imports_itertools_for_preserved_grid_search():
    path = (
        NOTEBOOK_ROOT / "url_corpus" /
        "5_mlp_phishing_detection_url_corpus.ipynb"
    )
    _, code = notebook_source(path)
    assert "itertools.product(" in code
    assert "import itertools\n" in code


def test_mlp_notebooks_store_whole_mean_fold_sample_counts():
    """Keep averaged fold sizes integer-valued before plotting and storage."""
    paths = (
        NOTEBOOK_ROOT / "original" / "5_mlp_phishing_detection.ipynb",
        NOTEBOOK_ROOT / "url_corpus" / "5_mlp_phishing_detection_url_corpus.ipynb",
    )
    for path in paths:
        _, code = notebook_source(path)
        assert "mean_learning_train_sizes = np.rint(" in code
        assert "learning_train_sizes.mean(axis=1)\n    ).astype(int)" in code
        assert "'learning_train_sizes': mean_learning_train_sizes" in code
        assert code.count("        mean_learning_train_sizes,\n") == 3


def test_imbalance_notebook_documents_url_controlled_experiment():
    _, original_code = notebook_source(
        NOTEBOOK_ROOT / "original" / "7_dataset_imbalance_research.ipynb"
    )
    _, url_code = notebook_source(
        NOTEBOOK_ROOT / "url_corpus" / "7_dataset_imbalance_research_url_corpus.ipynb"
    )
    assert "build_stratified_frames" in original_code
    assert "build_controlled_imbalance_frame" in url_code
    assert "ORIGINAL_IMBALANCE_RATIO" in url_code
    assert "test_df = frames['test']" in url_code


def test_imbalance_notebooks_use_shared_accessible_plot_contract():
    for directory in ("original", "url_corpus"):
        path = next((NOTEBOOK_ROOT / directory).glob("7_*.ipynb"))
        _, code = notebook_source(path)
        assert "plot_class_distribution" in code
        assert "plot_precision_recall_curves" in code
        assert "plot_imbalance_confusion_matrix" in code
        assert "shared_confusion_maximum" in code
        assert "cmap='cividis'" not in code
        assert "VARIANT_COLORS" not in code


def test_no_stale_hardcoded_feature_counts():
    stale_labels = (
        "111 cech",
        "73 cech",
        "Pełny (111)",
        "Pelny (111)",
        "Zoptymalizowany (73)",
    )
    for path in all_notebooks():
        _, code = notebook_source(path)
        assert not any(label in code for label in stale_labels), path.name


def test_original_model_notebooks_store_dynamic_feature_counts():
    for path in sorted((NOTEBOOK_ROOT / "original").glob("[1-5]*.ipynb")):
        _, code = notebook_source(path)
        assert "'feature_count': X.shape[1]" in code, path.name


def test_prediction_figures_show_one_labelled_test_sample_each():
    for directory in ("original", "url_corpus"):
        for path in sorted((NOTEBOOK_ROOT / directory).glob("[1-5]*.ipynb")):
            _, code = notebook_source(path)
            assert "sample_positions =" in code
            assert "plot_probability_figure(" not in code
            assert "plot_probability_distribution(" not in code
            assert "plot_sample_probability_prediction(" in code
            assert "evaluation_results[dataset_name]['y_pred_proba']" in code
            assert ".get('decision_threshold', 0.5)" in code
            assert "label_values[position]" in code
            assert "sample_id" in code
            assert "for language in ['pl', 'en']" in code
            assert "language=language" in code
            assert "plt.show()" in code


def test_notebooks_do_not_use_hatched_bars():
    for path in all_notebooks():
        _, code = notebook_source(path)
        assert "hatch=" not in code, path.name


def test_notebook_axis_labels_avoid_bare_absolute_value_notation():
    for path in all_notebooks():
        _, code = notebook_source(path)
        assert "|Wartość bezwzględna" not in code, path.name
        assert "x_label='|Współczynnik|'" not in code, path.name


def test_sample_predictions_replace_obsolete_probability_figures():
    prefixes = {
        "1_": "logistic_regression",
        "2_": "random_forest",
        "3_": "xgboost",
        "4_": "svm",
        "5_": "mlp",
    }
    for directory in ("original", "url_corpus"):
        for path in sorted((NOTEBOOK_ROOT / directory).glob("[1-5]*.ipynb")):
            _, code = notebook_source(path)
            prefix = prefixes[path.name[:2]]
            assert (
                f"f'{{language_folder}}/{prefix}_sample_prediction_"
                "{sample_kind}.png'" in code
            )
            assert "language_folder = 'pl' if language == 'pl' else 'eng'" in code
            assert "_{language}.png" not in code
            assert f"'{prefix}_probability_distribution.png'" in code
            assert "for obsolete_name in [" in code


def test_mlp_keeps_metric_name_on_axis_and_not_in_legend():
    for path in [
        NOTEBOOK_ROOT / "original" / "5_mlp_phishing_detection.ipynb",
        NOTEBOOK_ROOT / "url_corpus" / "5_mlp_phishing_detection_url_corpus.ipynb",
    ]:
        notebook, code = notebook_source(path)
        markdown = "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
            if cell.get("cell_type") == "markdown"
        )
        assert "set_ylabel('Binarna entropia krzyżowa')" in code
        assert "niżej = lepiej" not in code
        assert "Uczenie modelu — binarna entropia krzyżowa" not in code
        assert "Zbiór walidacyjny — binarna entropia krzyżowa" not in code
        assert "label='Zbiór uczący'" in code
        assert "label='Zbiór walidacyjny'" in code
        assert "label='Uczenie — BCE'" not in code
        assert "label='Walidacja — BCE'" not in code
        assert "## 7.4. Binarna entropia krzyżowa" not in markdown
        assert "Wartość `0` oznacza idealne przewidywania" not in markdown
        assert "Błąd BCE" not in code


def test_learning_curves_distinguish_cv_validation_from_final_testing():
    for directory in ("original", "url_corpus"):
        for path in sorted((NOTEBOOK_ROOT / directory).glob("[1-5]*.ipynb")):
            notebook, code = notebook_source(path)
            markdown = notebook_markdown(notebook)
            assert "x_label='Liczba próbek'" in code
            assert (
                "Oś pozioma nie przedstawia liczby próbek walidacyjnych "
                "ani testowych."
                in markdown
            )
            assert "x_label='Liczba próbek uczących'" not in code
            assert "plot_score_curve(" in code
            assert "learning_curves.png" in code

    for path in all_notebooks():
        _, code = notebook_source(path)
        assert "label='Testowanie'" not in code, path.name


def test_every_model_has_comparable_cross_validated_bce_curve():
    for directory in ("original", "url_corpus"):
        for path in sorted((NOTEBOOK_ROOT / directory).glob("[1-5]*.ipynb")):
            _, code = notebook_source(path)
            assert "y_label='Binarna entropia krzyżowa'" in code, path.name
            assert "(log loss)" not in code, path.name
            assert "bce_learning_curves.png" in code, path.name

            if path.name.startswith("5_"):
                assert "learning_validation_log_loss" in code, path.name
                assert "log_loss(" in code, path.name
            else:
                assert "scoring='neg_log_loss'" in code, path.name


def test_logistic_regression_regularization_curve_uses_linear_scale():
    for directory in ("original", "url_corpus"):
        path = next((NOTEBOOK_ROOT / directory).glob("1_*.ipynb"))
        _, code = notebook_source(path)
        assert "param_range = np.linspace(0.01, 100.0, 20)" in code
        assert (
            "x_label='Wartość parametru C "
            "(C > 0; odwrotność siły regularyzacji)', logarithmic_x=False"
            in code
        )
        assert "param_range = np.logspace" not in code


def test_every_model_has_comparable_classification_error_curve():
    for directory in ("original", "url_corpus"):
        for path in sorted((NOTEBOOK_ROOT / directory).glob("[1-5]*.ipynb")):
            _, code = notebook_source(path)
            assert "Współczynnik błędu klasyfikacji" in code, path.name
            assert "error_rate_learning_curves.png" in code, path.name

            if path.name.startswith("5_"):
                assert "learning_validation_error_rate" in code, path.name
            else:
                assert "scoring='accuracy'" in code, path.name
                assert "1.0 - validation_accuracy" in code, path.name


def test_every_model_reports_final_train_and_held_out_test_error():
    for directory in ("original", "url_corpus"):
        for path in sorted((NOTEBOOK_ROOT / directory).glob("[1-5]*.ipynb")):
            _, code = notebook_source(path)
            assert "train_error_rate = 1.0 - accuracy_score" in code, path.name
            assert "test_error_rate = 1.0 - accuracy" in code, path.name
            assert "plot_final_classification_error(" not in code, path.name
            assert "final_error_rate.png" in code, path.name
            assert "remove_dataset_figure(" in code, path.name
            assert "Błąd uczenia:" in code, path.name
            assert "Błąd testowania:" in code, path.name


def test_every_model_compares_final_train_and_held_out_test_metrics():
    for directory in ("original", "url_corpus"):
        for path in sorted((NOTEBOOK_ROOT / directory).glob("[1-5]*.ipynb")):
            _, code = notebook_source(path)
            assert "train_auc_roc = roc_auc_score" in code, path.name
            assert "train_bce = log_loss" in code, path.name
            assert "test_bce = log_loss" in code, path.name
            assert (
                code.count("plot_final_train_test_metric_groups(") == 1
            ), path.name
            assert code.count("classification_metric_sets(") == 1, path.name
            assert "evaluation_results[dataset_name]" in code, path.name
            assert "'accuracy': train_accuracy" not in code, path.name
            assert "_metrics_{group_name}.png" in code, path.name
            assert "for group_name, metric_figure" in code, path.name
            assert "plot_final_train_test_metric_figures(" not in code
            assert "final_train_test_comparison.png" not in code, path.name


def test_every_model_has_train_validation_error_by_parameter_or_epoch():
    for directory in ("original", "url_corpus"):
        for path in sorted((NOTEBOOK_ROOT / directory).glob("[1-5]*.ipynb")):
            _, code = notebook_source(path)
            assert "plot_parameter_train_validation_error(" in code, path.name
            if path.name.startswith("5_"):
                assert "epoch_end_validation_accuracy" in code, path.name
                assert "epoch_train_validation_error.png" in code, path.name
            else:
                assert "parameter_train_validation_error.png" in code, path.name
                assert "parameter_model.predict(X_val_curve)" in code, path.name
                assert "validation_errors" in code, path.name


def test_every_model_separates_metrics_time_and_feature_count_figures():
    algorithm_names = {
        "1": "logistic_regression",
        "2": "random_forest",
        "3": "xgboost",
        "4": "svm",
        "5": "mlp",
    }
    for directory in ("original", "url_corpus"):
        for path in sorted((NOTEBOOK_ROOT / directory).glob("[1-5]*.ipynb")):
            _, code = notebook_source(path)
            algorithm = algorithm_names[path.name[0]]
            assert "plot_classification_metrics(" not in code, path.name
            assert "plot_final_train_test_metric_groups(" in code, path.name
            assert "plot_training_time(" not in code, path.name
            assert "plot_training_time_comparison(" in code, path.name
            assert "plot_feature_count(" not in code, path.name
            assert "feature_count_comparison_rows(" in code, path.name
            assert "display(pd.DataFrame(feature_count_rows))" in code
            assert "remove_dataset_figure(" in code, path.name
            assert "characteristics = ['Czas uczenia [s]', 'Liczba cech']" not in code
            assert f"f'{algorithm}_metrics_{{group_name}}.png'" in code, path.name
            assert f"'{algorithm}_training_time.png'" in code, path.name
            assert (
                f"'{algorithm}_training_time_comparison.png'" in code
            ), path.name
            assert code.count(f"'{algorithm}_feature_count.png'") == 1


def test_model_summaries_do_not_print_verbose_methodology():
    for directory in ("original", "url_corpus"):
        for path in sorted((NOTEBOOK_ROOT / directory).glob("[1-5]*.ipynb")):
            _, code = notebook_source(path)
            assert "feature_count_comparison_rows," in code, path.name
            assert "training_time_measurement_summary," not in code, path.name
            assert "feature_count_measurement_summary," not in code, path.name
            assert "print(training_time_measurement_summary())" not in code
            assert "print(feature_count_measurement_summary())" not in code


def test_xgboost_saves_checkpoint_before_evaluation():
    for directory in ("original", "url_corpus"):
        path = next((NOTEBOOK_ROOT / directory).glob("3_*.ipynb"))
        _, code = notebook_source(path)
        checkpoint_position = code.index("checkpoint_data = {")
        evaluation_position = code.index("evaluation_results = {}")
        assert checkpoint_position < evaluation_position
        assert "'metrics': {}" in code
        assert "pickle.dump(checkpoint_data, file)" in code


def test_confusion_matrix_and_roc_are_separate_figures():
    for directory in ("original", "url_corpus"):
        for path in sorted((NOTEBOOK_ROOT / directory).glob("[1-5]*.ipynb")):
            _, code = notebook_source(path)
            assert "plot_confusion_and_roc" not in code, path.name
            assert "plot_confusion_matrix(" in code, path.name
            assert "plot_roc_curve(" in code, path.name
            assert "_confusion_matrix.png" in code, path.name
            assert "_roc_curve.png" in code, path.name
            assert "confusion_matrix_roc.png" not in code, path.name


def test_scientific_plot_contract_avoids_misleading_encodings():
    for path in all_notebooks():
        _, code = notebook_source(path)
        assert ".pie(" not in code, path.name
        assert "set_ylim(0.7" not in code, path.name
        assert "characteristics = ['Czas uczenia [s]', 'Liczba cech']" not in code
        assert "Czułość (recall)" not in code, path.name
        assert "Precyzja (precision)" not in code, path.name


def test_all_notebook_figures_omit_embedded_titles():
    forbidden = (".set_title(", ".suptitle(", "plt.title(", "subplot_titles=")
    for path in all_notebooks():
        _, code = notebook_source(path)
        assert not any(token in code for token in forbidden), path.name


def test_statistical_plots_are_independent_and_heatmap_is_readable():
    for directory in ("original", "url_corpus"):
        path = next((NOTEBOOK_ROOT / directory).glob("0_*.ipynb"))
        notebook, code = notebook_source(path)
        assert "np.triu(" not in code
        assert "z=matrix.values" in code
        assert "texttemplate=" not in code
        assert "top_n=10" in code
        assert "_balanced_correlation_features(" in code
        assert "zmin=-1, zmax=1, zmid=0" in code
        assert "_add_correlation_values(" in code
        assert "width=1600, height=1500" in code
        assert "correlation_heatmap.html" in code
        target_cell = next(
            cell for cell in notebook["cells"] if cell.get("id") == "5636a7f0"
        )
        target_code = "".join(target_cell["source"])
        assert "orientation='h'" in target_code
        assert "nlargest(10, 'Korelacja')" in target_code
        assert "color_continuous_scale" not in target_code
        assert "scientific_scale(" in target_code
        assert "symmetric=True" in target_code
        assert "target_intervals=4, rounded_bounds=False" in target_code
        assert "tickvals=ticks" in target_code
        importance_cell = next(
            cell for cell in notebook["cells"] if cell.get("id") == "b0e8eb8e"
        )
        importance_code = "".join(importance_cell["source"])
        assert "head(20)" in importance_code
        assert "textposition='outside'" in importance_code
        assert "scientific_scale(" in importance_code
        assert "tickformat=plotly_tick_format(ticks)" in importance_code
        assert "tickformat='.3~f'" not in importance_code
        assert "include_zero=True" in importance_code
        distribution_cell = next(
            cell for cell in notebook["cells"] if cell.get("id") == "48d9040b"
        )
        distribution_code = "".join(distribution_cell["source"])
        assert "np.quantile(combined, [0.01, 0.99])" in distribution_code
        assert "distribution_png = Path(png_dir) / 'distributions'" in distribution_code
        assert "save_bilingual_plotly_figure(" in distribution_code
        assert "figure.write_image(" not in distribution_code
        assert "width=1200, height=700" in distribution_code
        assert "_add_class_histograms(" in distribution_code
        assert "_distribution_geometry(" in distribution_code
        assert "_discrete_feature_values(" in distribution_code
        assert "unique.size <= 8" in distribution_code
        assert "bins='fd'" in distribution_code
        assert "np.clip(recommended, 3, 6)" in distribution_code
        assert "_discrete_ticks(categories)" in distribution_code
        assert "_integer_interval_edges(categories)" in distribution_code
        assert "xaxis_title=f'Wartość cechy: \"{feature_label}\"'" in (
            distribution_code
        )
        assert "_add_class_bars(" not in distribution_code
        assert "_add_histogram_bars(" in distribution_code
        assert "figure.add_bar(" in distribution_code
        assert "shape='hv'" not in distribution_code
        assert "mode='lines'" not in distribution_code
        assert "reserve_ratio=0.0" in distribution_code
        assert "target_intervals=count, rounded_bounds=True" in distribution_code
        assert "edges = scale[2]" in distribution_code
        assert "text=_percentage_labels(percentages)" in distribution_code
        assert "textposition='outside'" in distribution_code
        assert "'Przedział: [%{customdata[0]:.2~f}, '" in distribution_code
        assert "reserve_ratio=0.16" in distribution_code
        assert "lower = 0.0 if combined.min() >= 0" in distribution_code
        assert "edges, _, x_scale = _distribution_geometry" in distribution_code
        assert "scientific_scale(" in distribution_code
        assert "positions = centres + side * widths * 0.25" in distribution_code
        assert "width=widths * 0.5" in distribution_code
        assert "bargap=0.0, bargroupgap=0.0" in distribution_code
        assert "'Udział próbek klasy w pokazanym zakresie [%]', 'overlay'" in (
            distribution_code
        )
        assert "tick_step=edges[1] - edges[0]" not in distribution_code
        assert "tick_anchor=edges[0]" not in distribution_code
        assert distribution_code.count("rounded_bounds=False") >= 2
        assert "tickmode='array'" in distribution_code
        assert "tickvals=x_ticks" in distribution_code
        assert "tickformat=plotly_tick_format(x_ticks)" in distribution_code
        assert "tickvals=y_ticks" in distribution_code
        assert "_mark_x_axis_origin" not in distribution_code
        assert "linecolor='black', linewidth=3" in distribution_code
        assert "tickcolor='black', tickwidth=2" in distribution_code
        assert "title=dict(text=title" not in distribution_code
        assert "title='Częstość wartości -1" not in distribution_code
        assert "make_subplots(" not in distribution_code
        assert "orientation='h'" in distribution_code
        summary_cell = next(
            cell for cell in notebook["cells"] if cell.get("id") == "be860073"
        )
        summary_code = "".join(summary_cell["source"])
        assert "_distribution_artifacts('html')" in summary_code
        assert "correlation_heatmap.html" in summary_code
        assert "distributions_top8_features.html" not in summary_code


def test_distribution_percentages_and_labels_are_numerically_consistent():
    path = NOTEBOOK_ROOT / "original" / "0_comprehensive_statistical_analysis.ipynb"
    notebook, _ = notebook_source(path)
    cell = next(cell for cell in notebook["cells"] if cell.get("id") == "48d9040b")
    source = "".join(cell["source"])
    percentages = compiled_notebook_function(source, "_histogram_percentages")
    labels = compiled_notebook_function(source, "_percentage_labels")

    values = np.array([-100.0, 0.0, 1.0, 2.0, 100.0])
    result = percentages(values, np.array([-0.5, 0.5, 1.5, 2.5]))

    assert result.tolist() == [33.34, 33.33, 33.33]
    assert result.sum() == 100.0
    assert labels(np.array([99.97, 0.03, 0.0])) == ["99,97%", "0,03%", ""]


def test_distribution_bars_stay_between_shared_interval_boundaries():
    path = NOTEBOOK_ROOT / "original" / "0_comprehensive_statistical_analysis.ipynb"
    notebook, _ = notebook_source(path)
    cell = next(cell for cell in notebook["cells"] if cell.get("id") == "48d9040b")
    names = ("_percentage_labels", "_add_histogram_bars")
    _, add_bars = compiled_notebook_functions("".join(cell["source"]), names)
    figure, edges = Mock(), np.array([0.0, 2000.0, 4000.0])

    add_bars(figure, edges, [39.0, 61.0], "Bez phishingu", "blue", -1)
    safe = figure.add_bar.call_args.kwargs
    add_bars(figure, edges, [7.0, 93.0], "Phishing", "orange", 1)
    phishing = figure.add_bar.call_args.kwargs

    assert np.array_equal(safe["x"], [500.0, 2500.0])
    assert np.array_equal(phishing["x"], [1500.0, 3500.0])
    assert np.array_equal(safe["width"], [1000.0, 1000.0])


def test_distribution_axis_uses_only_observed_discrete_values():
    geometry = distribution_geometry_function()
    values = np.array([0, 0, 3, 3, 7, 7], dtype=float)

    _, centres, (_, _, ticks) = geometry(values, 0.0, 7.0)

    assert np.array_equal(centres, [0.0, 3.0, 7.0])
    assert np.array_equal(ticks, centres)


def test_dense_discrete_axis_uses_full_integer_interval_boundaries():
    geometry = distribution_geometry_function()
    values = np.arange(1.0, 9.0)

    edges, centres, (lower, upper, ticks) = geometry(values, 1.0, 8.0)

    assert np.array_equal(edges, np.arange(1.0, 10.0))
    assert np.array_equal(centres, np.arange(1.5, 9.0))
    assert np.array_equal(ticks, edges)
    assert (lower, upper) == (1.0, 9.0)


def test_continuous_distribution_avoids_fixed_twelve_intervals():
    geometry = distribution_geometry_function()
    values = np.linspace(0.0, 9876.0, 1001)

    edges, centres, (lower, upper, ticks) = geometry(values, 0.0, 9876.0)

    assert 3 <= len(centres) <= 6
    assert len(centres) != 12
    assert np.array_equal(ticks, edges)
    assert np.array_equal(edges, np.arange(0.0, 10001.0, 2000.0))
    assert lower == edges[0] == 0.0
    assert upper == edges[-1] == 10000.0


def test_notebook_sources_use_the_accessible_scientific_palette():
    forbidden = (
        "#0072B2", "#009E73", "#D55E00", "#CC79A7", "#E69F00",
        "#2166ac", "#b2182b", "palette=\"viridis\"", "cmap='Blues'",
    )
    for path in all_notebooks():
        _, code = notebook_source(path)
        assert not any(value in code for value in forbidden), path.name
    for path in all_notebooks():
        _, code = notebook_source(path)
        if path.name.startswith("0_"):
            assert "px.colors.sequential.Cividis" in code


def test_feature_value_axis_titles_quote_the_complete_feature_name():
    for path in all_notebooks():
        _, code = notebook_source(path)
        matching_lines = [
            line for line in code.splitlines()
            if "Wartość cechy:" in line
        ]
        assert all(
            'Wartość cechy: "{feature_label}"' in line
            for line in matching_lines
        ), path.name


def test_model_notebook_data_axes_use_shared_scientific_scaling():
    schematic_cells = {"d7c4f34d", "c3255e1c", "782e4322"}
    for path in sorted((NOTEBOOK_ROOT / "original").glob("[1-7]*.ipynb")):
        notebook, _ = notebook_source(path)
        for cell in notebook["cells"]:
            source = "".join(cell.get("source", []))
            has_limits = "set_xlim" in source or "set_ylim" in source
            has_scaling = "configure_axis_limit_ticks" in source
            has_scaling |= "configure_numeric_x_axis" in source
            assert not has_limits or has_scaling or cell.get("id") in schematic_cells


def test_phase_figures_are_saved_as_independent_scientific_charts():
    plot_code = PHASE_PLOT_PATH.read_text(encoding="utf-8")
    for directory in ("original", "url_corpus"):
        path = next((NOTEBOOK_ROOT / directory).glob("6_*.ipynb"))
        _, code = notebook_source(path)
        assert "phase_statistics.png" not in code
        assert "average_metrics.png" not in code
        assert "roc_auc_ranking.png" in code
        assert "classification_errors.png" in code
        assert "average_quality_metrics.png" in code
        assert "plot_confusion_matrix" not in code
        assert "Macierze pomyłek" not in code
        assert "validation_curve" not in code
        assert "plot_metric_heatmap" in code
    assert 'xticklabels=["Brak phishingu", "Phishing"]' in plot_code
    assert "configure_count_colorbar(axis)" in plot_code


def test_phase_notebooks_skip_only_optional_incomplete_final_comparison():
    """Preserve phase-test outputs when one model summary is not ready yet."""
    for directory in ("original", "url_corpus"):
        path = next((NOTEBOOK_ROOT / directory).glob("6_*.ipynb"))
        _, code = notebook_source(path)
        assert "missing_final_model_results" in code
        assert "if missing_comparisons:" in code
        assert "Pominięto końcowe porównanie modeli" in code


def test_curve_limits_are_scientifically_bounded():
    plot_code = PHASE_PLOT_PATH.read_text(encoding="utf-8")
    for directory in ("original", "url_corpus"):
        for path in sorted((NOTEBOOK_ROOT / directory).glob("[1-4]*.ipynb")):
            _, code = notebook_source(path)
            assert "y_limits = shared_limits" not in code, path.name
            assert "y_limits=y_limits" not in code, path.name

        mlp = next((NOTEBOOK_ROOT / directory).glob("5_*.ipynb"))
        _, mlp_code = notebook_source(mlp)
        assert "max_epochs" not in mlp_code
        assert "epoch_limits = shared_limits([epochs]" not in mlp_code
        assert (
            "configure_numeric_x_axis(axis, 'Numer epoki uczenia')"
            in mlp_code
        )
        assert "auc_limits = shared_limits([" in mlp_code
        assert "floor=0.0, ceiling=1.0" in mlp_code
        assert "loss_limits = shared_limits([" in mlp_code
        assert "loss_limits = (0.0, loss_bounds[1])" not in mlp_code

        model_test = next((NOTEBOOK_ROOT / directory).glob("6_*.ipynb"))
        _, model_test_code = notebook_source(model_test)
        assert "plot_quality_metrics_by_test_size" in model_test_code
        assert "plot_error_rate_by_test_size" in model_test_code
        assert "axis.set_ylim(bottom=0.0)" not in model_test_code
    assert "axis.set_ylim(0.0, 1.0)" in plot_code
    assert "axis.set_ylim(0.0, _percentage_axis_upper(" in plot_code
    assert "probability_limits(" not in plot_code
    assert "configure_dynamic_axis(" not in plot_code
    assert "configure_numeric_x_axis(axis" in plot_code


def test_notebooks_write_experiment_outputs_outside_data_directory():
    for path in all_notebooks():
        _, code = notebook_source(path)
        assert "data' / 'phase_testing_results" not in code
        assert "../data/phase_testing_results" not in code
        assert not any(
            f"../data/{name}_comparison.csv" in code
            for name in ("logistic_regression", "random_forest", "xgboost", "svm", "mlp")
        )

    for directory, corpus in (("original", "original"), ("url_corpus", "url_corpus")):
        for path in sorted((NOTEBOOK_ROOT / directory).glob("[1-5]*.ipynb")):
            _, code = notebook_source(path)
            assert f"'results' / '{corpus}' / 'model_comparisons'" in code

        model_test = next((NOTEBOOK_ROOT / directory).glob("6_*.ipynb"))
        _, code = notebook_source(model_test)
        assert f"'results' / '{corpus}' / 'phase_testing'" in code


def test_url_notebook_generator_is_deterministic():
    script = BACKEND_ROOT / "scripts" / "prepare_url_corpus_notebooks.py"
    subprocess.run(
        [sys.executable, str(script), "--check"],
        cwd=BACKEND_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )


def test_url_notebook_generator_clears_runtime_outputs():
    source_dir = NOTEBOOK_ROOT / "original"
    for name in SOURCE_NOTEBOOKS:
        rendered = json.loads(render_notebook(source_dir / name))
        for cell in rendered["cells"]:
            if cell.get("cell_type") == "code":
                assert cell.get("outputs", []) == []


def test_url_notebook_generator_preserves_cell_cache_by_stable_id():
    rendered = json.dumps({
        "cells": [
            {"id": "same", "cell_type": "code", "source": ["x = 1\n"], "outputs": [], "execution_count": None},
            {"id": "changed", "cell_type": "code", "source": ["y = 2\n"], "outputs": [], "execution_count": None},
        ]
    })
    existing = json.dumps({
        "cells": [
            {"id": "same", "cell_type": "code", "source": ["x = 1\n"], "outputs": [{"output_type": "stream", "name": "stdout", "text": ["cached\n"]}], "execution_count": 7},
            {"id": "changed", "cell_type": "code", "source": ["y = 1\n"], "outputs": [{"output_type": "stream", "name": "stdout", "text": ["stale\n"]}], "execution_count": 6},
        ]
    })

    merged = json.loads(preserve_matching_runtime_state(rendered, existing))
    assert merged["cells"][0]["execution_count"] == 7
    assert merged["cells"][0]["outputs"][0]["text"] == ["cached\n"]
    assert merged["cells"][1]["execution_count"] is None
    assert merged["cells"][1]["outputs"] == []


def test_url_model_notebooks_report_threshold_calibration():
    """Every URL-corpus model must preserve validation calibration evidence."""
    for path in sorted((NOTEBOOK_ROOT / "url_corpus").glob("[1-5]*.ipynb")):
        _, code = notebook_source(path)
        assert "'threshold_calibration': threshold_info" in code, path.name
        assert "Walidacyjny FPR przy wybranym progu" in code, path.name
        assert "Docelowy FPR kalibracji" in code, path.name
        marker = "from utils.threshold_calibration import ("
        assert marker in code, path.name
        imported = code.split(marker, 1)[1].split(")\n", 1)[0]
        assert "calibrate_threshold" in imported, path.name
        assert "DEFAULT_MAX_FALSE_POSITIVE_RATE" in imported, path.name
        assert "DEFAULT_MIN_RECALL" in imported, path.name


def test_url_logistic_regression_saves_threshold_in_model_payload():
    """Logistic regression must not lose its calibrated threshold on save."""
    path = next((NOTEBOOK_ROOT / "url_corpus").glob("1_*.ipynb"))
    _, code = notebook_source(path)
    assert "'model': model" in code
    assert "'scaler': scaler" in code
    assert "'decision_threshold': evaluation_results" in code
    assert "pickle.dump(model_payload, file)" in code
    assert "logistic_regression_scaler_full_url_corpus.pkl" not in code


def test_notebooks_write_experiment_outputs_only_under_results():
    for path in all_notebooks():
        _, code = notebook_source(path)
        assert "backend_root / 'static'" not in code, path.name
        assert "../static/" not in code, path.name

    for directory, corpus in (("original", "original"), ("url_corpus", "url_corpus")):
        for path in sorted((NOTEBOOK_ROOT / directory).glob("[1-5]*.ipynb")):
            _, code = notebook_source(path)
            assert f"backend_root.parent / 'results' / '{corpus}' / 'plots'" in code


def test_model_resource_monitoring_uses_disposable_training_processes():
    for directory in ("original", "url_corpus"):
        for path in sorted((NOTEBOOK_ROOT / directory).glob("[1-5]*.ipynb")):
            _, code = notebook_source(path)
            assert code.count("fit_model_isolated(") == 1, path.name
            assert "isolated_result.model" in code, path.name
            assert "isolated_result.monitor" in code, path.name
            assert "isolated_result.training_time" in code, path.name
            assert "ResourceMonitor(" not in code, path.name
            reload_position = code.index("utils.isolated_training'])")
            import_position = code.index(
                "from utils.isolated_training import fit_model_isolated"
            )
            assert reload_position < import_position, path.name


def test_model_resource_plots_explain_the_measured_baseline():
    for directory in ("original", "url_corpus"):
        for path in sorted((NOTEBOOK_ROOT / directory).glob("[1-5]*.ipynb")):
            _, code = notebook_source(path)
            assert "resource_baseline_summary," in code, path.name
            assert "print(resource_baseline_summary(monitor))" in code, path.name


def test_svm_resource_measurement_is_sequential_and_used_downstream():
    for directory in ("original", "url_corpus"):
        path = next((NOTEBOOK_ROOT / directory).glob("4_svm*.ipynb"))
        _, code = notebook_source(path)
        assert "ThreadPoolExecutor" not in code
        assert "concurrent.futures" not in code
        assert code.count("fit_model_isolated(") == 1
        assert "results_for_datasets[dataset_name].update(monitored_result)" in code
        assert "ZAKOŃCZONO SEKWENCYJNĄ OPTYMALIZACJĘ" in code


def test_logistic_architecture_uses_class_semantic_colors():
    for directory in ("original", "url_corpus"):
        path = next((NOTEBOOK_ROOT / directory).glob("1_logistic*.ipynb"))
        _, code = notebook_source(path)
        assert (
            "CLASS_COLORS['phishing'] if weight > 0 "
            "else CLASS_COLORS['safe']"
        ) in code
        assert "Waga dodatnia → phishing" in code
        assert "Waga ujemna → brak phishingu" in code


def test_binary_notebook_visuals_use_one_blue_orange_palette():
    for directory in ("original", "url_corpus"):
        statistical = next((NOTEBOOK_ROOT / directory).glob("0_*.ipynb"))
        _, statistical_code = notebook_source(statistical)
        assert "CLASS_COLORS['safe'] if value < 0" in statistical_code
        assert "CLASS_COLORS['phishing']" in statistical_code
        for prefix in ("1_", "4_", "5_"):
            path = next((NOTEBOOK_ROOT / directory).glob(f"{prefix}*.ipynb"))
            _, code = notebook_source(path)
            assert "#C98595" not in code
            assert "#DCE9E5" not in code
            assert "#6D4961" not in code


def test_url_model_notebooks_load_predefined_split_mappings():
    for path in sorted((NOTEBOOK_ROOT / "url_corpus").glob("[1-5]*.ipynb")):
        _, code = notebook_source(path)
        assert "load_split_frames" in code, path.name
        assert "datasets = split_frames" in code, path.name
        assert "train_df = frames['train']" in code, path.name
        assert "df_full = pd.read_csv" not in code, path.name
        assert "df_original = pd.read_csv" not in code, path.name


def test_url_xgboost_receives_scale_pos_weight_from_training_split():
    _, code = notebook_source(
        NOTEBOOK_ROOT
        / "url_corpus"
        / "3_xgboost_phishing_detection_url_corpus.ipynb"
    )

    assert "train_class_counts = train_df['phishing'].value_counts()" in code
    assert "if positive_count == 0:" in code
    assert "'scale_pos_weight': scale_pos_weight" in code
    assert "data_splits[dataset_name]['scale_pos_weight']" in code
