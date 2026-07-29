"""Normalize plot labels in both notebook corpora.

The script edits only code-cell source strings and keeps result caches intact.
"""

from __future__ import annotations

import json
import re
from pathlib import Path


REPLACEMENTS = {
    "Hiperpplaszczyzna": "Hiperpłaszczyzna",
    "Wyjscie w [0,1]": "Wyjście w [0,1]",
    "set_xlabel('Metryka')": "set_xlabel('Miara jakości klasyfikacji')",
    "set_ylabel('Wartość')": "set_ylabel('Wartość miary')",
    "set_xlabel('Charakterystyka')": "set_xlabel('Charakterystyka modelu')",
    "set_ylabel('Czułość (Recall)')": "set_ylabel('Czułość (recall)')",
    "set_xlabel('Czułość (Recall)')": "set_xlabel('Czułość (recall)')",
    "set_ylabel('Precyzja (Precision)')": "set_ylabel('Precyzja (precision)')",
    "set_xlabel('Rozmiar zbioru testowego')": (
        "set_xlabel('Liczba próbek w zbiorze testowym')"
    ),
    "set_ylabel('Współczynnik błędów [%]')": (
        "set_ylabel('Odsetek błędnie sklasyfikowanych próbek [%]')"
    ),
    "set_ylabel('Liczba błędów')": (
        "set_ylabel('Liczba błędnie sklasyfikowanych próbek')"
    ),
    "set_xlabel('Model')": "set_xlabel('Algorytm klasyfikacji')",
    "set_ylabel('Model')": "set_ylabel('Algorytm klasyfikacji')",
    "set_ylabel('Model i rozkład')": (
        "set_ylabel('Algorytm klasyfikacji i proporcja klas')"
    ),
    "set_xlabel('Suma błędów')": "set_xlabel('Łączna liczba błędnie sklasyfikowanych próbek')",
    "set_ylabel('Wartość metryki')": (
        "set_ylabel('Wartość miary jakości klasyfikacji')"
    ),
    "label='Walidacja'": "label='Walidacja krzyżowa (5-krotna)'",
    "label='Walidacja —": "label='Walidacja krzyżowa (5-krotna) —",
    "label='Uczenie'": "label='Uczenie modelu'",
    "label='Uczenie —": "label='Uczenie modelu —",
    "label='Uczenie modelu — binarna entropia krzyżowa'": (
        "label='Uczenie modelu'"
    ),
    "label='Zbiór walidacyjny — binarna entropia krzyżowa'": (
        "label='Zbiór walidacyjny'"
    ),
    "set_ylabel('Binarna entropia krzyżowa (BCE)')": (
        "set_ylabel('Binarna entropia krzyżowa')"
    ),
    "set_ylabel('AUC-ROC')": "set_ylabel('Pole pod krzywą ROC')",
    "set_xlabel('ROC-AUC')": "set_xlabel('Pole pod krzywą ROC')",
    "Pole pod krzywą ROC (AUC-ROC)": "Pole pod krzywą ROC",
    "Liczba próbek uczących w pięciokrotnej walidacji krzyżowej": (
        "Liczba próbek"
    ),
    "Liczba próbek uczących w przebiegu walidacji krzyżowej (5-krotnej)": (
        "Liczba próbek"
    ),
    "Liczba próbek wykorzystanych do uczenia modelu": (
        "Liczba próbek"
    ),
    "Liczba próbek użytych do uczenia w walidacji krzyżowej": (
        "Liczba próbek"
    ),
    "Liczba próbek w zbiorze uczącym": "Liczba próbek",
    "Parametr regularyzacji C": (
        "Wartość parametru C (C > 0; odwrotność siły regularyzacji)"
    ),
    "Wartość parametru C (odwrotność siły regularyzacji)": (
        "Wartość parametru C (C > 0; odwrotność siły regularyzacji)"
    ),
    "x_label='Liczba drzew'": "x_label='Liczba drzew decyzyjnych'",
    "x_label='Liczba estymatorów'": (
        "x_label='Liczba drzew decyzyjnych w modelu XGBoost'"
    ),
    "x_label='|Współczynnik|'": (
        "x_label='Wartość bezwzględna współczynnika modelu'"
    ),
    "x_label='|Wartość bezwzględna współczynnika modelu|'": (
        "x_label='Wartość bezwzględna współczynnika modelu'"
    ),
}


def notebook_paths(root: Path) -> list[Path]:
    """Return source and generated notebooks that contain experiment plots."""
    return sorted(root.glob("original/*.ipynb")) + sorted(
        root.glob("url_corpus/*.ipynb")
    )


def normalize_source(source: str) -> str:
    """Apply explicit, presentation-only label replacements."""
    source = source.replace(
        "backend_root / 'results'", "backend_root.parent / 'results'"
    )
    for normalizer in _normalization_pipeline():
        source = normalizer(source)
    return source


def _normalization_pipeline():
    """Return presentation transformations in their required order."""
    return (
        _replace_labels,
        _normalize_color_palette,
        _replace_visualization_labels,
        _normalize_model_summary_plot, _inject_feature_plot_cleanup,
        _normalize_probability_plot, _normalize_probability_examples_distribution,
        _normalize_polish_plot_language, _normalize_imbalance_plots,
        _normalize_statistical_bar_plots, _normalize_statistical_plots,
        _normalize_statistical_artifact_summary,
        _normalize_bilingual_plotly_exports,
        _normalize_tree_plots, _normalize_mlp_epoch_scales,
        _normalize_mlp_mean_sample_counts, _normalize_final_train_test_plots,
        _remove_duplicate_final_error_plot,
        _remove_duplicate_test_metric_plot, _remove_measurement_methodology_output,
        _normalize_svm_resource_measurement,
        _minimize_architecture_palette, _normalize_logistic_architecture_colors,
        normalize_resource_visualization,
    )


PLOTLY_EXPORT_REPLACEMENTS = {
    """    fig.write_html(Path(html_dir) / 'target_correlations.html')
    fig.write_image(
        Path(png_dir) / 'target_correlations.png',
        scale=2, width=1400, height=1000,
    )
    fig.show()""": """    save_bilingual_plotly_figure(
        fig, png_dir, html_dir, 'target_correlations',
        scale=2, width=1400, height=1000,
    )""",
    """    fig.write_html(Path(html_dir) / 'feature_importance.html')
    fig.write_image(
        Path(png_dir) / 'feature_importance.png',
        scale=2, width=1400, height=1000,
    )
    fig.show()""": """    save_bilingual_plotly_figure(
        fig, png_dir, html_dir, 'feature_importance',
        scale=2, width=1400, height=1000,
    )""",
    """    figure.write_html(html_output / f'{stem}.html')
    figure.write_image(png_output / f'{stem}.png', scale=2, width=1200, height=700)
    figure.show()""": """    save_bilingual_plotly_figure(
        figure, png_output, html_output, stem,
        scale=2, width=1200, height=700,
    )""",
    """    figure.write_html(Path(html_dir) / 'missing_values_minus1.html')
    figure.write_image(
        Path(png_dir) / 'missing_values_minus1.png',
        scale=2, width=1200, height=height,
    )
    figure.show()""": """    save_bilingual_plotly_figure(
        figure, png_dir, html_dir, 'missing_values_minus1',
        scale=2, width=1200, height=height,
    )""",
    """    figure.write_html(Path(html_dir) / 'correlation_heatmap.html')
    figure.write_image(
        Path(png_dir) / 'correlation_heatmap.png',
        scale=2, width=1600, height=1500,
    )
    figure.show()""": """    save_bilingual_plotly_figure(
        figure, png_dir, html_dir, 'correlation_heatmap',
        scale=2, width=1600, height=1500,
    )""",
}


def _normalize_bilingual_plotly_exports(source: str) -> str:
    """Route Plotly artifacts through the shared bilingual exporter."""
    import_line = (
        "from utils.figure_localization import "
        "save_bilingual_plotly_figure\n"
    )
    if (
        "import plotly.graph_objects as go\n" in source
        and import_line not in source
    ):
        source = source.replace(
            "import plotly.graph_objects as go\n",
            f"import plotly.graph_objects as go\n{import_line}",
        )
    for old, new in PLOTLY_EXPORT_REPLACEMENTS.items():
        source = source.replace(old, new)
    return source


COLOR_REPLACEMENTS = {
    'palette="viridis"': 'palette="colorblind"',
    "palette='viridis'": "palette='colorblind'",
    "px.colors.sequential.Viridis": "px.colors.sequential.Cividis",
    '"#0072B2"': '"#4477AA"',
    "'#0072B2'": "'#4477AA'",
    '"#009E73"': '"#228833"',
    "'#009E73'": "'#228833'",
    '"#D55E00"': '"#EE7733"',
    "'#D55E00'": "'#EE7733'",
    '"#CC79A7"': '"#AA3377"',
    "'#CC79A7'": "'#AA3377'",
    '"#E69F00"': '"#66CCEE"',
    "'#E69F00'": "'#66CCEE'",
    "'#2ecc71'": "'#4477AA'",
    "'#e74c3c'": "'#EE7733'",
    "'#3498db'": "'#4477AA'",
    "'#f39c12'": "'#EE7733'",
    "'#d35400'": "'#EE7733'",
    "'#4CAF50'": "'#4477AA'",
    "'#F44336'": "'#EE7733'",
    "'#2196F3'": "'#4477AA'",
    "'#FFD700'": "'#EE7733'",
    "'#FFA500'": "'#EE7733'",
    "'#FFE5E5'": "'#F5E9D8'",
    "'#E3F2FD'": "'#DCE6F2'",
    "'#2166ac'": "'#4477AA'",
    "'#b2182b'": "'#EE7733'",
    "'#053061'": "'#2F5F8F'",
    "'#4393c3'": "'#A9C2D8'",
    "'#d6604d'": "'#F2B38C'",
    "'#67001f'": "'#B84E16'",
    "facecolor='lightblue'": "facecolor='#DCE6F2'",
    "facecolor='lightyellow'": "facecolor='#F5E9D8'",
    "facecolor='yellow'": "facecolor='#E8B56A'",
    "facecolor='lightcoral'": "facecolor='#F5E9D8'",
    "facecolor='coral'": "facecolor='#EE7733'",
    "facecolor='lightgreen'": "facecolor='#DCE6F2'",
    "color='darkgreen'": "color='#4477AA'",
    "color='darkred'": "color='#EE7733'",
    "color='blue'": "color='#4477AA'",
    "'#DCE9E5'": "'#DCE6F2'",
    "'#EADCE2'": "'#F5E9D8'",
    "'#E8B56A'": "'#EE7733'",
    "'#C98595'": "'#EE7733'",
    "'#356B62'": "'#4477AA'",
    "'#6D4961'": "'#EE7733'",
}


def _normalize_color_palette(source: str) -> str:
    """Apply one neutral, color-vision-safe scientific palette."""
    for old, new in COLOR_REPLACEMENTS.items():
        source = source.replace(old, new)
    return source


def _normalize_svm_resource_measurement(source: str) -> str:
    """Keep SVM optimization and final resource measurement sequential."""
    source = source.replace(
        'print(f"\n{\'=\'*80}")', 'print(f"\\n{\'=\'*80}")'
    )
    if "def train_model_for_dataset" in source:
        source = _remove_parallel_svm_resource_measurement(source)
        source = _replace_parallel_svm_optimization(source)
    if "results_with_monitoring = {}" in source:
        source = _merge_svm_monitored_results(source)
    return source


def _minimize_architecture_palette(source: str) -> str:
    """Limit model diagrams to the shared primary and secondary colors."""
    replacements = {
        "layer_color = '#AA3377'": "layer_color = '#4477AA'",
        "layer_color = '#66CCEE'": "layer_color = '#EE7733'",
        "func_color = '#EADCE2'": "func_color = '#F5E9D8'",
        "func_edge = '#8B0000'": "func_edge = '#EE7733'",
        "func_edge = '#00008B'": "func_edge = '#4477AA'",
        "color='#AA3377'": "color='#EE7733'",
    }
    for old, new in replacements.items():
        source = source.replace(old, new)
    return source


def _normalize_logistic_architecture_colors(source: str) -> str:
    """Align logistic coefficient colors with the safe/phishing class map."""
    source = source.replace(
        "color = '#4477AA' if weight > 0 else '#EE7733'",
        "color = CLASS_COLORS['phishing'] if weight > 0 "
        "else CLASS_COLORS['safe']",
    )
    source = source.replace(
        "mpatches.Patch(facecolor='#4477AA', alpha=0.8, "
        "label='Waga dodatnia -> Phishing')",
        "mpatches.Patch(facecolor=CLASS_COLORS['phishing'], alpha=0.8, "
        "label='Waga dodatnia → phishing')",
    )
    return source.replace(
        "mpatches.Patch(facecolor='#EE7733', alpha=0.8, "
        "label='Waga ujemna -> Brak phishingu')",
        "mpatches.Patch(facecolor=CLASS_COLORS['safe'], alpha=0.8, "
        "label='Waga ujemna → brak phishingu')",
    )


def _remove_parallel_svm_resource_measurement(source: str) -> str:
    """Remove the redundant resource trace collected during parallel search."""
    pattern = re.compile(
        r'    print\(f"\[\{threading\.current_thread\(\)\.name\}\] '
        r'Trenowanie finalnego modelu\.\.\."\)\n'
        r".*?    stats = resource_monitor\.get_stats\(\)\n",
        re.DOTALL,
    )
    source = pattern.sub("", source, count=1)
    source = source.replace("        'resource_monitor': resource_monitor,\n", "")
    return source.replace("        'resource_stats': stats,\n", "")


def _replace_parallel_svm_optimization(source: str) -> str:
    """Run both SVM optimization modes one after another."""
    pattern = re.compile(
        r"results_for_datasets = \{\}\n"
        r"dataset_names = .*?\Z",
        re.DOTALL,
    )
    if "ThreadPoolExecutor" not in source:
        return source
    return pattern.sub(lambda _match: SVM_SEQUENTIAL_OPTIMIZATION, source, count=1)


SVM_SEQUENTIAL_OPTIMIZATION = """results_for_datasets = {}
dataset_names = ['Pełny zbiór', 'Zoptymalizowany zbiór']

for dataset_name in dataset_names:
    name, result_data = train_model_for_dataset(dataset_name)
    results_for_datasets[name] = result_data

print(f"\\n{'='*80}")
print("ZAKOŃCZONO SEKWENCYJNĄ OPTYMALIZACJĘ MODELI DLA OBU ZBIORÓW")
print(f"{'='*80}")
"""


def _merge_svm_monitored_results(source: str) -> str:
    """Make all downstream SVM cells use independently measured models."""
    marker = "results_for_datasets[dataset_name].update(monitored_result)"
    if marker in source:
        return source
    update = (
        "\nfor dataset_name, monitored_result in "
        "results_with_monitoring.items():\n"
        f"    {marker}\n"
    )
    return source.rstrip() + "\n" + update


def _replace_labels(source: str) -> str:
    """Replace ambiguous Polish labels with precise experiment terminology."""
    for old, new in REPLACEMENTS.items():
        source = source.replace(old, new)
    return source


def _normalize_statistical_bar_plots(source: str) -> str:
    """Replace dense notebook-0 bars with readable horizontal figures."""
    if "def analyze_correlations" in source:
        source = _replace_target_correlation_bars(source)
    if "def calculate_feature_importance" in source:
        source = _replace_feature_importance_bars(source)
    return _remove_statistical_plot_titles(source)


def _remove_statistical_plot_titles(source: str) -> str:
    """Remove titles that belong in thesis captions below exported figures."""
    titles = (
        "        title='Najsilniejsze dodatnie i ujemne korelacje z klasą docelową',\n",
        "        title='Dwadzieścia najważniejszych cech według modelu Random Forest',\n",
    )
    for title in titles:
        source = source.replace(title, "")
    return source


def _replace_target_correlation_bars(source: str) -> str:
    """Use balanced horizontal bars for target correlations."""
    pattern = re.compile(
        r"    (?:top_30_data =|positive = target_correlations\[)"
        r".*?    fig\.show\(\)\n",
        re.DOTALL,
    )
    return pattern.sub(TARGET_CORRELATION_BARS, source, count=1)


TARGET_CORRELATION_BARS = """    positive = target_correlations[
        target_correlations['Korelacja'] > 0
    ].nlargest(10, 'Korelacja')
    negative = target_correlations[
        target_correlations['Korelacja'] < 0
    ].nsmallest(10, 'Korelacja')
    displayed = pd.concat([negative, positive]).sort_values('Korelacja')
    labels = [get_readable_name(feature) for feature in displayed.index]
    values = displayed['Korelacja'].to_numpy()
    colors = [
        CLASS_COLORS['safe'] if value < 0 else CLASS_COLORS['phishing']
        for value in values
    ]
    value_labels = [f'{value:.2f}'.replace('.', ',') for value in values]
    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation='h',
        marker=dict(color=colors),
        text=value_labels, textposition='outside', cliponaxis=False,
        hovertemplate=(
            '%{y}<br>Współczynnik korelacji Pearsona: '
            '%{x:.3f}<extra></extra>'
        ),
    ))
    lower, upper, ticks = scientific_scale(
        values, symmetric=True, reserve_ratio=0.08,
        target_intervals=4, rounded_bounds=False,
    )
    fig.update_layout(
        width=1400, height=1000,
        xaxis_title='Współczynnik korelacji Pearsona',
        yaxis_title='Cecha', showlegend=False,
        plot_bgcolor='white', separators=', ',
        margin=dict(l=360, r=100, t=40, b=90),
    )
    fig.update_xaxes(
        range=[lower, upper], tickmode='array', tickvals=ticks,
        zeroline=True, zerolinewidth=2,
        zerolinecolor='black', showgrid=True, gridcolor='lightgray',
        tickformat=plotly_tick_format(ticks),
    )
    fig.update_yaxes(showgrid=False)
    fig.write_html(Path(html_dir) / 'target_correlations.html')
    fig.write_image(
        Path(png_dir) / 'target_correlations.png',
        scale=2, width=1400, height=1000,
    )
    fig.show()
"""


def _replace_feature_importance_bars(source: str) -> str:
    """Limit feature-importance bars and label their values directly."""
    pattern = re.compile(
        r"    (?:top_30 =|displayed = feature_importance_df\.head)"
        r".*?    fig\.show\(\)\n",
        re.DOTALL,
    )
    return pattern.sub(FEATURE_IMPORTANCE_BARS, source, count=1)


FEATURE_IMPORTANCE_BARS = """    displayed = feature_importance_df.head(20).sort_values('Ważność')
    labels = [get_readable_name(feature) for feature in displayed['Cecha']]
    values = displayed['Ważność'].to_numpy()
    value_labels = [f'{value:.3f}'.replace('.', ',') for value in values]
    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation='h',
        marker=dict(color=CLASS_COLORS['safe']),
        text=value_labels, textposition='outside', cliponaxis=False,
        hovertemplate='%{y}<br>Ważność cechy: %{x:.4f}<extra></extra>',
    ))
    lower, upper, ticks = scientific_scale(
        values, include_zero=True, reserve_ratio=0.08,
        target_intervals=5, rounded_bounds=False,
    )
    fig.update_layout(
        width=1400, height=1000,
        xaxis_title='Ważność cechy',
        yaxis_title='Cecha', showlegend=False,
        plot_bgcolor='white', separators=', ',
        margin=dict(l=360, r=100, t=40, b=90),
    )
    fig.update_xaxes(
        range=[lower, upper], tickmode='array', tickvals=ticks,
        showgrid=True, gridcolor='lightgray', zeroline=False,
        tickformat=plotly_tick_format(ticks),
    )
    fig.update_yaxes(showgrid=False)
    fig.write_html(Path(html_dir) / 'feature_importance.html')
    fig.write_image(
        Path(png_dir) / 'feature_importance.png',
        scale=2, width=1400, height=1000,
    )
    fig.show()
"""


def _inject_scientific_plot_helpers(source: str) -> str:
    """Import shared helpers needed by normalized model figures."""
    names = (
        "classification_metric_sets", "feature_count_comparison_rows",
        "plot_final_train_test_metric_groups",
        "plot_sample_probability_prediction",
        "plot_training_time_comparison",
        "remove_dataset_figure", "resource_baseline_summary",
        "save_comparison_figure",
    )
    return _inject_plot_import_names(source, names)


def _inject_statistical_plot_helpers(source: str) -> str:
    """Import shared scientific scales needed by notebook-zero figures."""
    return _inject_plot_import_names(
        source, ("plotly_tick_format", "scientific_scale")
    )


def _inject_plot_import_names(source: str, names) -> str:
    """Add selected names to the shared plotting import block."""
    marker = "from utils.notebook_plot_style import ("
    if marker not in source:
        return source
    missing = [name for name in names if f"    {name},\n" not in source]
    if not missing:
        return source
    lines = "".join(f"    {name},\n" for name in missing)
    return source.replace(marker + "\n", marker + "\n" + lines, 1)


def _normalize_model_summary_plot(source: str) -> str:
    """Keep timing as a chart and present feature counts as a table."""
    if "plot_feature_count(" not in source and "plot_training_time(" not in source:
        return source
    filename = re.search(r"'([a-z_]+)_training_time\.png'", source)
    if filename is None:
        filename = re.search(r"'([a-z_]+)_metrics_comparison\.png'", source)
    if filename is None:
        return source
    return _model_summary_source(filename.group(1), source)


MODEL_SUMMARY_TEMPLATE = """training_times = {}
feature_counts = {}
for dataset_name in ['Pełny zbiór', 'Zoptymalizowany zbiór']:
    training_times[dataset_name] = float($TRAINING$)
    feature_counts[dataset_name] = int($FEATURES$)
    remove_dataset_figure(
        PLOTS_DIR, dataset_name, '$MODEL_PREFIX$_training_time.png'
    )
    remove_dataset_figure(
        PLOTS_DIR, dataset_name, '$MODEL_PREFIX$_feature_count.png'
    )

time_figure, _ = plot_training_time_comparison(training_times)
save_comparison_figure(
    time_figure, PLOTS_DIR / '$MODEL_PREFIX$_training_time_comparison.png'
)
plt.show()

feature_count_rows = feature_count_comparison_rows(
    feature_counts['Pełny zbiór'],
    feature_counts['Zoptymalizowany zbiór'],
)
display(pd.DataFrame(feature_count_rows))
"""


def _model_summary_source(model_prefix: str, source: str) -> str:
    """Build canonical source for time and feature-count summary charts."""
    training, features = _summary_expressions(source)
    return (
        MODEL_SUMMARY_TEMPLATE
        .replace("$MODEL_PREFIX$", model_prefix)
        .replace("$TRAINING$", training)
        .replace("$FEATURES$", features)
    )


def _remove_measurement_methodology_output(source: str) -> str:
    """Remove verbose methodology text printed beside concise summaries."""
    description = (
        "print(training_time_measurement_summary())\n"
        "print(feature_count_measurement_summary())\n\n"
    )
    return source.replace(description, "", 1)


def _inject_feature_plot_cleanup(source: str) -> str:
    """Remove obsolete feature-count figures when rebuilding the summary."""
    if "feature_count_rows =" not in source:
        return source
    if "remove_dataset_figure(" in source:
        return source
    match = re.search(r"'([a-z_]+)_training_time\.png'", source)
    if match is None:
        return source
    old = "    plt.show()\n\n    feature_counts[dataset_name]"
    cleanup = (
        "    plt.show()\n"
        "    remove_dataset_figure(\n"
        f"        PLOTS_DIR, dataset_name, '{match.group(1)}_feature_count.png'\n"
        "    )\n\n"
        "    feature_counts[dataset_name]"
    )
    return source.replace(old, cleanup, 1)


def _summary_expressions(source: str) -> tuple[str, str]:
    """Return training-time and feature-count expressions for a model cell."""
    if "results_for_datasets[dataset_name]['training_time']" in source:
        return (
            "results_for_datasets[dataset_name]['training_time']",
            "len(data_splits[dataset_name]['X'].columns)",
        )
    return (
        "trained_models[dataset_name]['training_time']",
        "data_splits[dataset_name]['feature_count']",
    )


def _normalize_probability_plot(source: str) -> str:
    """Replace probability pie charts with fixed-scale horizontal bars."""
    if "axis.pie(" not in source or "Etykieta rzeczywista" not in source:
        return source
    probability = _probability_expression(source)
    pattern = re.compile(
        r"        axis\.pie\(\n.*?        axis\.text\(\n.*?        \)\n",
        re.DOTALL,
    )
    replacement = (
        "        plot_probability_bars(\n"
        f"            axis, {probability},\n"
        "            sample_label=(\n"
        "                f'Próbka #{sample_index}; Etykieta rzeczywista: '\n"
        "                f'{true_label_name} ({true_label})'\n"
        "            ),\n"
        "        )\n"
    )
    return pattern.sub(replacement, source, count=1)


def _normalize_probability_examples_distribution(source: str) -> str:
    """Create separate one-sample probability figures for both true classes."""
    legacy = "sample_specs =" in source and "plot_probability_figure(" in source
    distribution = "plot_probability_distribution(" in source
    monolingual = (
        "plot_sample_probability_prediction(" in source
        and "language=language" not in source
    )
    flat_bilingual = "_{language}.png" in source
    folder_bilingual = "f'{language}/" in source
    if not any((legacy, distribution, monolingual, flat_bilingual, folder_bilingual)):
        return source
    spec = _probability_distribution_spec(source)
    if spec is None:
        return source
    return _probability_distribution_source(*spec)


PROBABILITY_EXAMPLE_SPECS = (
    ("logistic_regression", "train_test_data[dataset_name]['y_test']", "decision"),
    ("random_forest", "train_test_data[dataset_name]['y_test']", "voting"),
    ("xgboost", "results_for_datasets[dataset_name]['y_test']", "prediction"),
    ("svm", "data_splits[dataset_name]['y_test']", "prediction"),
    ("mlp", "data_splits[dataset_name]['y_test']", "prediction"),
)


def _probability_distribution_spec(source: str):
    """Return model prefix, test labels and obsolete example stem."""
    for prefix, labels, stem in PROBABILITY_EXAMPLE_SPECS:
        markers = (
            f"{prefix}_{stem}_example_",
            f"{prefix}_probability_distribution.png",
            f"{prefix}_sample_prediction_",
        )
        marker = next((item for item in markers if item in source), None)
        if marker is not None:
            return prefix, labels, stem
    return None


PROBABILITY_DISTRIBUTION_TEMPLATE = """for dataset_name in ['Pełny zbiór', 'Zoptymalizowany zbiór']:
    true_labels = $LABELS$
    label_values = np.asarray(true_labels, dtype=int)
    probabilities = np.asarray(
        evaluation_results[dataset_name]['y_pred_proba'], dtype=float
    ).reshape(-1)
    threshold = evaluation_results[dataset_name].get('decision_threshold', 0.5)
    sample_positions = {
        'phishing': int(np.flatnonzero(label_values == 1)[0]),
        'bez_phishingu': int(np.flatnonzero(label_values == 0)[0]),
    }
    for sample_kind, position in sample_positions.items():
        sample_id = (
            true_labels.index[position]
            if hasattr(true_labels, 'index') else position
        )
        for language in ['pl', 'en']:
            language_folder = 'pl' if language == 'pl' else 'eng'
            figure, _ = plot_sample_probability_prediction(
                probabilities[position], label_values[position], sample_id,
                decision_threshold=threshold, language=language,
            )
            save_dataset_figure(
                figure, PLOTS_DIR, dataset_name,
                f'{language_folder}/$PREFIX$_sample_prediction_{sample_kind}.png',
            )
            plt.show()
    for obsolete_name in [
        '$PREFIX$_$STEM$_example_phishing.png',
        '$PREFIX$_$STEM$_example_bez_phishingu.png',
        '$PREFIX$_$STEM$_examples.png',
        '$PREFIX$_probability_distribution.png',
        '$PREFIX$_sample_prediction_phishing.png',
        '$PREFIX$_sample_prediction_bez_phishingu.png',
        '$PREFIX$_sample_prediction_phishing_pl.png',
        '$PREFIX$_sample_prediction_phishing_en.png',
        '$PREFIX$_sample_prediction_bez_phishingu_pl.png',
        '$PREFIX$_sample_prediction_bez_phishingu_en.png',
    ]:
        remove_dataset_figure(PLOTS_DIR, dataset_name, obsolete_name)
"""


def _probability_distribution_source(prefix, labels, old_stem) -> str:
    """Return one representative probability-distribution figure."""
    return (
        PROBABILITY_DISTRIBUTION_TEMPLATE
        .replace("$PREFIX$", prefix)
        .replace("$LABELS$", labels)
        .replace("$STEM$", old_stem)
    )


def _probability_expression(source: str) -> str:
    """Return the class-score expression used by a prediction example."""
    if "votes = np.asarray" in source:
        return "[np.sum(votes == 0), np.sum(votes == 1)]"
    if "phishing_probability = float(" in source:
        return "[1 - phishing_probability, phishing_probability]"
    return "probability"


def _normalize_polish_plot_language(source: str) -> str:
    """Remove mixed Polish-English metric labels from plot presentation."""
    replacements = {
        "Czułość (recall)": "Czułość",
        "Czułość (Recall)": "Czułość",
        "Precyzja (precision)": "Precyzja",
        "Precyzja (Precision)": "Precyzja",
        "syntetyczne_nadprobkowanie": "syntetyczne nadpróbkowanie",
        "niedoprobkowanie": "niedopróbkowanie",
        "nadprobkowanie": "nadpróbkowanie",
        "Bezpieczne": "Brak phishingu",
        "; etykieta rzeczywista: ": "; Etykieta rzeczywista: ",
    }
    for old, new in replacements.items():
        source = source.replace(old, new)
    return source


def _normalize_mlp_mean_sample_counts(source: str) -> str:
    """Store averaged cross-validation fold sizes as whole sample counts."""
    if "mean_learning_train_sizes = np.rint(" in source:
        return source
    expression = "learning_train_sizes.mean(axis=1)"
    if expression not in source:
        return source
    updated = source.replace(expression, "mean_learning_train_sizes")
    marker = "    cv_scores = learning_validation_scores[-1]\n"
    definition = (
        marker
        + "    mean_learning_train_sizes = np.rint(\n"
        + "        learning_train_sizes.mean(axis=1)\n"
        + "    ).astype(int)\n"
    )
    return updated.replace(marker, definition, 1)


def _normalize_imbalance_plots(source: str) -> str:
    """Apply fixed scales, semantic classes and Polish numeric notation."""
    source = source.replace(
        "f'{count:,}'",
        "f'{int(count):,}'.replace(',', ' ')",
    ).replace(
        "f'{percentage:.2f}%'",
        "f\"{percentage:.2f}\".replace('.', ',') + '%'",
    )
    if "precision_recall_curves.png" in source:
        source = _normalize_precision_recall_cell(source)
    if "sns.heatmap(matrix" in source:
        source = source.replace(
            "cmap='cividis', ax=axis, cbar=False)",
            "cmap='cividis', ax=axis, cbar=False,\n"
            "                xticklabels=['Brak phishingu', 'Phishing'],\n"
            "                yticklabels=['Brak phishingu', 'Phishing'])",
        )
    return _remove_imbalance_plot_titles(source)


def _remove_imbalance_plot_titles(source: str) -> str:
    """Remove titles that are supplied as captions in the thesis."""
    lines = (
        "axis.set_title('Rozkład klas w zbiorze uczącym — liczebność')\n",
        "axis.set_title('Rozkład klas w zbiorze uczącym — udział procentowy')\n",
        "    axis.set_title(f'Macierz pomyłek — {variant}')\n",
    )
    for line in lines:
        source = source.replace(line, "")
    return source


PRECISION_RECALL_AXIS_SOURCE = """    axis.set_ylabel('Precyzja')
    baseline = float((y_test == minority_label).mean())
    axis.axhline(
        baseline, color='#555555', linestyle='--',
        label=f'Poziom bazowy={baseline:.3f}',
    )
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    configure_axis_limit_ticks(axis, 'x')
    configure_axis_limit_ticks(axis, 'y')
"""


def _normalize_precision_recall_cell(source: str) -> str:
    """Configure precision-recall curves on the conventional [0, 1] scale."""
    source = source.replace(
        "label=f'{variant} (AP={auc_val:.3f})'",
        "label=f'{variant} (średnia precyzja={auc_val:.3f})'",
    )
    if "baseline = float((y_test == minority_label).mean())" in source:
        return source
    marker = "    axis.set_ylabel('Precyzja')\n"
    return source.replace(marker, PRECISION_RECALL_AXIS_SOURCE, 1)


def _normalize_statistical_plots(source: str) -> str:
    """Improve density and remove redundant correlation annotations."""
    if "def analyze_distributions" in source:
        return DISTRIBUTION_ANALYSIS_SOURCE
    if "def create_correlation_heatmap" in source:
        return _correlation_heatmap_source()
    return source


def _normalize_statistical_artifact_summary(source: str) -> str:
    """Replace the obsolete combined-distribution artifact checklist."""
    if "PODSUMOWANIE WSZYSTKICH ZAPISANYCH PLIKÓW" not in source:
        return source
    return STATISTICAL_ARTIFACT_SUMMARY_SOURCE


STATISTICAL_ARTIFACT_SUMMARY_SOURCE = """from pathlib import Path


def _file_size_label(path):
    '''Return a readable file-size label.'''
    size = path.stat().st_size
    if size > 1024 ** 2:
        return f'{size / 1024 ** 2:.2f} MB'
    if size > 1024:
        return f'{size / 1024:.2f} KB'
    return f'{size} B'


def _print_artifact(path):
    '''Print the existence and size of one expected artifact.'''
    if path.exists():
        print(f'  ✅ {path} ({_file_size_label(path)})')
    else:
        print(f'  ❌ {path} (nie znaleziono)')


def _distribution_artifacts(extension):
    '''Return independently generated feature-distribution figures.'''
    root = Path(png_dir if extension == 'png' else html_dir) / 'distributions'
    return sorted(root.glob(f'*.{extension}'))


saved_files = {
    'Zbiory danych': [
        Path('../data/dataset_full.csv'),
        Path('../data/dataset_optimized.csv'),
    ],
    'Raporty CSV': [
        Path(csv_dir) / 'descriptive_statistics.csv',
        Path(csv_dir) / 'target_correlations.csv',
        Path(csv_dir) / 'feature_importance.csv',
        Path(csv_dir) / 'distribution_analysis.csv',
        Path(csv_dir) / 'missing_values_minus1.csv',
        Path(csv_dir) / 'features_to_eliminate.csv',
        Path(csv_dir) / 'features_to_keep.csv',
    ],
    'Wizualizacje PNG': [
        Path(png_dir) / 'target_correlations.png',
        Path(png_dir) / 'feature_importance.png',
        *_distribution_artifacts('png'),
        Path(png_dir) / 'missing_values_minus1.png',
        Path(png_dir) / 'correlation_heatmap.png',
    ],
    'Wizualizacje HTML': [
        Path(html_dir) / 'target_correlations.html',
        Path(html_dir) / 'feature_importance.html',
        *_distribution_artifacts('html'),
        Path(html_dir) / 'missing_values_minus1.html',
        Path(html_dir) / 'correlation_heatmap.html',
    ],
    'Dokumentacja': [Path(docs_dir) / 'analysis_summary.md'],
}

print('=' * 80)
print('PODSUMOWANIE WSZYSTKICH ZAPISANYCH PLIKÓW')
print('=' * 80)
for category, paths in saved_files.items():
    print(f'\\n{category}:')
    for path in paths:
        _print_artifact(path)

png_count = len(_distribution_artifacts('png'))
html_count = len(_distribution_artifacts('html'))
print(f'\\nOsobne rozkłady cech: PNG={png_count}/8, HTML={html_count}/8')
if png_count != 8 or html_count != 8:
    print('⚠️ Oczekiwano po osiem osobnych wykresów rozkładów.')

print('\\n' + '=' * 80)
print(f'Analiza statystyczna: {output_dir}')
print('✅ Analiza zakończona pomyślnie.')
"""


DISTRIBUTION_ANALYSIS_SOURCE = """def _normality_record(df, column, sample_size):
    '''Return the Shapiro-Wilk result after excluding missing feature values.'''
    values = df[column].replace(-1, np.nan).dropna()
    sample = values.sample(min(len(values), sample_size), random_state=42)
    if len(sample) < 3:
        statistic, p_value = np.nan, np.nan
    else:
        statistic, p_value = stats.shapiro(sample)
    return {
        'Cecha': column,
        'Statystyka Shapiro-Wilk': statistic,
        'wartość-p': p_value,
        'Rozkład Normalny': bool(p_value > 0.05) if pd.notna(p_value) else False,
    }


def _class_feature_values(df, feature):
    '''Return valid feature values for both target classes.'''
    safe = df.loc[df[TARGET_COL] == 0, feature]
    phishing = df.loc[df[TARGET_COL] == 1, feature]
    return safe[safe != -1].dropna().values, phishing[phishing != -1].dropna().values


def _feature_file_stem(position, feature):
    '''Build a stable filesystem-safe plot name.'''
    safe_name = ''.join(character if character.isalnum() else '_' for character in feature)
    return f'{position:02d}_{safe_name.strip("_")}'


def _new_distribution_figure():
    '''Create an empty feature-distribution figure.'''
    return go.Figure()


def _robust_feature_range(safe_values, phishing_values):
    '''Return a robust display range and the combined source values.'''
    combined = np.concatenate([safe_values, phishing_values])
    quantile_lower, upper = np.quantile(combined, [0.01, 0.99])
    lower = 0.0 if combined.min() >= 0 else quantile_lower
    if lower == upper:
        upper = lower + 1.0
    return combined, float(lower), float(upper)


def _discrete_feature_values(values):
    '''Return observed discrete categories when their axis remains readable.'''
    unique = np.unique(values)
    integers = np.isclose(unique, np.round(unique)).all()
    return unique if integers and unique.size <= 8 else None


def _discrete_edges(categories):
    '''Build category boundaries without adding unobserved integer ticks.'''
    if categories.size == 1:
        return np.array([categories[0] - 0.5, categories[0] + 0.5])
    midpoints = (categories[:-1] + categories[1:]) / 2
    first = categories[0] - (midpoints[0] - categories[0])
    last = categories[-1] + (categories[-1] - midpoints[-1])
    return np.concatenate(([first], midpoints, [last]))


def _adaptive_bin_count(values):
    '''Choose a data-dependent readable bin count using Freedman-Diaconis.'''
    recommended = len(np.histogram_bin_edges(values, bins='fd')) - 1
    if recommended <= 1:
        recommended = len(np.histogram_bin_edges(values, bins='sturges')) - 1
    return int(np.clip(recommended, 3, 6))


def _discrete_ticks(categories):
    '''Limit labels while preserving every observed discrete-value bar.'''
    if categories.size <= 6:
        return categories
    indices = np.linspace(0, categories.size - 1, 5).round().astype(int)
    return categories[np.unique(indices)]


def _integer_interval_edges(categories):
    '''Return full integer boundaries for consecutive count values.'''
    if not np.isclose(np.diff(categories), 1.0).all():
        return None
    return np.arange(categories[0], categories[-1] + 2.0)


def _continuous_geometry(visible, lower, upper):
    '''Return adaptive continuous bins, centres, limits and ticks.'''
    count = _adaptive_bin_count(visible)
    scale = scientific_scale(
        [lower, upper], include_zero=lower >= 0.0, reserve_ratio=0.0,
        target_intervals=count, rounded_bounds=True,
    )
    edges = scale[2]
    centres = (edges[:-1] + edges[1:]) / 2
    return edges, centres, scale


def _distribution_geometry(combined, lower, upper):
    '''Build an axis individually for the observed feature distribution.'''
    visible = combined[(combined >= lower) & (combined <= upper)]
    categories = _discrete_feature_values(visible)
    if categories is None:
        return _continuous_geometry(visible, lower, upper)
    integer_edges = _integer_interval_edges(categories)
    if integer_edges is not None:
        centres = (integer_edges[:-1] + integer_edges[1:]) / 2
        scale = (float(integer_edges[0]), float(integer_edges[-1]), integer_edges)
        return integer_edges, centres, scale
    edges = _discrete_edges(categories)
    scale = (float(edges[0]), float(edges[-1]), _discrete_ticks(categories))
    return edges, categories, scale


def _visible_values(values, lower, upper):
    '''Select observations visible in a robust histogram range.'''
    return values[(values >= lower) & (values <= upper)]


def _histogram_percentages(values, edges):
    '''Calculate percentages that sum to exactly 100.00 in the visible range.'''
    counts, _ = np.histogram(values, bins=edges)
    if counts.sum() == 0:
        return np.zeros(len(edges) - 1)
    raw_units = counts / counts.sum() * 10000
    units = np.floor(raw_units).astype(int)
    remainder = 10000 - int(units.sum())
    order = np.argsort(-(raw_units - units), kind='stable')
    units[order[:remainder]] += 1
    return units / 100


def _add_class_histograms(figure, safe_values, phishing_values, edges):
    '''Add grouped class histograms using the same numeric intervals.'''
    safe_percent = _histogram_percentages(safe_values, edges)
    phishing_percent = _histogram_percentages(phishing_values, edges)
    _add_histogram_bars(
        figure, edges, safe_percent,
        'Brak phishingu', CLASS_COLORS['safe'], -1,
    )
    _add_histogram_bars(
        figure, edges, phishing_percent,
        'Phishing', CLASS_COLORS['phishing'], 1,
    )
    safe_visible = _visible_values(safe_values, edges[0], edges[-1])
    phishing_visible = _visible_values(phishing_values, edges[0], edges[-1])
    excluded = len(safe_values) + len(phishing_values)
    excluded -= len(safe_visible) + len(phishing_visible)
    maximum = max(safe_percent.max(), phishing_percent.max(), 0.0)
    return excluded, float(maximum)


def _percentage_labels(percentages):
    '''Format non-zero percentages with consistent Polish precision.'''
    return [
        '' if value == 0 else f'{value:.2f}%'.replace('.', ',')
        for value in percentages
    ]


def _add_histogram_bars(figure, edges, percentages, label, color, side):
    '''Place one class in its half of every shared histogram interval.'''
    widths = np.diff(edges)
    centres = (edges[:-1] + edges[1:]) / 2
    positions = centres + side * widths * 0.25
    intervals = np.column_stack((edges[:-1], edges[1:]))
    figure.add_bar(
        x=positions, y=percentages, width=widths * 0.5, name=label,
        marker=dict(color=color, line=dict(color='black', width=0.8)),
        customdata=intervals, text=_percentage_labels(percentages),
        textposition='outside', cliponaxis=False,
        hovertemplate=(
            'Przedział: [%{customdata[0]:.2~f}, '
            '%{customdata[1]:.2~f})<br>'
            'Udział próbek klasy: '
            '%{y:.2f}%<extra></extra>'
        ),
    )


def _style_distribution_figure(figure, feature_label, y_title, barmode):
    '''Apply a readable scientific layout to one distribution figure.'''
    figure.update_layout(
        width=1200, height=700, barmode=barmode,
        xaxis_title=f'Wartość cechy: "{feature_label}"', yaxis_title=y_title,
        plot_bgcolor='white', font=dict(size=16),
        legend=dict(
            orientation='h', x=0.5, xanchor='center',
            y=1.0, yanchor='bottom',
        ),
        margin=dict(l=110, r=60, t=120, b=110),
        bargap=0.0, bargroupgap=0.0, separators=', ',
    )
    figure.update_xaxes(
        showgrid=True, gridcolor='lightgray', zeroline=False,
        automargin=True, showline=True,
        linecolor='black', linewidth=3, ticks='outside',
        tickcolor='black', tickwidth=2, ticklabeloverflow='allow',
        ticklabelstandoff=8,
    )
    figure.update_yaxes(
        showgrid=True, gridcolor='lightgray',
        rangemode='tozero', tickformat='.1f', automargin=True,
        showline=True, linecolor='black', linewidth=3,
        ticks='outside', tickcolor='black', tickwidth=2,
    )


def _save_distribution_figure(figure, position, feature, png_output, html_output):
    '''Save one distribution as independent raster and interactive files.'''
    stem = _feature_file_stem(position, feature)
    figure.write_html(html_output / f'{stem}.html')
    figure.write_image(png_output / f'{stem}.png', scale=2, width=1200, height=700)
    figure.show()
    return stem


def _remove_legacy_distribution_files():
    '''Remove obsolete files containing eight distributions in one image.'''
    legacy_files = (
        Path(png_dir) / 'distributions_top8_features.png',
        Path(html_dir) / 'distributions_top8_features.html',
    )
    for path in legacy_files:
        if path.exists():
            path.unlink()


def _plot_feature_distribution(df, feature, position, png_output, html_output):
    '''Create and save one independently scaled feature-distribution chart.'''
    safe_values, phishing_values = _class_feature_values(df, feature)
    figure = _new_distribution_figure()
    feature_label = get_readable_name(feature)
    combined, lower, upper = _robust_feature_range(safe_values, phishing_values)
    edges, _, x_scale = _distribution_geometry(combined, lower, upper)
    excluded, maximum = _add_class_histograms(
        figure, safe_values, phishing_values, edges,
    )
    _style_distribution_figure(
        figure, feature_label,
        'Udział próbek klasy w pokazanym zakresie [%]', 'overlay',
    )
    x_lower, x_upper, x_ticks = x_scale
    y_lower, y_upper, y_ticks = scientific_scale(
        [0.0, max(maximum, 1.0)], include_zero=True, reserve_ratio=0.16,
        target_intervals=5, rounded_bounds=False,
    )
    figure.update_xaxes(
        range=[x_lower, x_upper], tickmode='array',
        tickvals=x_ticks, tickformat=plotly_tick_format(x_ticks),
    )
    figure.update_yaxes(
        range=[y_lower, y_upper], tickmode='array',
        tickvals=y_ticks, tickformat='.2~f',
    )
    total = len(safe_values) + len(phishing_values)
    excluded_percent = excluded / total * 100 if total else 0.0
    print(
        f'{feature_label}: pokazany zakres {edges[0]:.2f}–{edges[-1]:.2f}; '
        f'poza zakresem {excluded} próbek ({excluded_percent:.2f}%).'
    )
    return _save_distribution_figure(
        figure, position, feature, png_output, html_output,
    )


def _missing_value_frame(df, features):
    '''Build class-wise percentages of the -1 missing-value sentinel.'''
    rows = []
    for feature in features:
        safe = df.loc[df[TARGET_COL] == 0, feature]
        phishing = df.loc[df[TARGET_COL] == 1, feature]
        rows.append({
            'Cecha': get_readable_name(feature),
            'Brak phishingu': float((safe == -1).mean() * 100),
            'Phishing': float((phishing == -1).mean() * 100),
            'Brak phishingu (liczba)': int((safe == -1).sum()),
            'Phishing (liczba)': int((phishing == -1).sum()),
        })
    return pd.DataFrame(rows)


def _add_missing_value_bars(figure, missing_df):
    '''Add both target classes to the missing-value comparison chart.'''
    for label, color in (
        ('Brak phishingu', CLASS_COLORS['safe']),
        ('Phishing', CLASS_COLORS['phishing']),
    ):
        values = missing_df[label]
        value_labels = [
            f'{value:.2f}%'.replace('.', ',') if value > 0 else ''
            for value in values
        ]
        figure.add_bar(
            name=label, y=missing_df['Cecha'], x=values,
            orientation='h', marker_color=color,
            text=value_labels, textposition='outside', cliponaxis=False,
            customdata=missing_df[f'{label} (liczba)'],
            hovertemplate=(
                '%{y}<br>Udział wartości -1: %{x:.2f}%'
                '<br>Liczba próbek: %{customdata}<extra></extra>'
            ),
        )


def _plot_missing_values(df, features):
    '''Plot the sentinel-value rate independently of distributions.'''
    missing_df = _missing_value_frame(df, features)
    missing_df.to_csv(Path(csv_dir) / 'missing_values_minus1.csv', index=False)
    missing_df = missing_df[
        missing_df[['Brak phishingu', 'Phishing']].max(axis=1) > 0
    ]
    if missing_df.empty:
        print('Brak wartości -1 w analizowanych cechach.')
        return
    figure = go.Figure()
    _add_missing_value_bars(figure, missing_df)
    maximum = float(missing_df[['Brak phishingu', 'Phishing']].max().max())
    lower, upper, ticks = scientific_scale(
        [0.0, maximum], include_zero=True, reserve_ratio=0.12,
        target_intervals=5, rounded_bounds=False,
    )
    height = max(500, 105 * len(missing_df) + 180)
    figure.update_layout(
        width=1200, height=height, barmode='group',
        xaxis_title='Udział próbek z wartością -1 [%]',
        plot_bgcolor='white', font=dict(size=16),
        legend=dict(
            orientation='h', x=0.5, xanchor='center',
            y=1.08, yanchor='bottom',
        ),
        margin=dict(l=360, r=100, t=80, b=100), separators=', ',
    )
    figure.update_xaxes(
        range=[lower, upper], tickmode='array', tickvals=ticks,
        showgrid=True, gridcolor='lightgray',
        tickformat=plotly_tick_format(ticks),
    )
    figure.update_yaxes(showgrid=False, autorange='reversed')
    figure.write_html(Path(html_dir) / 'missing_values_minus1.html')
    figure.write_image(
        Path(png_dir) / 'missing_values_minus1.png',
        scale=2, width=1200, height=height,
    )
    figure.show()


def analyze_distributions(df, sample_size=1000):
    '''Analyze normality and save eight independent feature distributions.'''
    print('=' * 80)
    print('ANALIZA ROZKŁADÓW')
    print('=' * 80)
    numeric = [
        column for column in feature_columns(df)
        if pd.api.types.is_numeric_dtype(df[column])
    ]
    normality_df = pd.DataFrame([
        _normality_record(df, column, sample_size) for column in numeric
    ])
    normal_count = int(normality_df['Rozkład Normalny'].sum())
    print(
        f'Cechy z rozkładem normalnym: {normal_count}/{len(normality_df)} '
        '(test Shapiro-Wilka, poziom istotności 0,05)'
    )
    top_features = feature_importance['Cecha'].head(8).tolist()
    _remove_legacy_distribution_files()
    distribution_png = Path(png_dir) / 'distributions'
    distribution_html = Path(html_dir) / 'distributions'
    distribution_png.mkdir(parents=True, exist_ok=True)
    distribution_html.mkdir(parents=True, exist_ok=True)
    for position, feature in enumerate(top_features, start=1):
        _plot_feature_distribution(
            df, feature, position, distribution_png, distribution_html,
        )
    _plot_missing_values(df, top_features)
    normality_df.to_csv(Path(csv_dir) / 'distribution_analysis.csv', index=False)
    print(f'✓ Osobne wykresy rozkładów zapisano w: {distribution_png}')
    print(f'✓ Analizę normalności zapisano w: {csv_dir}/distribution_analysis.csv')
    return normality_df


distribution_analysis = analyze_distributions(df_clean)
"""


def _normalize_distribution_ranges(source: str) -> str:
    """Limit continuous histograms to an explicitly reported robust range."""
    marker = (
        "        else:\n"
        "            fig.add_trace(go.Histogram(\n"
        "                x=data0,"
    )
    if marker in source and "visible_data0" not in source:
        source = source.replace(marker, _robust_histogram_prefix(), 1)
        source = source.replace("                x=data1,", "                x=visible_data1,", 1)
        source = _replace_histogram_range_block(source)
    return _normalize_distribution_axis_labels(source)


def _normalize_distribution_axis_labels(source: str) -> str:
    """Describe class-normalized and robust-range percentages precisely."""
    source = source.replace(
        "title_text='Procent [%]'",
        "title_text='Udział próbek danej klasy [%]'",
        1,
    ).replace(
        "title_text='Procent [%]'",
        "title_text='Udział próbek w pokazanym zakresie [%]'",
        1,
    )
    return source.replace(
        "Rozkład wartości ośmiu najważniejszych cech według klasy "
        "(udział w klasie, bez wartości -1)",
        "Rozkład wartości ośmiu najważniejszych cech według klasy "
        "(bez wartości -1; cechy ciągłe: zakres 1.–99. percentyla)",
    )


def _robust_histogram_prefix() -> str:
    """Return notebook code preparing visible continuous-feature observations."""
    return (
        "        else:\n"
        "            combined_data = np.concatenate([data0, data1])\n"
        "            x_min, x_max = np.quantile(combined_data, [0.01, 0.99])\n"
        "            if x_min == x_max:\n"
        "                x_min, x_max = np.quantile(combined_data, [0.001, 0.999])\n"
        "            if x_min == x_max:\n"
        "                x_min, x_max = combined_data.min(), combined_data.max()\n"
        "            visible_data0 = data0[(data0 >= x_min) & (data0 <= x_max)]\n"
        "            visible_data1 = data1[(data1 >= x_min) & (data1 <= x_max)]\n"
        "            fig.add_trace(go.Histogram(\n"
        "                x=visible_data0,"
    )


def _replace_histogram_range_block(source: str) -> str:
    """Replace full-range axis bounds with stable robust-range padding."""
    old = (
        "            all_data = np.concatenate([data0, data1])\n"
        "            x_min, x_max = all_data.min(), all_data.max()\n"
        "            x_range = x_max - x_min\n"
    )
    new = (
        "            all_data = np.concatenate([visible_data0, visible_data1])\n"
        "            x_range = max(x_max - x_min, 1e-9)\n"
    )
    return source.replace(old, new, 1)


CORRELATION_HEATMAP_SOURCE = """from textwrap import wrap


def _wrapped_correlation_label(label, width=24):
    '''Wrap a feature label without abbreviating its meaning.'''
    return '<br>'.join(wrap(label, width=width))


def _correlation_labels(columns, target_column):
    '''Return readable, wrapped labels for heatmap axes.'''
    return [
        _wrapped_correlation_label(
            'Klasa docelowa' if column == target_column else get_readable_name(column)
        )
        for column in columns
    ]


def _balanced_correlation_features(top_n):
    '''Select the strongest positive and negative target correlations.'''
    correlations = target_correlations['Korelacja']
    positive_count = top_n // 2
    positive = correlations[correlations > 0].nlargest(positive_count)
    negative = correlations[correlations < 0].nsmallest(top_n - positive_count)
    selected = list(positive.index) + list(negative.index)
    if len(selected) < top_n:
        remaining = correlations.drop(index=selected).abs().nlargest(top_n - len(selected))
        selected.extend(remaining.index.tolist())
    return selected


def _add_correlation_values(figure, matrix, labels):
    '''Annotate every matrix cell using contrast-aware text colors.'''
    for row, column in np.ndindex(matrix.shape):
        value = float(matrix.iloc[row, column])
        label = f'{value:.2f}'.replace('.', ',')
        figure.add_annotation(
            x=labels[column], y=labels[row], text=label,
            showarrow=False,
            font=dict(size=13, color='white' if abs(value) >= 0.55 else 'black'),
        )


def _style_correlation_heatmap(figure):
    '''Apply a large, publication-oriented heatmap layout.'''
    figure.update_layout(
        width=1600, height=1500,
        margin=dict(l=310, r=170, t=50, b=330),
        xaxis=dict(title='Cecha', tickangle=-55, tickfont=dict(size=13)),
        yaxis=dict(title='Cecha', autorange='reversed', tickfont=dict(size=13)),
        plot_bgcolor='white', separators=', ',
    )


def create_correlation_heatmap(df, top_n=10):
    '''Create one readable, complete Pearson correlation matrix.'''
    target_column = TARGET_COL
    top_features = _balanced_correlation_features(top_n)
    matrix = df[top_features + [target_column]].corr()
    labels = _correlation_labels(matrix.columns, target_column)
    figure = go.Figure(data=go.Heatmap(
        z=matrix.values, x=labels, y=labels,
        colorscale=[
            [0.00, '#2F5F8F'], [0.25, '#A9C2D8'], [0.50, '#f7f7f7'],
            [0.75, '#F2B38C'], [1.00, '#B84E16'],
        ],
        zmin=-1, zmax=1, zmid=0, xgap=1, ygap=1,
        colorbar=dict(
            title='Współczynnik<br>korelacji Pearsona',
            tickvals=[-1, -0.5, 0, 0.5, 1],
            ticktext=['-1', '-0,5', '0', '0,5', '1'],
        ),
        hovertemplate=(
            '%{y}<br>%{x}<br>Współczynnik korelacji: '
            '%{z:.3f}<extra></extra>'
        ),
    ))
    _add_correlation_values(figure, matrix, labels)
    _style_correlation_heatmap(figure)
    figure.write_html(Path(html_dir) / 'correlation_heatmap.html')
    figure.write_image(
        Path(png_dir) / 'correlation_heatmap.png',
        scale=2, width=1600, height=1500,
    )
    figure.show()
    print(f'✓ Mapa korelacji zapisana do: {png_dir}/correlation_heatmap.png')
    return matrix


correlation_heatmap = create_correlation_heatmap(df_clean, top_n=10)
"""


def _correlation_heatmap_source() -> str:
    """Return the complete symmetric correlation-matrix cell."""
    return CORRELATION_HEATMAP_SOURCE


def _normalize_tree_plots(source: str) -> str:
    """Translate technical tree annotations shown inside saved figures."""
    if "plot_tree(" in source and "for node_label in axis.texts:" not in source:
        marker = "        impurity=True,\n    )\n"
        addition = marker + _sklearn_tree_translation()
        source = source.replace(marker, addition, 1)
    if "xgb.plot_tree(" in source and "for node_label in axis.texts:" not in source:
        marker = "    xgb.plot_tree(model, num_trees=0, ax=axis, rankdir='LR')\n"
        source = source.replace(marker, marker + _xgboost_tree_translation(), 1)
    return source


MLP_AUC_LIMITS_PATTERN = re.compile(
    r"    auc_limits = shared_limits\(\[.*?    \], floor=0\.0\)\n",
    re.DOTALL,
)
MLP_LOSS_LIMITS_PATTERN = re.compile(
    r"    loss_limits = shared_limits\(\[.*?    \], floor=0\.0\)\n",
    re.DOTALL,
)
LEGACY_MLP_LOSS_LIMITS_SOURCE = (
    "    loss_bounds = shared_limits([\n"
    "        result['epoch_end_train_loss'], history['val_loss'],\n"
    "    ], floor=0.0)\n"
    "    loss_limits = (0.0, loss_bounds[1])\n"
)
LEGACY_MLP_AUC_LIMITS_SOURCE = (
    "    auc_bounds = shared_limits([\n"
    "        result['epoch_end_train_auc'], history['val_auc'],\n"
    "    ], padding_ratio=0.08, floor=0.0)\n"
    "    auc_limits = (auc_bounds[0], min(1.0, auc_bounds[1]))\n"
)
MLP_LOSS_LIMITS_SOURCE = (
    "    loss_limits = shared_limits([\n"
    "        result['epoch_end_train_loss'], history['val_loss'],\n"
    "    ], padding_ratio=0.08, floor=0.0)\n"
)
MLP_AUC_LIMITS_SOURCE = (
    "    auc_limits = shared_limits([\n"
    "        result['epoch_end_train_auc'], history['val_auc'],\n"
    "    ], padding_ratio=0.08, floor=0.0, ceiling=1.0)\n"
)


def _normalize_mlp_epoch_scales(source: str) -> str:
    """Fit MLP epoch scales to observations inside valid score domains."""
    if "epoch_end_train_auc" not in source or "auc_limits" not in source:
        return source
    source = source.replace(
        "label='Uczenie modelu'", "label='Zbiór uczący'"
    )
    source = _replace_mlp_limit_definitions(source)
    return source.replace("figure.tight_layout()", "figure.tight_layout(pad=1.2)")


def _replace_mlp_limit_definitions(source: str) -> str:
    """Replace fixed MLP epoch bounds with domain-aware dynamic limits."""
    source = source.replace(
        "    auc_limits = (0.0, 1.0)\n", MLP_AUC_LIMITS_SOURCE, 1
    )
    source = source.replace(
        LEGACY_MLP_AUC_LIMITS_SOURCE, MLP_AUC_LIMITS_SOURCE, 1
    )
    source = source.replace(
        LEGACY_MLP_LOSS_LIMITS_SOURCE, MLP_LOSS_LIMITS_SOURCE, 1
    )
    source = MLP_AUC_LIMITS_PATTERN.sub(
        MLP_AUC_LIMITS_SOURCE, source, count=1
    )
    source = MLP_LOSS_LIMITS_PATTERN.sub(
        MLP_LOSS_LIMITS_SOURCE, source, count=1
    )
    return source


FINAL_METRICS_SAVE_PATTERN = re.compile(
    r"    save_dataset_figure\(\s*"
    r"figure, PLOTS_DIR, dataset_name,\s*"
    r"'(?P<prefix>[a-z_]+)_final_train_test_comparison\.png'\s*"
    r"\)\n    plt\.show\(\)"
)

FINAL_ERROR_PLOT_PATTERN = re.compile(
    r"    figure, _ = plot_final_classification_error\(.*?"
    r"    plt\.show\(\)\n",
    re.DOTALL,
)

FINAL_METRIC_FIGURES_PATTERN = re.compile(
    r"    metric_figures = plot_final_train_test_metric_figures\(\n"
    r"(?P<arguments>.*?)"
    r"    \)\n"
    r"    for metric_name, metric_figure in metric_figures\.items\(\):\n"
    r"        save_dataset_figure\(\n"
    r"            metric_figure, PLOTS_DIR, dataset_name,\n"
    r"            f'(?P<prefix>[a-z_]+)_final_train_test_"
    r"\{metric_name\}\.png',\n"
    r"        \)\n"
    r"        plt\.show\(\)\n",
    re.DOTALL,
)

INLINE_GROUPED_METRIC_PATTERN = re.compile(
    r"    figure, _ = plot_final_train_test_metrics\(\n"
    r"(?P<arguments>(?:        .*\n)+?)"
    r"    \)\n",
)

CURRENT_GROUPED_METRIC_PATTERN = re.compile(
    r"    train_metrics, test_metrics = classification_metric_sets\(\n"
    r"        evaluation_results\[dataset_name\]\n"
    r"    \)\n"
    r"    figure, _ = plot_final_train_test_metrics\(\n"
    r"        train_metrics, test_metrics\n"
    r"    \)\n"
    r"    save_dataset_figure\(\n"
    r"        figure, PLOTS_DIR, dataset_name,\n"
    r"        '(?P<prefix>[a-z_]+)_metrics_comparison\.png',\n"
    r"    \)\n"
    r"    plt\.show\(\)\n",
)

DUPLICATE_TEST_METRIC_PATTERN = re.compile(
    r"metric_keys = \['accuracy', 'precision', 'recall', 'f1', 'auc_roc'\]\n\n"
    r"for dataset_name in \['Pełny zbiór', 'Zoptymalizowany zbiór'\]:\n"
    r"    values = \[evaluation_results\[dataset_name\]\[key\] "
    r"for key in metric_keys\]\n"
    r"    figure, _ = plot_classification_metrics"
    r"\(metric_keys, values, dataset_name\)\n"
    r"    save_dataset_figure\(\n"
    r"        figure, PLOTS_DIR, dataset_name, "
    r"'[a-z_]+_metrics_comparison\.png'\n"
    r"    \)\n"
    r"    plt\.show\(\)\n\n",
)


def _normalize_final_train_test_plots(source: str) -> str:
    """Save final train/test metrics in two readable grouped figures."""
    source, replacements = FINAL_METRIC_FIGURES_PATTERN.subn(
        _grouped_final_metrics_source, source, count=1
    )
    if not replacements:
        source = FINAL_METRICS_SAVE_PATTERN.sub(
            _final_metrics_save_source, source, count=1
        )
    source = INLINE_GROUPED_METRIC_PATTERN.sub(
        _coherent_grouped_metric_call, source, count=1
    )
    return CURRENT_GROUPED_METRIC_PATTERN.sub(
        _two_metric_groups_source, source, count=1,
    )


def _remove_duplicate_final_error_plot(source: str) -> str:
    """Keep final classification error only in the grouped error figure."""
    return FINAL_ERROR_PLOT_PATTERN.sub(
        _obsolete_error_figure_cleanup, source, count=1,
    )


def _obsolete_error_figure_cleanup(match: re.Match) -> str:
    """Return cleanup code for the superseded standalone error figure."""
    filename = re.search(
        r"'([a-z_]+_final_error_rate\.png)'", match.group(0)
    )
    if filename is None:
        return match.group(0)
    return (
        "    remove_dataset_figure(\n"
        f"        PLOTS_DIR, dataset_name, '{filename.group(1)}'\n"
        "    )\n"
    )


def _final_metrics_save_source(match: re.Match) -> str:
    """Return the save operation for a legacy grouped metric figure."""
    prefix = match.group("prefix")
    return (
        "    save_dataset_figure(\n"
        "        figure, PLOTS_DIR, dataset_name,\n"
        f"        '{prefix}_metrics_comparison.png',\n"
        "    )\n"
        "    plt.show()"
    )


def _grouped_final_metrics_source(match: re.Match) -> str:
    """Return two metric-group plots built from one evaluation result."""
    prefix = match.group("prefix")
    return _metric_groups_source(prefix)


def _metric_groups_source(prefix: str) -> str:
    """Return notebook code saving both metric-comparison groups."""
    return (
        "    train_metrics, test_metrics = classification_metric_sets(\n"
        "        evaluation_results[dataset_name]\n"
        "    )\n"
        "    metric_figures = plot_final_train_test_metric_groups(\n"
        "        train_metrics, test_metrics\n"
        "    )\n"
        "    for group_name, metric_figure in metric_figures.items():\n"
        "        save_dataset_figure(\n"
        "            metric_figure, PLOTS_DIR, dataset_name,\n"
        f"            f'{prefix}_metrics_{{group_name}}.png',\n"
        "        )\n"
        "        plt.show()\n"
    )


def _two_metric_groups_source(match: re.Match) -> str:
    """Replace one crowded comparison with two metric groups."""
    return _metric_groups_source(match.group("prefix"))


def _coherent_grouped_metric_call(match: re.Match) -> str:
    """Replace duplicated local values with the stored evaluation result."""
    if "'accuracy': train_accuracy" not in match.group("arguments"):
        return match.group(0)
    return (
        "    train_metrics, test_metrics = classification_metric_sets(\n"
        "        evaluation_results[dataset_name]\n"
        "    )\n"
        "    figure, _ = plot_final_train_test_metrics(\n"
        "        train_metrics, test_metrics\n"
        "    )\n"
    )


def _remove_duplicate_test_metric_plot(source: str) -> str:
    """Keep time and feature charts after removing test-only metric duplicate."""
    return DUPLICATE_TEST_METRIC_PATTERN.sub(
        "for dataset_name in ['Pełny zbiór', 'Zoptymalizowany zbiór']:\n",
        source,
        count=1,
    )


def _sklearn_tree_translation() -> str:
    """Return notebook code translating scikit-learn tree node labels."""
    return (
        "    for node_label in axis.texts:\n"
        "        translated = node_label.get_text()\n"
        "        translated = translated.replace('samples', 'próbki')\n"
        "        translated = translated.replace('value', 'udział klas')\n"
        "        translated = translated.replace('class', 'klasa')\n"
        "        translated = translated.replace('True', 'prawda')\n"
        "        translated = translated.replace('False', 'fałsz')\n"
        "        node_label.set_text(translated)\n"
    )


def _xgboost_tree_translation() -> str:
    """Return notebook code translating XGBoost tree node labels."""
    return (
        "    for node_label in axis.texts:\n"
        "        translated = node_label.get_text()\n"
        "        translated = translated.replace('yes', 'tak')\n"
        "        translated = translated.replace('no', 'nie')\n"
        "        translated = translated.replace('missing', 'brak')\n"
        "        translated = translated.replace('leaf', 'liść')\n"
        "        translated = translated.replace('gain', 'zysk')\n"
        "        translated = translated.replace('cover', 'pokrycie')\n"
        "        node_label.set_text(translated)\n"
    )


def _replace_visualization_labels(source: str) -> str:
    """Normalize validation and parameter-error labels in notebook source."""
    source = normalize_parameter_error_curve(source)
    source = normalize_mlp_epoch_metrics(source)
    return _replace_validation_labels(source)


def _replace_validation_labels(source: str) -> str:
    """Use the concise scientific name for the validation split."""
    source = source.replace(
        "plot_parameter_train_test_error",
        "plot_parameter_train_validation_error",
    )
    source = source.replace(
        "Zbiór walidacyjny (odłożony)",
        "Zbiór walidacyjny",
    )
    source = source.replace(
        "Walidacja na odłożonym zbiorze",
        "Zbiór walidacyjny",
    )
    source = source.replace(
        "Walidacja na wydzielonym zbiorze",
        "Zbiór walidacyjny",
    )
    return source


def normalize_parameter_error_curve(source: str) -> str:
    """Use validation data for parameter-error diagnostics."""
    if "validation_results = {}" not in source or "test_errors" not in source:
        return source
    replacements = {
        "X_test_curve": "X_val_curve",
        "y_test_curve": "y_val_curve",
        "test_errors": "validation_errors",
        "predict(X_test_curve)": "predict(X_val_curve)",
        "plot_parameter_train_test_error": "plot_parameter_train_validation_error",
        "parameter_train_test_error.png": "parameter_train_validation_error.png",
    }
    for old, new in replacements.items():
        source = source.replace(old, new)
    return source


def normalize_mlp_epoch_metrics(source: str) -> str:
    """Prevent MLP epoch monitoring from reading the final test split."""
    markers = (
        "EpochEndEvaluationCallback",
        "epoch_end_test_accuracy",
        "epoch_train_test_error.png",
        "epoch_train_validation_error.png",
    )
    if not any(marker in source for marker in markers):
        return source
    source = source.replace("from utils.keras_epoch_metrics import EpochEndEvaluationCallback\n", "")
    source = _remove_mlp_test_callback(source)
    return _replace_mlp_epoch_result_fields(source)


def _remove_mlp_test_callback(source: str) -> str:
    """Remove callback construction and registration from the MLP fit."""
    start = source.find("    epoch_end_metrics = EpochEndEvaluationCallback(")
    if start >= 0:
        end = source.find("    monitor = ResourceMonitor", start)
        source = source[:start] + source[end:]
    return source.replace(
        "callbacks=[early_stop, reduce_lr, epoch_end_metrics]",
        "callbacks=[early_stop, reduce_lr]",
    )


def _replace_mlp_epoch_result_fields(source: str) -> str:
    """Derive epoch plots from Keras history and validation metrics."""
    return _rename_mlp_error_plot(_replace_mlp_history_fields(source))


def _replace_mlp_history_fields(source: str) -> str:
    """Replace callback metrics with values already collected by Keras."""
    old = (
        "        'epoch_end_train_loss': epoch_end_metrics.loss,\n"
        "        'epoch_end_train_accuracy': epoch_end_metrics.accuracy,\n"
        "        'epoch_end_train_auc': epoch_end_metrics.auc,\n"
        "        'epoch_end_test_accuracy': epoch_end_metrics.test_accuracy,"
    )
    new = (
        "        'epoch_end_train_loss': history.history['loss'],\n"
        "        'epoch_end_train_accuracy': history.history['accuracy'],\n"
        "        'epoch_end_train_auc': history.history['auc'],\n"
        "        'epoch_end_validation_accuracy': history.history['val_accuracy'],"
    )
    return source.replace(old, new).replace(
        "epoch_end_test_accuracy", "epoch_end_validation_accuracy"
    )


def _rename_mlp_error_plot(source: str) -> str:
    """Rename the epoch error plot and use the validation-specific helper."""
    replacements = {
        "epoch_train_test_error.png": "epoch_train_validation_error.png",
        "plot_parameter_train_test_error": "plot_parameter_train_validation_error",
        "Walidacja krzyżowa (5-krotna) — binarna entropia krzyżowa": "Zbiór walidacyjny",
        "Walidacja na odłożonym zbiorze — binarna entropia krzyżowa": "Zbiór walidacyjny",
        "Uczenie modelu — binarna entropia krzyżowa": "Uczenie modelu",
        "label='Walidacja krzyżowa (5-krotna)'": "label='Zbiór walidacyjny'",
        "Binarna entropia krzyżowa (BCE)": "Binarna entropia krzyżowa",
    }
    for old, new in replacements.items():
        source = source.replace(old, new)
    return source


def normalize_resource_visualization(source: str) -> str:
    """Make resource plots visible in executed notebook output and report paths."""
    if "monitor.save_phase_trace" not in source:
        return source
    source = _capture_trace_path(source)
    source = _inject_resource_baseline_summary(source)
    if "display(cpu_figure)" not in source:
        source = _inject_display_import(source)
        source = _append_resource_output(source)
    source = _capture_figure_paths(source)
    return source


def _inject_resource_baseline_summary(source: str) -> str:
    """Print the measured baseline definition beside resource figures."""
    call = "    print(resource_baseline_summary(monitor))\n"
    if call in source:
        return source
    pattern = re.compile(
        r"(?m)^(?P<indent>    )stats = [^\n]+\n"
    )
    return pattern.sub(
        lambda match: match.group(0) + call,
        source, count=1,
    )


def _capture_trace_path(source: str) -> str:
    """Assign inline trace paths to a name used by the output message."""
    pattern = re.compile(
        r"(?m)^(?P<indent>\s*)monitor\.save_phase_trace\((?P<path>trace_dir / f[^\n]+)\)$"
    )
    replacement = (
        r"\g<indent>trace_path = \g<path>\n"
        r"\g<indent>monitor.save_phase_trace(trace_path)"
    )
    return pattern.sub(replacement, source, count=1)


def _inject_display_import(source: str) -> str:
    """Add the notebook display helper before the resource loop."""
    return source.replace(
        "for dataset_name in [",
        "from IPython.display import display\n\nfor dataset_name in [",
        1,
    )


def _capture_figure_paths(source: str) -> str:
    """Keep paths returned by the figure-saving helper for reporting."""
    source = source.replace(
        "    save_dataset_figure(cpu_figure,",
        "    cpu_path = save_dataset_figure(cpu_figure,",
        1,
    )
    return source.replace(
        "    save_dataset_figure(ram_figure,",
        "    ram_path = save_dataset_figure(ram_figure,",
        1,
    )


def _append_resource_output(source: str) -> str:
    """Print generated paths and display both resource figures."""
    output = (
        "    print(f'Zapisano ślad: {trace_path}')\n"
        "    print(f'Zapisano wykres CPU: {cpu_path}')\n"
        "    print(f'Zapisano wykres RAM: {ram_path}')\n"
        "    display(cpu_figure)\n"
        "    display(ram_figure)\n"
        "    plt.close(cpu_figure)"
    )
    return source.replace("    plt.close(cpu_figure)", output, 1)


def normalize_notebook(path: Path) -> bool:
    """Normalize one notebook and report whether its source changed."""
    notebook = json.loads(path.read_text(encoding="utf-8"))
    changed = _normalize_notebook_cells(notebook, path)
    changed = _ensure_class_colors_import(notebook) or changed
    changed = _ensure_count_colorbar_import(notebook) or changed
    changed = _remove_unused_plot_imports(notebook) or changed
    if changed:
        _write_notebook(path, notebook)
    return changed


def _ensure_class_colors_import(notebook: dict) -> bool:
    """Import the shared class palette when a notebook references it."""
    code = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook.get("cells", [])
        if cell.get("cell_type") == "code"
    )
    if "CLASS_COLORS[" not in code or "    CLASS_COLORS,\n" in code:
        return False
    for cell in notebook.get("cells", []):
        source = "".join(cell.get("source", []))
        updated = _inject_plot_import_names(source, ("CLASS_COLORS",))
        if updated != source:
            cell["source"] = updated.splitlines(keepends=True)
            return True
    return False


def _ensure_count_colorbar_import(notebook: dict) -> bool:
    """Import the integer color-bar helper when a notebook calls it."""
    code = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook.get("cells", [])
        if cell.get("cell_type") == "code"
    )
    name = "configure_count_colorbar"
    if f"{name}(axis)" not in code or f"    {name},\n" in code:
        return False
    for cell in notebook.get("cells", []):
        source = "".join(cell.get("source", []))
        updated = _inject_plot_import_names(source, (name,))
        if updated != source:
            cell["source"] = updated.splitlines(keepends=True)
            return True
    return False


def _normalize_notebook_cells(notebook: dict, path: Path) -> bool:
    """Normalize all relevant cell sources in one loaded notebook."""
    changed = _remove_redundant_model_testing_cells(notebook)
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") == "markdown":
            changed = _normalize_markdown_cell(cell) or changed
            continue
        if cell.get("cell_type") != "code":
            continue
        changed = _normalize_code_cell(cell, path) or changed
    return changed


REDUNDANT_MODEL_TESTING_CELL_IDS = {"44c87fe0", "544cb550"}


def _remove_redundant_model_testing_cells(notebook: dict) -> bool:
    """Remove diagnostics already presented by per-algorithm notebooks."""
    cells = notebook.get("cells", [])
    filtered = [
        cell for cell in cells
        if cell.get("id") not in REDUNDANT_MODEL_TESTING_CELL_IDS
    ]
    if len(filtered) == len(cells):
        return False
    notebook["cells"] = filtered
    return True


def _normalize_code_cell(cell: dict, path: Path) -> bool:
    """Normalize source and cached outputs in one code cell."""
    source = cell.get("source", [])
    current = "".join(source) if isinstance(source, list) else source
    updated = normalize_source(current)
    updated = _normalize_cell_plot_layout(cell.get("id"), updated)
    if path.name.startswith("0_"):
        updated = _inject_statistical_plot_helpers(updated)
    elif path.name[:1] in {"1", "2", "3", "4", "5"}:
        updated = _inject_scientific_plot_helpers(updated)
        updated = _stabilize_resource_baseline(updated)
        if path.name.startswith("4_"):
            updated = updated.replace("import concurrent.futures\n", "")
    changed = updated != current
    if changed:
        cell["source"] = updated.splitlines(keepends=True)
    outputs_changed = _normalize_result_output_paths(cell)
    return _migrate_plot_output(cell) or changed or outputs_changed


CLASSICAL_TRAINING_PATTERN = re.compile(
    r"    (?P<monitor>monitor|resource_monitor) = "
    r"ResourceMonitor\(interval=0\.05, debug=False\)\n"
    r"    (?P<model>model|best_model) = (?P<constructor>[^\n]+)\n"
    r"    gc\.collect\(\)\n"
    r"    (?P=monitor)\.start\(\)\n"
    r"    time\.sleep\(0\.5\)\n"
    r"    (?P=monitor)\.mark_training_start\(\)\n\n?"
    r"    (?:start_time|training_start) = time\.time\(\)\n"
    r"    (?P=model)\.fit\((?P<features>[^,\n]+), "
    r"(?P<target>[^\)\n]+)\)\n"
    r"    training_time = time\.time\(\) - "
    r"(?:start_time|training_start)\n"
    r"    (?P=monitor)\.mark_training_end\(\)\n"
    r"    gc\.collect\(\)\n"
    r"    (?P=monitor)\.wait_for_post_training_idle\(\)\n"
    r"    (?P=monitor)\.stop\(\)\n\n"
    r"    stats = (?P=monitor)\.get_stats\(\)\n"
)

MLP_TRAINING_PATTERN = re.compile(
    r"    monitor = ResourceMonitor\(interval=0\.05, debug=False\)\n"
    r"    gc\.collect\(\)\n"
    r"    monitor\.start\(\)\n"
    r"    time\.sleep\(0\.5\)\n"
    r"    monitor\.mark_training_start\(\)\n"
    r"    training_start = time\.time\(\)\n"
    r"    history = model\.fit\(\n"
    r"        data\['X_train_scaled'\], data\['y_train'\],\n"
    r"        validation_data=\(data\['X_val_scaled'\], data\['y_val'\]\),\n"
    r"        epochs=100, batch_size=128,\n"
    r"        callbacks=\[early_stop, reduce_lr\], verbose=0,\n"
    r"    \)\n"
    r"    training_time = time\.time\(\) - training_start\n"
    r"    monitor\.mark_training_end\(\)\n"
    r"    gc\.collect\(\)\n"
    r"    monitor\.wait_for_post_training_idle\(\)\n"
    r"    monitor\.stop\(\)\n"
)

MLP_ISOLATED_TRAINING = (
    "    isolated_result = fit_model_isolated(\n"
    "        model, data['X_train_scaled'], data['y_train'],\n"
    "        fit_kwargs={\n"
    "            'validation_data': (\n"
    "                data['X_val_scaled'], data['y_val'],\n"
    "            ),\n"
    "            'epochs': 100,\n"
    "            'batch_size': 128,\n"
    "            'callbacks': [early_stop, reduce_lr],\n"
    "            'verbose': 0,\n"
    "        },\n"
    "        interval=0.05,\n"
    "    )\n"
    "    model = isolated_result.model\n"
    "    history = isolated_result.history\n"
    "    monitor = isolated_result.monitor\n"
    "    training_time = isolated_result.training_time\n"
)


def _stabilize_resource_baseline(source: str) -> str:
    """Move measured fitting into a disposable worker process."""
    source = _replace_resource_monitor_import(source)
    source = CLASSICAL_TRAINING_PATTERN.sub(
        _isolated_classical_training, source
    )
    return MLP_TRAINING_PATTERN.sub(MLP_ISOLATED_TRAINING, source)


def _replace_resource_monitor_import(source: str) -> str:
    """Import the process-isolated fitting facade in model notebooks."""
    source = source.replace("import gc\n", "")
    old = "from utils.resource_monitor import ResourceMonitor\n"
    new = (
        "if 'utils.isolated_training' in sys.modules:\n"
        "    importlib.reload(sys.modules['utils.isolated_training'])\n"
        "from utils.isolated_training import fit_model_isolated\n"
    )
    return source.replace(old, new)


def _isolated_classical_training(match: re.Match) -> str:
    """Build isolated fitting code while retaining notebook variable names."""
    monitor, model = match.group("monitor"), match.group("model")
    constructor = match.group("constructor")
    features, target = match.group("features"), match.group("target")
    return (
        f"    {model} = {constructor}\n"
        f"    isolated_result = fit_model_isolated(\n"
        f"        {model}, {features}, {target}, interval=0.05,\n"
        f"    )\n"
        f"    {model} = isolated_result.model\n"
        f"    {monitor} = isolated_result.monitor\n"
        f"    training_time = isolated_result.training_time\n"
        f"    stats = isolated_result.stats\n"
    )


def _markdown_replacements() -> dict[str, str]:
    """Return notebook-section text keyed by stable cell identifier."""
    replacements = {
        "03927e29": EVALUATION_SECTION_MARKDOWN,
        "f5597cc1": EVALUATION_SECTION_MARKDOWN,
        "5d925829": EVALUATION_SECTION_MARKDOWN,
        "7e2a8c58": EVALUATION_SECTION_MARKDOWN,
        "5122d8ef": EVALUATION_SECTION_MARKDOWN,
        "4580abae": CV_SECTION_MARKDOWN,
        "0140b17d": CV_SECTION_MARKDOWN,
        "81376369": CV_SECTION_MARKDOWN,
        "fb9c7eb9": CV_SECTION_MARKDOWN,
        "c63fdbdd": MLP_PREDICTION_SECTION_MARKDOWN,
    }
    return replacements | _phase_testing_markdown_replacements()


def _phase_testing_markdown_replacements() -> dict[str, str]:
    """Return section descriptions used only by model-testing notebooks."""
    return {
        "7d5a73a6": PHASE_TESTING_INTRO_MARKDOWN,
        "940ea04e": PHASE_TESTING_SIZE_MARKDOWN,
        "b4aa0997": PHASE_TESTING_AVERAGES_MARKDOWN,
        "83f00e15": PHASE_TESTING_ERRORS_MARKDOWN,
        "f892db2c": PHASE_TESTING_ERROR_TRENDS_MARKDOWN,
        "07e15698": PHASE_TESTING_RANKING_MARKDOWN,
        "8cf3c17c": PHASE_TESTING_STATISTICS_MARKDOWN,
        "ea4bbceb": PHASE_TESTING_SUMMARY_MARKDOWN,
        "final-model-comparison-heading": FINAL_COMPARISON_MARKDOWN,
    }


def _normalize_markdown_cell(cell: dict) -> bool:
    """Clarify the scientific role of selected notebook sections."""
    cell_id = cell.get("id")
    current = "".join(cell.get("source", []))
    replacements = _markdown_replacements()
    replacement = replacements.get(cell_id) if isinstance(cell_id, str) else None
    updated = replacement or current
    if current == updated:
        return False
    cell["source"] = updated.splitlines(keepends=True)
    return True


def _normalize_result_output_paths(cell: dict) -> bool:
    """Update cached textual paths after moving results to the project root."""
    changed = False
    for output in cell.get("outputs", []):
        changed = _normalize_output_text(output) or changed
    return changed


def _normalize_output_text(output: dict) -> bool:
    """Replace the obsolete backend-results path in one notebook output."""
    text = output.get("text")
    if text is None:
        return False
    joined = "".join(text) if isinstance(text, list) else text
    updated = joined.replace("/backend/results/", "/results/")
    if joined == updated:
        return False
    output["text"] = updated.splitlines(keepends=True)
    return True


EVALUATION_SECTION_MARKDOWN = """## 6. Porównanie wyników uczenia i testowania

Miary dla procesu uczenia obliczane są na zbiorze uczącym, a miary testowe na
niezależnym zbiorze testowym. Każda miara jest prezentowana na osobnym wykresie
i wykorzystuje wyniki uzyskane przez wytrenowany model, bez jego ponownego
uczenia na potrzeby wizualizacji.

Walidacja oznacza ocenę modelu na danych, które nie uczestniczyły w jego
dopasowaniu w danym przebiegu, ale nadal należą do części danych przeznaczonej
do opracowania modelu. Nie jest ona testem końcowym. Dla każdego rozmiaru
podzbioru uczącego linia uczenia i linia walidacji przedstawiają średnią z pięciu
przebiegów, a półprzezroczyste pasmo przedstawia jedno odchylenie standardowe.
Niezależny zbiór testowy nie uczestniczy w wyznaczaniu tych krzywych.
Oś pozioma nie przedstawia liczby próbek walidacyjnych ani testowych.
"""


CV_SECTION_MARKDOWN = """## 7.2. Krzywe uczenia w pięciokrotnej walidacji krzyżowej

Oś pozioma przedstawia dziesięć rosnących rozmiarów podzbioru uczącego: od
10% do 100% danych dostępnych do uczenia w danym podziale. Dla każdego rozmiaru
model jest dopasowywany w pięciu podziałach. Każdy punkt linii uczenia jest
średnią miary obliczonej na pięciu podzbiorach uczących, a każdy punkt linii
walidacji jest średnią miary obliczonej na pięciu odpowiadających im
podzbiorach walidacyjnych. Pasmo oznacza ± jedno odchylenie standardowe.
Oś pozioma nie przedstawia liczby próbek walidacyjnych ani testowych.

Walidacja służy do oceny uogólniania modelu podczas jego opracowywania.
Nie jest testem końcowym. Niezależny zbiór testowy nie uczestniczy w
wyznaczaniu tych krzywych; porównanie końcowych wyników uczenia i testowania
znajduje się w sekcji ewaluacji modelu.
"""


MLP_PREDICTION_SECTION_MARKDOWN = """## 7.2b. Przykładowe wyniki klasyfikacji MLP

Każdy wykres przedstawia rozkład prawdopodobieństwa klas dla jednej próbki ze
zbioru testowego. Oś pozioma przedstawia przewidywane prawdopodobieństwo, a oś
pionowa — klasy rozpatrywane przez model.
"""


PHASE_TESTING_INTRO_MARKDOWN = """# Kompleksowe testowanie wytrenowanych modeli

Notebook porównuje wszystkie wytrenowane modele na dedykowanych zbiorach
testowych. Modele nie są tutaj ponownie uczone.

Wpływ hiperparametrów oraz macierze pomyłek są prezentowane w notebookach
poszczególnych algorytmów i nie są tutaj powielane.

### Mapa wyników

- **Sekcja 7** — wpływ liczby próbek testowych na miary jakości.
- **Sekcje 8 i 11** — zagregowane miary jakości i ranking modeli.
- **Sekcje 9 i 10** — liczba, udział i przykłady błędnych klasyfikacji.
- **Sekcja 12** — statystyki końcowej fazy testowej.
- **Sekcja 14** — końcowe porównanie algorytmów przeznaczone do pracy.

Notebook wyświetla tylko reprezentatywną część szczegółowych wykresów.
Komplet polskich i angielskich grafik jest zapisywany w katalogu
`results/<zbiór>/phase_testing/plots`.
"""


PHASE_TESTING_SIZE_MARKDOWN = """## 7. Wpływ liczby próbek testowych na miary jakości

Każdy wykres przedstawia jedną miarę i jeden wariant danych. Linie odpowiadają
modelom, a kolejne punkty — wynikom dla rosnącej liczby próbek testowych.
Miary jakości mają pełny, niezawężony zakres od 0 do 1, dlatego wykres nie
wyolbrzymia niewielkich różnic między modelami.
"""


PHASE_TESTING_AVERAGES_MARKDOWN = """## 8. Średnie miary jakości modeli

Tabela agreguje wyniki faz 2–5 i wszystkich analizowanych rozkładów klas.
Służy do porównania modeli bez powielania osobnego wykresu dla każdej miary.
"""


PHASE_TESTING_ERRORS_MARKDOWN = """## 9. Błędne klasyfikacje w końcowej fazie

Zestawienie dotyczy fazy 5 i zrównoważonego rozkładu klas 50/50. Pokazuje
konkretne próbki błędnie sklasyfikowane przez modele.
"""


PHASE_TESTING_ERROR_TRENDS_MARKDOWN = """## 10. Wpływ liczby próbek na błędy klasyfikacji

W notebooku pokazano szczegółowe wykresy dla fazy 5 i rozkładu 50/50 oraz
zbiorcze krzywe współczynnika błędu. Pozostałe warianty zapisano do plików.
"""


PHASE_TESTING_RANKING_MARKDOWN = """## 11. Końcowy ranking modeli

Ranking wykorzystuje te same zagregowane wyniki co sekcja 9. Nie wykonuje
dodatkowego uczenia ani predykcji.
"""


PHASE_TESTING_STATISTICS_MARKDOWN = """## 12. Statystyki fazowe — scenariusz końcowy

W notebooku pokazano szczegółowe statystyki fazy 5 osobno dla pełnego i
zoptymalizowanego wariantu danych. Statystyki faz 1–4 zapisano do plików.
Rankingi i mapy miar jakości korzystają z pełnej skali od 0 do 1.
"""


PHASE_TESTING_SUMMARY_MARKDOWN = """## 13. Podsumowanie testowania fazowego
"""


FINAL_COMPARISON_MARKDOWN = """## 14. Końcowe porównanie algorytmów
"""


PLOT_OUTPUT_VERSIONS = {
    "5636a7f0": 1,
    "b0e8eb8e": 1,
    "48d9040b": 5,
    "a2cfb665": 4,
    "d3b82b5d": 1,
    "c203e287": 1,
    "799a4957": 3,
    "fb0fe5ed": 1,
    "56682f27": 3,
    "c7af28e0": 3,
    "05281381": 1,
    "7f6b19ad": 1,
    "94d6626c": 1,
    "a60b5168": 1,
    "final-model-comparison": 1,
}


def _migrate_plot_output(cell: dict) -> bool:
    """Clear obsolete cached plots once after their layout migration."""
    cell_id = cell.get("id")
    required_version = (
        PLOT_OUTPUT_VERSIONS.get(cell_id) if isinstance(cell_id, str) else None
    )
    if required_version is None:
        return False
    metadata = cell.setdefault("metadata", {})
    if metadata.get("scientific_plot_version") == required_version:
        return False
    metadata["scientific_plot_version"] = required_version
    cell["outputs"] = []
    cell["execution_count"] = None
    return True


def _write_notebook(path: Path, notebook: dict) -> None:
    """Serialize one normalized notebook deterministically."""
    serialized = json.dumps(notebook, ensure_ascii=False, indent=2) + "\n"
    path.write_text(serialized, encoding="utf-8")


def _normalize_cell_plot_layout(cell_id: str | None, source: str) -> str:
    """Replace dense compound phase figures with standalone charts."""
    replacements = {
        "d3b82b5d": _mlp_prediction_examples_source,
        "799a4957": _size_impact_source,
        "fb0fe5ed": _average_metrics_source,
        "56682f27": _misclassification_source,
        "c7af28e0": _phase_statistics_source,
        "05281381": _imbalance_distribution_source,
        "7f6b19ad": _imbalance_precision_recall_source,
        "94d6626c": _imbalance_confusion_source,
    }
    factory = replacements.get(cell_id) if cell_id is not None else None
    return factory() if factory is not None else source


def _mlp_prediction_examples_source() -> str:
    """Return the complete MLP probability distribution by true class."""
    return _probability_distribution_source(
        "mlp", "data_splits[dataset_name]['y_test']", "prediction",
    )


IMBALANCE_DISTRIBUTION_SOURCE = """from utils.imbalance_research_plots import (
    plot_class_distribution,
    remove_legacy_imbalance_figures,
)

class_labels = ['Brak phishingu', 'Phishing']
class_counts = y_train.value_counts().reindex([0, 1], fill_value=0)
class_percentages = class_counts.div(class_counts.sum()).mul(100)
remove_legacy_imbalance_figures(plots_root)

distribution_specs = [
    (class_counts.values, False, 'class_distribution_counts.png'),
    (class_percentages.values, True, 'class_distribution_percentages.png'),
]
for values, percentage, filename in distribution_specs:
    figure = plot_class_distribution(
        class_labels, values, percentage=percentage,
    )
    figure.tight_layout()
    save_dataset_figure(figure, plots_root, 'full', filename)
    plt.show()
"""


def _imbalance_distribution_source() -> str:
    """Return separate, semantic count and percentage class charts."""
    return IMBALANCE_DISTRIBUTION_SOURCE


IMBALANCE_PRECISION_RECALL_SOURCE = """from utils.imbalance_research_plots import (
    plot_precision_recall_curves,
)

if pr_curves:
    average_precision = {
        variant: results_df.loc[
            results_df['wariant'] == variant, 'pr_auc'
        ].iloc[0]
        for variant in pr_curves
    }
    baseline = float((y_test == minority_label).mean())
    figure = plot_precision_recall_curves(
        pr_curves, average_precision, baseline,
    )
    figure.tight_layout()
    save_dataset_figure(
        figure, plots_root, 'full', 'precision_recall_curves.png',
    )
    plt.show()
else:
    print('Krzywe precyzja–czułość są niedostępne dla obecnych etykiet.')
"""


def _imbalance_precision_recall_source() -> str:
    """Return accessible precision-recall curves for balancing variants."""
    return IMBALANCE_PRECISION_RECALL_SOURCE


IMBALANCE_CONFUSION_SOURCE = """from utils.imbalance_research_plots import (
    plot_imbalance_confusion_matrix,
    shared_confusion_maximum,
)

confusion_maximum = shared_confusion_maximum(cms)
for variant, matrix in cms.items():
    figure = plot_imbalance_confusion_matrix(matrix, confusion_maximum)
    figure.tight_layout()
    safe_variant = variant.replace(' ', '_')
    save_dataset_figure(
        figure, plots_root, 'full',
        f'confusion_matrix_{safe_variant}.png',
    )
    plt.show()
"""


def _imbalance_confusion_source() -> str:
    """Return comparable confusion matrices with one shared count scale."""
    return IMBALANCE_CONFUSION_SOURCE


SIZE_IMPACT_SOURCE = """from utils.phase_testing_plots import (
    plot_quality_by_test_size,
    remove_legacy_phase_testing_figures,
)

remove_legacy_phase_testing_figures(PLOTS_DIR)

phase_sizes = {
    'Faza 1: Mini Test': 2,
    'Faza 2: 1,000 probek': 1000, 'Faza 2: 1,000 próbek': 1000,
    'Faza 3: 5,000 probek': 5000, 'Faza 3: 5,000 próbek': 5000,
    'Faza 4: 10,000 probek': 10000, 'Faza 4: 10,000 próbek': 10000,
    'Faza 5: 20,000 probek': 20000, 'Faza 5: 20,000 próbek': 20000,
}
phase_summary_df['Rozmiar'] = phase_summary_df['Faza'].map(phase_sizes)
metric_specs = [
    ('ROC-AUC', 'Pole pod krzywą ROC', 'roc_auc'),
    ('F1', 'Miara F1', 'f1'),
    ('Dokladnosc', 'Dokładność klasyfikacji', 'dokladnosc'),
    ('Precyzja', 'Precyzja', 'precyzja'),
]

for dataset_type in ['full', 'optimized']:
    for metric_key, metric_label, filename in metric_specs:
        model_series = {}
        for model_name in phase_summary_df['Model'].unique():
            model_data = phase_summary_df[
                (phase_summary_df['Model'] == model_name)
                & (phase_summary_df['Zbior'] == dataset_type)
                & (phase_summary_df['Rozklad'] == '50% phishing, 50% bezpieczne')
                & (phase_summary_df['Faza'] != 'Faza 1: Mini Test')
            ].sort_values('Rozmiar')
            if not model_data.empty:
                model_series[model_name] = (
                    model_data['Rozmiar'].tolist(),
                    model_data[metric_key].tolist(),
                )
        if not model_series:
            continue
        figure = plot_quality_by_test_size(model_series, metric_label)
        figure.tight_layout()
        print(
            f'Wariant danych: {dataset_display_name(dataset_type)} | '
            f'Miara: {metric_label}'
        )
        save_dataset_figure(
            figure, PLOTS_DIR / 'summary', dataset_type,
            f'size_impact_{filename}.png',
        )
        plt.close(figure)
"""


def _size_impact_source() -> str:
    """Return standalone test-size figures for every quality measure."""
    return SIZE_IMPACT_SOURCE


AVERAGE_METRICS_SOURCE = """from utils.phase_testing_plots import (
    remove_obsolete_average_metric_figures,
)

avg_metrics_data = []
for phase_name, phase_data in phase_results.items():
    if phase_name == 'Faza 1: Mini Test':
        continue
    for dist_label in [
        '70% phishing, 30% bezpieczne', '30% phishing, 70% bezpieczne',
        '50% phishing, 50% bezpieczne',
    ]:
        for model_name, datasets in phase_data.items():
            for dataset_type in ['full', 'optimized']:
                if dataset_type in datasets and dist_label in datasets[dataset_type]:
                    metrics = datasets[dataset_type][dist_label]['metrics']
                    avg_metrics_data.append({
                        'model': model_name, 'dataset': dataset_type,
                        **{key: metrics[key] for key in
                           ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']},
                    })

avg_df = pd.DataFrame(avg_metrics_data)
model_averages = avg_df.groupby(['model', 'dataset']).mean(numeric_only=True).reset_index()
remove_obsolete_average_metric_figures(PLOTS_DIR)

weights = {'roc_auc': 0.3, 'f1': 0.25, 'accuracy': 0.2,
           'precision': 0.15, 'recall': 0.1}
ranking_rows = []
for _, row in model_averages.iterrows():
    ranking_rows.append({
        'Model': row['model'],
        'Zbiór': dataset_display_name(
            row['dataset'], DATASET_FEATURE_COUNTS[row['dataset']]
        ),
        'Wynik ważony': sum(row[key] * weight for key, weight in weights.items()),
        'ROC-AUC': row['roc_auc'], 'F1': row['f1'],
        'Dokładność': row['accuracy'], 'Precyzja': row['precision'],
        'Czułość': row['recall'],
    })
final_ranking = pd.DataFrame(ranking_rows).sort_values('Wynik ważony', ascending=False)
final_ranking.to_csv(
    RESULTS_DIR / 'summary' / 'average_model_performance.csv', index=False
)
print(final_ranking.round(4))
"""


def _average_metrics_source() -> str:
    """Return standalone average-metric comparisons and ranking data."""
    return AVERAGE_METRICS_SOURCE


MISCLASSIFICATION_SOURCE = """from utils.phase_testing_plots import (
    plot_count_ranking,
    plot_error_rate_by_test_size,
    plot_outcome_shares,
)

phase_mapping = {
    'Faza 1: Mini Test': 1, 'Faza 2: 1,000 próbek': 2,
    'Faza 3: 5,000 próbek': 3, 'Faza 4: 10,000 próbek': 4,
    'Faza 5: 20,000 próbek': 5,
}
for phase_name, phase_data in phase_results.items():
    phase_folder = PLOTS_DIR / f"phase_{phase_mapping[phase_name]}"
    phase_size = phase_summary_df[
        phase_summary_df['Faza'] == phase_name
    ]['Rozmiar'].iloc[0]
    for dist_label in [
        '70% phishing, 30% bezpieczne', '30% phishing, 70% bezpieczne',
        '50% phishing, 50% bezpieczne',
    ]:
        safe_distribution = dist_label.replace('%', 'pct').replace(',', '').replace(' ', '_')
        for dataset_type in ['full', 'optimized']:
            rows = []
            for model_name, datasets in phase_data.items():
                if dataset_type in datasets and dist_label in datasets[dataset_type]:
                    errors = len(datasets[dataset_type][dist_label]['misclassified'])
                    rows.append((model_name, phase_size - errors, errors))
            if not rows:
                continue
            models, correct, errors = map(list, zip(*rows))
            show_inline = (
                phase_mapping[phase_name] == 5
                and dist_label == '50% phishing, 50% bezpieczne'
            )
            if show_inline:
                print(
                    f'Wariant danych: {dataset_display_name(dataset_type)} | '
                    'Faza: 5 | Rozkład klas: 50/50 | '
                    'Udział wyników klasyfikacji'
                )
            figure = plot_outcome_shares(models, correct, errors)
            figure.tight_layout()
            save_dataset_figure(
                figure, phase_folder, dataset_type,
                f'classification_outcomes_{safe_distribution}.png',
                display_in_notebook=show_inline,
            )
            plt.close(figure)

            if show_inline:
                print(
                    f'Wariant danych: {dataset_display_name(dataset_type)} | '
                    'Faza: 5 | Rozkład klas: 50/50 | '
                    'Liczba błędnych klasyfikacji'
                )
            figure = plot_count_ranking(models, errors)
            figure.tight_layout()
            save_dataset_figure(
                figure, phase_folder, dataset_type,
                f'classification_error_counts_{safe_distribution}.png',
                display_in_notebook=show_inline,
            )
            plt.close(figure)

print(
    'Zapisano szczegółowe wykresy błędów dla wszystkich faz i rozkładów. '
    'W notebooku pokazano fazę 5 z rozkładem klas 50/50.'
)

comparison_data = {'full': {}, 'optimized': {}}
for phase_name, phase_data in phase_results.items():
    if phase_name == 'Faza 1: Mini Test':
        continue
    for dataset_type in comparison_data:
        comparison_data[dataset_type][phase_name] = {}
        for model_name, datasets in phase_data.items():
            balanced = '50% phishing, 50% bezpieczne'
            if dataset_type in datasets and balanced in datasets[dataset_type]:
                errors = len(datasets[dataset_type][balanced]['misclassified'])
                comparison_data[dataset_type][phase_name][model_name] = (
                    errors / phase_sizes[phase_name] * 100
                )

for dataset_type, dataset_data in comparison_data.items():
    model_series = {}
    for model_name in MODEL_COLORS:
        x_values, y_values = [], []
        for phase_name, phase in dataset_data.items():
            if model_name in phase:
                x_values.append(phase_sizes[phase_name])
                y_values.append(phase[model_name])
        if x_values:
            model_series[model_name] = (x_values, y_values)
    if not model_series:
        continue
    figure = plot_error_rate_by_test_size(model_series)
    figure.tight_layout()
    save_dataset_figure(
        figure, PLOTS_DIR / 'summary', dataset_type,
        'error_rates_by_test_size.png',
    )
    plt.close(figure)
"""


def _misclassification_source() -> str:
    """Return independent count and rate charts for classification errors."""
    return MISCLASSIFICATION_SOURCE


PHASE_STATISTICS_SOURCE = """from utils.phase_testing_plots import (
    plot_count_ranking,
    plot_metric_heatmap,
    plot_probability_ranking,
)

phase_mapping = {
    'Faza 1: Mini Test': 1, 'Faza 2: 1,000 próbek': 2,
    'Faza 3: 5,000 próbek': 3, 'Faza 4: 10,000 próbek': 4,
    'Faza 5: 20,000 próbek': 5,
}
for phase_name in phase_summary_df['Faza'].unique():
    phase_folder = PLOTS_DIR / f"phase_{phase_mapping[phase_name]}"
    for dataset_type in ['full', 'optimized']:
        data = phase_summary_df[
            (phase_summary_df['Faza'] == phase_name)
            & (phase_summary_df['Zbior'] == dataset_type)
        ]
        if data.empty:
            continue
        show_inline = phase_mapping[phase_name] == 5
        top = data.nlargest(10, 'ROC-AUC').reset_index(drop=True)
        labels = top['Model'] + ' | ' + top['Rozklad']
        if show_inline:
            print(
                f'Wariant danych: {dataset_display_name(dataset_type)} | '
                'Faza: 5 | Ranking według pola pod krzywą ROC'
            )
        figure = plot_probability_ranking(
            labels, top['ROC-AUC'].to_numpy(), dataset_type,
        )
        figure.tight_layout()
        save_dataset_figure(
            figure, phase_folder, dataset_type, 'roc_auc_ranking.png',
            display_in_notebook=show_inline,
        )
        plt.close(figure)

        errors = data.groupby('Model')['Bledne'].sum().sort_values()
        if show_inline:
            print(
                f'Wariant danych: {dataset_display_name(dataset_type)} | '
                'Faza: 5 | Łączna liczba błędnych klasyfikacji'
            )
        figure = plot_count_ranking(
            errors.index.tolist(), errors.to_numpy(), total=True,
        )
        figure.tight_layout()
        save_dataset_figure(
            figure, phase_folder, dataset_type, 'classification_errors.png',
            display_in_notebook=show_inline,
        )
        plt.close(figure)

        metrics = data.groupby('Model')[['ROC-AUC', 'F1', 'Dokladnosc']].mean()
        metrics = metrics.rename(columns={
            'ROC-AUC': 'Pole pod krzywą ROC',
            'F1': 'Miara F1',
            'Dokladnosc': 'Dokładność klasyfikacji',
        })
        if show_inline:
            print(
                f'Wariant danych: {dataset_display_name(dataset_type)} | '
                'Faza: 5 | Średnie miary jakości'
            )
        figure = plot_metric_heatmap(metrics, dataset_type)
        figure.tight_layout()
        save_dataset_figure(
            figure, phase_folder, dataset_type, 'average_quality_metrics.png',
            display_in_notebook=show_inline,
        )
        plt.close(figure)

print(
    'Zapisano statystyki dla wszystkich faz. '
    'W notebooku pokazano szczegółowo wyłącznie fazę 5.'
)

full_summary = phase_summary_df[phase_summary_df['Zbior'] == 'full'].groupby('Model').agg({
    'ROC-AUC': ['mean', 'std', 'count'], 'F1': ['mean', 'std'],
    'Dokladnosc': ['mean', 'std'], 'Bledne': 'sum',
}).round(4)
opt_summary = phase_summary_df[phase_summary_df['Zbior'] == 'optimized'].groupby('Model').agg({
    'ROC-AUC': ['mean', 'std', 'count'], 'F1': ['mean', 'std'],
    'Dokladnosc': ['mean', 'std'], 'Bledne': 'sum',
}).round(4)
improvement_rows = []
for model in set(full_summary.index) & set(opt_summary.index):
    full_roc = full_summary.loc[model, ('ROC-AUC', 'mean')]
    opt_roc = opt_summary.loc[model, ('ROC-AUC', 'mean')]
    full_f1 = full_summary.loc[model, ('F1', 'mean')]
    opt_f1 = opt_summary.loc[model, ('F1', 'mean')]
    improvement_rows.append({
        'Model': model, 'ROC-AUC Poprawa (%)': (opt_roc-full_roc)/full_roc*100,
        'F1 Poprawa (%)': (opt_f1-full_f1)/full_f1*100,
        'Pelny ROC-AUC': full_roc, 'Zoptym ROC-AUC': opt_roc,
    })
pd.DataFrame(improvement_rows).to_csv(
    RESULTS_DIR / 'summary' / 'dataset_improvement_analysis.csv', index=False
)
"""


def _phase_statistics_source() -> str:
    """Return three independent statistical summaries for each test phase."""
    return PHASE_STATISTICS_SOURCE


def _remove_unused_plot_imports(notebook: dict) -> bool:
    """Remove shared plotting names left unused after source normalization."""
    unused = _unused_plot_imports(notebook)
    if not unused:
        return False
    for cell in notebook.get("cells", []):
        _remove_import_names_from_cell(cell, unused)
    return True


PLOT_IMPORT_NAMES = {
    "CLASS_COLORS", "DATASET_COLORS", "HEATMAP_CMAPS",
    "classification_metric_sets",
    "configure_axis_limit_ticks",
    "configure_count_colorbar", "configure_numeric_x_axis",
    "feature_count_comparison_rows", "feature_count_measurement_summary",
    "plot_feature_count",
    "plot_final_classification_error",
    "plot_final_train_test_metric_groups",
    "plot_final_train_test_metrics",
    "plot_probability_bars", "plot_probability_distribution",
    "plot_probability_figure", "plot_sample_probability_prediction",
    "plot_training_time",
    "plot_training_time_comparison", "remove_dataset_figure",
    "resource_baseline_summary",
    "save_comparison_figure", "scientific_scale", "shared_limits",
    "training_time_measurement_summary",
    "make_subplots", "sns",
}


def _unused_plot_imports(notebook: dict) -> set[str]:
    """Return shared plotting imports referenced only by their import line."""
    code = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook.get("cells", [])
        if cell.get("cell_type") == "code"
    )
    return {
        name for name in PLOT_IMPORT_NAMES
        if len(re.findall(rf"\b{re.escape(name)}\b", code)) == 1
    }


def _remove_import_names_from_cell(cell: dict, unused: set[str]) -> None:
    """Remove selected one-name lines from a notebook import cell."""
    if cell.get("cell_type") != "code":
        return
    text = "".join(cell.get("source", []))
    cell["source"] = [
        line for line in text.splitlines(keepends=True)
        if _keep_import_line(line, unused)
    ]


def _keep_import_line(line: str, unused: set[str]) -> bool:
    """Return whether an import or import-list line remains necessary."""
    stripped = line.strip().rstrip(",")
    if stripped in unused:
        return False
    direct_imports = {
        "make_subplots": "from plotly.subplots import make_subplots",
        "sns": "import seaborn as sns",
    }
    return not any(
        name in unused and stripped == statement
        for name, statement in direct_imports.items()
    )


def main() -> None:
    """Normalize all notebook plot labels in place."""
    root = Path(__file__).resolve().parents[1] / "notebooks"
    for path in notebook_paths(root):
        if normalize_notebook(path):
            print(f"Zaktualizowano etykiety: {path}")


if __name__ == "__main__":
    main()
