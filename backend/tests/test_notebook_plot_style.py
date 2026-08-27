from pathlib import Path

import numpy as np

from utils.figure_localization import english_matplotlib_copy
from utils.notebook_plot_style import (
    CLASS_COLORS,
    ACADEMIC_FONT_FAMILY,
    DATASET_COLORS,
    HEATMAP_CMAPS,
    MODEL_COLORS,
    TRAINING_END_COLOR,
    TRAINING_END_STYLE,
    TRAINING_START_COLOR,
    TRAINING_START_STYLE,
    TRAIN_COLOR,
    VALIDATION_COLOR,
    classification_metric_sets,
    apply_academic_style,
    clear_figure_titles,
    configure_axis_limit_ticks,
    configure_count_colorbar,
    configure_dynamic_axis,
    configure_numeric_x_axis,
    dataset_plot_dir,
    feature_count_comparison_rows,
    feature_count_measurement_summary,
    plot_confusion_matrix,
    plot_feature_ranking,
    plot_final_classification_error,
    plot_final_train_test_metric_groups,
    plot_final_train_test_metrics,
    plot_parameter_train_test_error,
    plot_parameter_train_validation_error,
    plot_precision_recall_curve,
    plot_probability_bars,
    plot_probability_figure,
    plot_resource_usage,
    plot_resource_usage_separate,
    plot_roc_curve,
    plot_score_curve,
    plot_sample_probability_prediction,
    plot_training_time,
    plot_training_time_comparison,
    plotly_tick_format,
    probability_limits,
    resource_baseline_summary,
    save_dataset_figure,
    scientific_scale,
    shared_limits,
    training_time_measurement_summary,
)


def assert_equal_tick_intervals(ticks):
    """Assert that a numeric scale uses one constant tick interval."""
    differences = np.diff(np.asarray(ticks, dtype=float))
    assert differences.size > 0
    assert np.allclose(differences, differences[0])


def test_semantic_colors_are_distinct_and_stable():
    assert TRAIN_COLOR == "#4477AA"
    assert VALIDATION_COLOR == "#EE7733"
    assert [TRAIN_COLOR, VALIDATION_COLOR] == list(DATASET_COLORS.values())
    assert len(set(MODEL_COLORS.values())) == len(MODEL_COLORS)
    assert CLASS_COLORS == {"safe": "#4477AA", "phishing": "#EE7733"}
    assert HEATMAP_CMAPS == {"full": "Blues", "optimized": "Blues"}


def test_dataset_plot_dirs_are_separate(tmp_path: Path):
    full = dataset_plot_dir(tmp_path, "Pełny zbiór")
    optimized = dataset_plot_dir(tmp_path, "Zoptymalizowany zbiór")

    assert full != optimized
    assert full.name == "full"
    assert optimized.name == "optimized"
    assert full.is_dir()
    assert optimized.is_dir()


def test_dataset_figures_never_collide(tmp_path: Path):
    from matplotlib import pyplot as plt

    figure, axis = plt.subplots()
    axis.plot([0, 1], [0, 1])
    full = save_dataset_figure(figure, tmp_path, "Pełny zbiór", "curve.png")
    optimized = save_dataset_figure(
        figure, tmp_path, "Zoptymalizowany zbiór", "curve.png"
    )
    plt.close(figure)

    assert full != optimized
    assert full.is_file()
    assert optimized.is_file()
    assert full.parent.name == "pl"
    assert optimized.parent.name == "pl"
    assert (full.parent.parent / "eng" / full.name).is_file()
    assert (optimized.parent.parent / "eng" / optimized.name).is_file()
    assert not (full.parent.parent / "en" / full.name).exists()


def test_dataset_figure_can_be_saved_without_inline_display(tmp_path: Path):
    from matplotlib import pyplot as plt

    figure, axis = plt.subplots()
    figure_number = figure.number
    axis.plot([0, 1], [0, 1])

    output = save_dataset_figure(
        figure,
        tmp_path,
        "full",
        "details.png",
        display_in_notebook=False,
    )

    assert output.is_file()
    assert (output.parent.parent / "eng" / output.name).is_file()
    assert not plt.fignum_exists(figure_number)


def test_english_figure_copy_translates_scientific_labels():
    from matplotlib import pyplot as plt

    figure, axis = plt.subplots()
    axis.set_xlabel("Liczba próbek")
    axis.set_ylabel("Dokładność klasyfikacji")
    axis.plot([0, 1], [0, 1], label="Zbiór testowy")
    axis.legend()

    translated = english_matplotlib_copy(figure)
    translated_axis = translated.axes[0]

    assert translated_axis.get_xlabel() == "Number of samples"
    assert translated_axis.get_ylabel() == "Classification accuracy"
    assert translated_axis.get_legend().get_texts()[0].get_text() == "Test set"
    plt.close(figure)
    plt.close(translated)


def test_shared_limits_cover_every_series():
    limits = shared_limits(([0.18, 0.10], [0.16, 0.06]), floor=0.0)

    assert limits[0] <= 0.06
    assert limits[1] >= 0.18
    assert np.isfinite(limits).all()


def test_probability_limits_zoom_without_leaving_probability_domain():
    limits = probability_limits(([0.94, 0.96], [0.91, 0.95]))

    assert 0.0 <= limits[0] < 0.91
    assert 0.96 < limits[1] <= 1.0
    assert limits[1] - limits[0] >= 0.02
    assert limits[1] - limits[0] < 0.06


def test_figure_titles_are_removed_without_touching_axis_labels():
    from matplotlib import pyplot as plt

    figure, axis = plt.subplots()
    figure.suptitle("Title")
    axis.set_title("Axis title")
    axis.set_xlabel("Horizontal axis")
    clear_figure_titles(figure)

    assert axis.get_title() == ""
    assert figure._suptitle.get_text() == ""
    assert axis.get_xlabel() == "Horizontal axis"
    plt.close(figure)


def test_academic_style_uses_consistent_typography_and_restrained_axes():
    from matplotlib import pyplot as plt

    figure, axis = plt.subplots()
    axis.plot([0, 1], [0, 1], label="Seria")
    axis.legend()
    apply_academic_style(figure)

    assert axis.xaxis.label.get_fontfamily() == [ACADEMIC_FONT_FAMILY]
    assert not axis.spines["top"].get_visible()
    assert not axis.spines["right"].get_visible()
    assert not axis.get_legend().get_frame_on()
    assert figure.get_facecolor() == (1.0, 1.0, 1.0, 1.0)
    plt.close(figure)


def test_confusion_matrix_and_roc_helpers_create_independent_figures():
    from matplotlib import pyplot as plt

    confusion_figure, confusion_axis = plot_confusion_matrix(
        np.array([[8, 2], [1, 9]]), "Pełny zbiór"
    )
    roc_figure, roc_axis = plot_roc_curve(
        np.array([0, 0, 1, 1]),
        np.array([0.1, 0.4, 0.6, 0.9]),
        0.9,
        "Pełny zbiór",
    )

    assert confusion_figure is not roc_figure
    assert confusion_axis.figure is confusion_figure
    assert roc_axis.figure is roc_figure
    assert len(confusion_figure.axes) == 2
    assert len(roc_figure.axes) == 1
    assert roc_axis.get_xlim() == (0.0, 1.0)
    assert roc_axis.get_ylim() == (0.0, 1.0)
    assert roc_axis.lines[1].get_label() == "Klasyfikator losowy"
    translated = english_matplotlib_copy(confusion_figure)
    translated.canvas.draw()
    translated_axis = translated.axes[0]
    assert [text.get_text() for text in translated_axis.get_xticklabels()] == [
        "Non-phishing", "Phishing",
    ]
    assert [text.get_text() for text in translated_axis.get_yticklabels()] == [
        "Non-phishing", "Phishing",
    ]
    plt.close(confusion_figure)
    plt.close(roc_figure)
    plt.close(translated)


def test_precision_recall_curve_has_fixed_scale_and_prevalence_baseline():
    from matplotlib import pyplot as plt

    figure, axis = plot_precision_recall_curve(
        np.array([0, 0, 1, 1]),
        np.array([0.1, 0.4, 0.6, 0.9]),
        0.9,
        "Pełny zbiór",
    )

    assert axis.get_xlim() == (0.0, 1.0)
    assert axis.get_ylim() == (0.0, 1.0)
    assert axis.lines[1].get_label().startswith("Udział klasy phishing")
    plt.close(figure)


def test_summary_charts_do_not_mix_units_and_metric_bars_start_at_zero():
    from matplotlib import pyplot as plt

    metric_figure, metric_axis = plot_final_train_test_metrics(
        {"accuracy": 0.95, "precision": 0.9},
        {"accuracy": 0.93, "precision": 0.88},
    )
    time_figure, time_axis = plot_training_time(2.5, "Pełny zbiór")

    assert metric_axis.get_ylim() == (0.0, 1.0)
    assert 0.0 in metric_axis.get_yticks()
    assert time_axis.get_ylabel() == "Czas uczenia [s]"
    assert 0.0 in time_axis.get_yticks()
    for figure in (metric_figure, time_figure):
        plt.close(figure)


def test_feature_counts_are_compared_in_a_compact_table():
    rows = feature_count_comparison_rows(111, 73)

    assert rows[0] == {
        "Wariant danych": "Pełny zbiór",
        "Liczba cech": 111,
        "Redukcja liczby cech": 0,
        "Redukcja [%]": 0.0,
    }
    assert rows[1] == {
        "Wariant danych": "Zoptymalizowany zbiór",
        "Liczba cech": 73,
        "Redukcja liczby cech": 38,
        "Redukcja [%]": 34.23,
    }


def test_training_time_comparison_uses_two_directly_labelled_bars():
    from matplotlib import pyplot as plt

    figure, axis = plot_training_time_comparison({
        "Pełny zbiór": 12.5,
        "Zoptymalizowany zbiór": 7.25,
    })

    assert [bar.get_height() for bar in axis.patches] == [12.5, 7.25]
    assert axis.get_ylim()[0] == 0.0
    assert 0.0 in axis.get_yticks()
    assert [label.get_text() for label in axis.get_xticklabels()] == [
        "Pełny zbiór", "Zoptymalizowany zbiór",
    ]
    assert [text.get_text() for text in axis.texts] == ["12,50", "7,25"]
    assert [text.get_text() for text in axis.get_legend().get_texts()] == [
        "Pełny zbiór", "Zoptymalizowany zbiór",
    ]
    translated = english_matplotlib_copy(figure)
    translated.canvas.draw()
    assert [text.get_text() for text in translated.axes[0].get_xticklabels()] == [
        "Full dataset", "Optimized dataset",
    ]
    plt.close(figure)
    plt.close(translated)


def test_scalar_measure_summaries_define_timing_and_feature_scope():
    timing = training_time_measurement_summary()
    features = feature_count_measurement_summary()

    assert "time.perf_counter()" in timing
    assert "bezpośrednio przed wywołaniem model.fit(...)" in timing
    assert "serializacji modelu" in timing
    assert "sekwencyjnym wykonaniu notebooków" in timing
    assert "liczbę kolumn macierzy uczącej" in features
    assert "Zmienna docelowa nie jest liczona jako cecha" in features


def test_probability_bars_use_fixed_percentage_scale():
    from matplotlib import pyplot as plt

    figure, axis = plt.subplots()
    plot_probability_bars(
        axis, [3, 1], sample_label="Próbka; Etykieta rzeczywista: Phishing"
    )

    assert axis.get_xlim() == (0.0, 100.0)
    assert list(axis.get_xticks())[-1] == 100.0
    assert [bar.get_width() for bar in axis.patches] == [75.0, 25.0]
    assert all(bar.get_hatch() is None for bar in axis.patches)
    assert axis.texts[-1].get_text().startswith("Próbka;")
    plt.close(figure)


def test_probability_figure_contains_exactly_one_sample_axis():
    from matplotlib import pyplot as plt

    figure, axis = plot_probability_figure(
        [0.25, 0.75], sample_label="Próbka #7; Etykieta rzeczywista: Phishing"
    )

    assert figure.axes == [axis]
    assert len(axis.patches) == 2
    plt.close(figure)


def test_sample_probability_plot_shows_prediction_and_true_label():
    from matplotlib import pyplot as plt

    figure, axis = plot_sample_probability_prediction(
        0.75, 1, 42, decision_threshold=0.55,
    )

    assert axis.get_xlim() == (0.0, 100.0)
    assert axis.get_xlabel() == "Prawdopodobieństwo przewidywane przez model [%]"
    assert axis.get_ylabel() == "Klasa"
    assert [label.get_text() for label in axis.get_yticklabels()] == [
        "Bezpieczny", "Phishing — etykieta rzeczywista",
    ]
    assert [bar.get_width() for bar in axis.patches] == [25.0, 75.0]
    assert [text.get_text() for text in axis.texts[:2]] == ["25,00%", "75,00%"]
    assert "Wynik: poprawny" in axis.texts[-1].get_text()
    assert [patch.get_linewidth() for patch in axis.patches] == [1.0, 3.0]
    assert [text.get_text() for text in axis.get_legend().get_texts()] == [
        "Prawdopodobieństwo klasy „Bezpieczny”",
        "Prawdopodobieństwo phishingu",
        "Decyzja modelu: Phishing",
        "Próg decyzji phishing: 55,00%",
    ]
    plt.close(figure)


def test_sample_probability_plot_is_localized_in_english():
    from matplotlib import pyplot as plt

    figure, axis = plot_sample_probability_prediction(
        0.25, 0, "abc", decision_threshold=0.5, language="en",
    )

    assert axis.get_xlabel() == "Probability predicted by the model [%]"
    assert axis.get_ylabel() == "Class"
    assert [label.get_text() for label in axis.get_yticklabels()] == [
        "Non-phishing — actual label", "Phishing",
    ]
    assert [text.get_text() for text in axis.texts[:2]] == ["75.00%", "25.00%"]
    assert "Result: correct" in axis.texts[-1].get_text()
    assert [text.get_text() for text in axis.get_legend().get_texts()] == [
        "Non-phishing probability",
        "Phishing probability",
        "Model decision: Non-phishing",
        "Phishing decision threshold: 50.00%",
    ]
    plt.close(figure)


def test_sample_probability_plot_marks_incorrect_decision():
    from matplotlib import pyplot as plt

    figure, axis = plot_sample_probability_prediction(
        0.37, 1, 17, decision_threshold=0.55,
    )

    assert "Etykieta rzeczywista: Phishing" in axis.texts[-1].get_text()
    assert "Decyzja modelu: Bezpieczny" in axis.texts[-1].get_text()
    assert "Wynik: błędny" in axis.texts[-1].get_text()
    plt.close(figure)


def test_axis_ticks_use_equal_rounded_intervals_with_reserve():
    from matplotlib import pyplot as plt

    figure, axis = plt.subplots()
    axis.set_ylim(0.91, 0.987)
    configure_axis_limit_ticks(axis, "y")

    assert axis.get_ylim()[0] <= 0.91
    assert axis.get_ylim()[1] >= 0.987
    assert_equal_tick_intervals(axis.get_yticks())
    plt.close(figure)


def test_sample_count_y_axis_uses_only_integer_equal_ticks():
    from matplotlib import pyplot as plt

    figure, axis = plt.subplots()
    axis.set_ylabel("Liczba próbek")
    axis.set_ylim(0.0, 17.4)
    configure_axis_limit_ticks(axis, "y")

    ticks = axis.get_yticks()
    assert np.allclose(ticks, np.rint(ticks))
    assert_equal_tick_intervals(ticks)
    plt.close(figure)


def test_fractional_sample_counts_are_rejected():
    from matplotlib import pyplot as plt
    import pytest

    figure, axis = plt.subplots()
    axis.plot([10.0, 20.5], [0.7, 0.8])

    with pytest.raises(ValueError, match="Sample counts must be integers"):
        configure_numeric_x_axis(axis, "Liczba próbek")
    plt.close(figure)


def test_score_curve_rounds_average_fold_sizes_to_whole_samples():
    from matplotlib import pyplot as plt

    figure, axis = plot_score_curve(
        [100.2, 200.6], [0.8, 0.9], [0.01, 0.01],
        [0.75, 0.85], [0.02, 0.02], x_label="Liczba próbek",
    )

    assert np.array_equal(axis.lines[0].get_xdata(), [100, 201])
    assert np.array_equal(axis.lines[1].get_xdata(), [100, 201])
    plt.close(figure)


def test_sample_count_x_labels_never_contain_decimal_places():
    from matplotlib import pyplot as plt

    figure, axis = plt.subplots()
    axis.plot([3922, 7845, 11768, 15691, 19614], [0.7] * 5)
    configure_numeric_x_axis(axis, "Liczba próbek")
    figure.canvas.draw()

    labels = [label.get_text() for label in axis.get_xticklabels()]
    assert labels
    assert all("," not in label and "." not in label for label in labels)
    plt.close(figure)


def test_count_colorbar_uses_only_integer_ticks():
    from matplotlib import pyplot as plt
    import seaborn as sns

    figure, axis = plt.subplots()
    sns.heatmap(np.array([[0, 3], [1, 7]]), ax=axis)
    configure_count_colorbar(axis)
    figure.canvas.draw()

    ticks = figure.axes[1].get_yticks()
    assert np.allclose(ticks, np.rint(ticks))
    plt.close(figure)


def test_scientific_scale_has_equal_ticks_and_reserved_bounds():
    lower, upper, ticks = scientific_scale(
        [0.013, 0.087], include_zero=True, reserve_ratio=0.1
    )

    assert lower == 0.0
    assert upper > 0.087
    assert ticks[0] == lower
    assert ticks[-1] == upper
    assert_equal_tick_intervals(ticks)


def test_scientific_scale_can_be_symmetric_around_zero():
    lower, upper, ticks = scientific_scale(
        [-0.21, 0.47], symmetric=True, reserve_ratio=0.1
    )

    assert lower < -0.47
    assert upper > 0.47
    assert np.isclose(abs(lower), upper)
    assert 0.0 in ticks
    assert not np.any((ticks < 0.0) & np.isclose(ticks, 0.0))
    assert_equal_tick_intervals(ticks)


def test_compact_scale_does_not_expand_correlation_beyond_reserved_data():
    lower, upper, ticks = scientific_scale(
        [-0.46, 0.75],
        symmetric=True,
        reserve_ratio=0.08,
        target_intervals=4,
        rounded_bounds=False,
    )

    assert -1.0 < lower < -0.75
    assert 0.75 < upper < 1.0
    assert ticks[0] >= lower
    assert ticks[-1] <= upper
    assert_equal_tick_intervals(ticks)


def test_compact_scale_keeps_ticks_inside_individual_data_range():
    lower, upper, ticks = scientific_scale(
        [0.0, 111.0],
        include_zero=True,
        reserve_ratio=0.02,
        target_intervals=5,
        rounded_bounds=False,
    )

    assert lower == 0.0
    assert 111.0 < upper < 115.0
    assert ticks[-1] <= upper
    assert ticks[-1] < 125.0


def test_scientific_scale_accepts_predefined_tick_interval_and_anchor():
    lower, upper, ticks = scientific_scale(
        [0.5, 3.5],
        reserve_ratio=0.0,
        rounded_bounds=False,
        tick_step=1.0,
        tick_anchor=0.5,
    )

    assert (lower, upper) == (0.5, 3.5)
    assert np.array_equal(ticks, [0.5, 1.5, 2.5, 3.5])


def test_scientific_scale_rejects_invalid_predefined_tick_interval():
    import pytest

    with pytest.raises(ValueError, match="finite positive"):
        scientific_scale([0.0, 1.0], tick_step=0.0)


def test_score_curve_uses_cross_validation_label():
    from matplotlib import pyplot as plt

    figure, axis = plot_score_curve(
        [1, 2],
        [0.7, 0.8],
        [0.01, 0.01],
        [0.6, 0.75],
        [0.02, 0.02],
        x_label="Rozmiar zbioru",
        y_label="Wynik",
    )

    assert [line.get_label() for line in axis.lines] == [
        "Zbiór uczący", "Zbiór walidacyjny"
    ]
    assert [text.get_text() for text in axis.get_legend().get_texts()] == [
        "Zbiór uczący",
        "Zbiór walidacyjny",
        "± odchylenie standardowe",
    ]
    assert 0.0 <= axis.get_ylim()[0] < 0.58
    assert axis.get_ylim()[1] > 0.81
    assert axis.get_xlim() == (1.0, 2.0)
    assert_equal_tick_intervals(axis.get_xticks())
    translated = english_matplotlib_copy(figure)
    translated_legend = translated.axes[0].get_legend().get_texts()
    assert [text.get_text() for text in translated_legend] == [
        "Training set", "Validation set", "± standard deviation",
    ]
    plt.close(figure)
    plt.close(translated)


def test_roc_score_curve_uses_a_readable_bounded_scale():
    from matplotlib import pyplot as plt

    figure, axis = plot_score_curve(
        [100, 200], [0.97, 0.98], [0.005, 0.004],
        [0.94, 0.96], [0.01, 0.008],
        x_label="Liczba próbek",
    )

    lower, upper = axis.get_ylim()
    assert 0.0 < lower < 0.94
    assert 0.98 < upper <= 1.0
    plt.close(figure)


def test_final_classification_error_distinguishes_train_and_real_test():
    from matplotlib import pyplot as plt

    figure, axis = plot_final_classification_error(0.01, 0.03)

    assert [tick.get_text() for tick in axis.get_xticklabels()] == [
        "Zbiór uczący", "Zbiór testowy"
    ]
    assert axis.get_ylabel() == "Współczynnik błędu klasyfikacji"
    assert [bar.get_height() for bar in axis.patches] == [0.01, 0.03]
    plt.close(figure)


def test_final_train_test_metrics_use_one_consistent_grouped_figure():
    from matplotlib import pyplot as plt

    train = {
        "accuracy": 0.99, "precision": 0.98, "recall": 0.97,
        "false_positive_rate": 0.02, "error_rate": 0.01,
        "f1": 0.975, "roc_auc": 0.995, "bce": 0.08,
    }
    test = {
        "accuracy": 0.94, "precision": 0.93, "recall": 0.92,
        "false_positive_rate": 0.07, "error_rate": 0.06,
        "f1": 0.925, "roc_auc": 0.96, "bce": 0.12,
    }

    figure, axis = plot_final_train_test_metrics(train, test)

    assert len(figure.axes) == 1
    assert axis.get_ylabel() == "Wartość miary"
    assert [tick.get_text() for tick in axis.get_xticklabels()] == [
        "Dokładność klasyfikacji", "Precyzja", "Czułość",
        "Odsetek fałszywych alarmów (FPR)",
        "Współczynnik błędu klasyfikacji",
        "Miara F1", "Pole pod krzywą ROC",
    ]
    assert [bar.get_height() for bar in axis.patches[:7]] == [
        train[key] for key in (
            "accuracy", "precision", "recall",
            "false_positive_rate", "error_rate", "f1", "roc_auc",
        )
    ]
    assert [bar.get_height() for bar in axis.patches[7:]] == [
        test[key] for key in (
            "accuracy", "precision", "recall",
            "false_positive_rate", "error_rate", "f1", "roc_auc",
        )
    ]
    assert [text.get_text() for text in axis.get_legend().get_texts()] == [
        "Zbiór uczący", "Zbiór testowy",
    ]
    assert axis.get_legend().get_bbox_to_anchor()._bbox.y0 > 1.0
    translated = english_matplotlib_copy(figure)
    translated.canvas.draw()
    translated_axis = translated.axes[0]
    assert [tick.get_text() for tick in translated_axis.get_xticklabels()] == [
        "Classification accuracy", "Precision", "Recall",
        "False-positive rate (FPR)", "Classification error rate",
        "F1 score", "Area under the ROC curve",
    ]
    assert [text.get_text() for text in translated_axis.get_legend().get_texts()] == [
        "Training set", "Test set",
    ]
    plt.close(figure)
    plt.close(translated)


def test_final_metrics_are_split_into_two_readable_groups():
    from matplotlib import pyplot as plt

    train = {
        "accuracy": 0.99, "precision": 0.98, "recall": 0.97,
        "false_positive_rate": 0.02, "error_rate": 0.01,
        "f1": 0.975, "roc_auc": 0.995,
    }
    test = {
        "accuracy": 0.94, "precision": 0.93, "recall": 0.92,
        "false_positive_rate": 0.07, "error_rate": 0.06,
        "f1": 0.925, "roc_auc": 0.96,
    }

    figures = plot_final_train_test_metric_groups(train, test)

    assert list(figures) == ["jakosc", "bledy"]
    basic_labels = [
        tick.get_text() for tick in figures["jakosc"].axes[0].get_xticklabels()
    ]
    error_labels = [
        tick.get_text()
        for tick in figures["bledy"].axes[0].get_xticklabels()
    ]
    assert basic_labels == [
        "Dokładność klasyfikacji", "Precyzja", "Czułość", "Miara F1",
        "Pole pod krzywą ROC",
    ]
    assert error_labels == [
        "Odsetek fałszywych alarmów (FPR)",
        "Współczynnik błędu klasyfikacji",
    ]
    assert figures["bledy"].axes[0].get_ylim()[1] < 0.1
    for figure in figures.values():
        plt.close(figure)


def test_classification_metric_sets_use_one_evaluation_result():
    result = {
        "train_accuracy": 0.99, "accuracy": 0.94,
        "train_precision": 0.98, "precision": 0.93,
        "train_recall": 0.97, "recall": 0.92,
        "train_false_positive_rate": 0.02,
        "false_positive_rate": 0.07,
        "train_error_rate": 0.01, "test_error_rate": 0.06,
        "train_f1": 0.975, "f1": 0.925,
        "train_auc_roc": 0.995, "auc_roc": 0.96,
    }

    train_metrics, test_metrics = classification_metric_sets(result)

    assert train_metrics["accuracy"] == result["train_accuracy"]
    assert test_metrics["accuracy"] == result["accuracy"]
    assert train_metrics["error_rate"] == result["train_error_rate"]
    assert test_metrics["error_rate"] == result["test_error_rate"]
    assert train_metrics["roc_auc"] == result["train_auc_roc"]
    assert test_metrics["roc_auc"] == result["auc_roc"]


def test_resource_baseline_summary_explains_reference_level():
    class Monitor:
        num_cores = 8
        phases = ["baseline", "baseline", "training"]
        cpu_percent = [1.0, 3.0, 20.0]
        ram_gb = [0.5, 0.7, 1.5]

    summary = resource_baseline_summary(Monitor())

    assert "mediana 2 próbek" in summary
    assert "CPU: 2,00%" in summary
    assert "8 rdzeni logicznych" in summary
    assert "RAM: 0,60 GB" in summary
    assert "pomiar minus poziom bazowy" in summary


def test_sample_count_score_curve_starts_near_first_measurement():
    from matplotlib import pyplot as plt

    figure, axis = plot_score_curve(
        [100, 200, 300], [0.9, 0.92, 0.94], [0.01] * 3,
        [0.85, 0.88, 0.9], [0.02] * 3,
        x_label="Liczba próbek",
    )

    figure.canvas.draw()
    assert axis.get_xlim()[0] < 100.0
    assert axis.get_xlim()[1] > 300.0
    assert axis.get_xticks()[0] == 100.0
    assert axis.get_xticks()[-1] == 300.0
    assert_equal_tick_intervals(axis.get_xticks())
    plt.close(figure)


def test_sample_count_curve_uses_equal_intervals_on_both_numeric_axes():
    from matplotlib import pyplot as plt

    figure, axis = plot_score_curve(
        [101, 253, 502, 997], [0.78, 0.84, 0.88, 0.91],
        [0.02] * 4, [0.72, 0.8, 0.85, 0.89], [0.03] * 4,
        x_label="Liczba próbek w zbiorze testowym",
    )

    assert axis.get_xlim()[0] < 101.0
    assert axis.get_xlim()[1] > 997.0
    assert_equal_tick_intervals(axis.get_xticks())
    assert_equal_tick_intervals(axis.get_yticks())
    assert np.equal(np.mod(axis.get_xticks(), 1.0), 0.0).all()
    plt.close(figure)


def test_linear_parameter_c_axis_marks_zero_without_fake_measurement():
    from matplotlib import pyplot as plt

    values = np.linspace(0.01, 100.0, 20)
    figure, axis = plot_score_curve(
        values, [0.9] * 20, [0.01] * 20,
        [0.88] * 20, [0.01] * 20,
        x_label="Wartość parametru C (C > 0; odwrotność siły regularyzacji)",
    )

    figure.canvas.draw()
    labels = [label.get_text() for label in axis.get_xticklabels()]
    assert axis.get_xlim()[0] == 0.0
    assert axis.get_xlim()[1] > 100.0
    assert axis.get_xticks()[0] == 0.0
    assert axis.lines[0].get_xdata()[0] == 0.01
    assert_equal_tick_intervals(axis.get_xticks())
    assert all(not label.endswith(",00") for label in labels)
    plt.close(figure)


def test_logarithmic_parameter_c_axis_labels_every_measured_value():
    from matplotlib import pyplot as plt

    values = np.logspace(-2, 3, 6)
    figure, axis = plot_score_curve(
        values, [0.9] * 6, [0.01] * 6,
        [0.88] * 6, [0.01] * 6,
        x_label="Wartość parametru C (C > 0; odwrotność siły regularyzacji)",
        logarithmic_x=True,
    )

    assert np.array_equal(axis.get_xticks(), values)
    assert 0.0 not in axis.get_xticks()
    formatter = axis.xaxis.get_major_formatter()
    assert formatter(1000.0, 0) == "1000"
    plt.close(figure)


def test_learning_curve_uses_dense_scale_without_excessive_end_space():
    from matplotlib import pyplot as plt

    x_values = np.linspace(3900, 39000, 10)
    figure, axis = plot_score_curve(
        x_values, [0.98] * 10, [0.001] * 10,
        [0.97] * 10, [0.001] * 10,
        x_label="Liczba próbek",
    )

    assert axis.get_xlim()[0] < 3900.0
    assert axis.get_xlim()[1] > 39000.0
    assert_equal_tick_intervals(axis.get_xticks())
    assert axis.get_ylim()[0] > 0.95
    figure.canvas.draw()
    labels = [label.get_text() for label in axis.get_yticklabels()]
    assert all("×10" not in label for label in labels)
    assert all(label.startswith("0,") for label in labels)
    plt.close(figure)


def test_feature_ranking_matches_statistical_analysis_color():
    import pandas as pd
    from matplotlib import pyplot as plt
    from matplotlib.colors import to_rgba

    frame = pd.DataFrame({
        "Cecha": ['Ścieżka: Liczba "/"', "Domena: Długość"],
        "Ważność": [0.8, 0.4],
    })
    figure, axis = plot_feature_ranking(
        frame, "Zoptymalizowany zbiór",
        x_column="Ważność", x_label="Ważność cechy"
    )

    colors = {tuple(patch.get_facecolor()) for patch in axis.patches}
    assert colors == {to_rgba(CLASS_COLORS["safe"])}
    assert [text.get_text() for text in axis.get_legend().get_texts()] == [
        "Zoptymalizowany zbiór"
    ]
    translated = english_matplotlib_copy(figure)
    translated.canvas.draw()
    assert [text.get_text() for text in translated.axes[0].get_yticklabels()] == [
        'Path: Number of "/" characters', "Domain: Length",
    ]
    assert translated.axes[0].get_ylabel() == "Feature"
    plt.close(figure)
    plt.close(translated)


def test_parameter_error_curve_uses_real_test_label():
    from matplotlib import pyplot as plt

    figure, axis = plot_parameter_train_test_error(
        [1, 2, 3], [0.10, 0.08, 0.07], [0.12, 0.10, 0.11],
        x_label="Parametr",
    )

    assert [line.get_label() for line in axis.lines] == [
        "Zbiór uczący", "Zbiór testowy"
    ]
    assert axis.get_ylabel() == "Współczynnik błędu klasyfikacji"
    assert axis.get_xlim() == (1.0, 3.0)
    assert_equal_tick_intervals(axis.get_xticks())
    plt.close(figure)


def test_public_numeric_axis_uses_observed_ticks_and_requested_zero():
    from matplotlib import pyplot as plt

    figure, axis = plt.subplots()
    axis.plot([250, 500, 1000], [0.8, 0.85, 0.9])
    configure_numeric_x_axis(
        axis, "Liczba próbek w zbiorze testowym", include_zero=True
    )

    assert axis.get_xlim()[0] == 0.0
    assert axis.get_xlim()[1] > 1000.0
    assert 0.0 in axis.get_xticks()
    assert_equal_tick_intervals(axis.get_xticks())
    plt.close(figure)


def test_numeric_axis_accepts_predefined_tick_interval():
    from matplotlib import pyplot as plt

    figure, axis = plt.subplots()
    axis.plot([2.0, 12.0], [0.8, 0.9])
    configure_numeric_x_axis(
        axis, "Parametr", tick_step=2.0, tick_anchor=0.0
    )

    assert_equal_tick_intervals(axis.get_xticks())
    assert np.allclose(np.mod(axis.get_xticks(), 2.0), 0.0)
    plt.close(figure)


def test_numeric_x_axis_labels_use_at_most_two_decimal_places():
    from matplotlib import pyplot as plt

    figure, axis = plt.subplots()
    axis.plot([0.0, 1.0], [0.8, 0.9])
    configure_numeric_x_axis(
        axis, "Parametr", tick_step=1.0 / 3.0, tick_anchor=0.0
    )

    formatter = axis.xaxis.get_major_formatter()
    assert formatter(0.0) == "0"
    assert formatter(1.0 / 3.0) == "0,33"
    plt.close(figure)


def test_numeric_x_axis_uses_full_numbers_for_integer_ticks():
    from matplotlib import pyplot as plt

    figure, axis = plt.subplots()
    axis.plot([0, 10], [0.8, 0.9])
    configure_numeric_x_axis(axis, "Parametr", tick_step=2.0)

    formatter = axis.xaxis.get_major_formatter()
    assert formatter(2.0) == "2"
    plt.close(figure)


def test_small_x_axis_steps_use_precise_scientific_notation():
    from matplotlib import pyplot as plt

    figure, axis = plt.subplots()
    axis.plot([0.0, 0.003], [0.8, 0.9])
    configure_numeric_x_axis(axis, "Parametr", tick_step=0.001)

    formatter = axis.xaxis.get_major_formatter()
    assert formatter(0.001) == "1,00×10⁻³"
    assert plotly_tick_format([0.0, 0.001, 0.002]) == ".2e"
    plt.close(figure)


def test_plotly_tick_format_uses_integer_or_two_decimal_labels():
    assert plotly_tick_format([0.0, 1.0, 2.0]) == ".0f"
    assert plotly_tick_format([0.0, 0.25, 0.5]) == ".2~f"


def test_parameter_validation_error_curve_uses_holdout_label():
    from matplotlib import pyplot as plt

    figure, axis = plot_parameter_train_validation_error(
        [1, 2], [0.10, 0.08], [0.12, 0.10], x_label="Parametr"
    )

    assert [line.get_label() for line in axis.lines] == [
        "Zbiór uczący", "Zbiór walidacyjny"
    ]
    assert 0.0 <= axis.get_ylim()[0] <= 0.08
    assert axis.get_ylim()[1] >= 0.12
    plt.close(figure)


def test_dynamic_axis_zooms_to_positive_data_without_negative_scale():
    from matplotlib import pyplot as plt

    figure, axis = plt.subplots()
    axis.plot([1, 2, 3], [0.81, 0.84, 0.86])
    configure_dynamic_axis(axis, "y", ([0.81, 0.84, 0.86],), floor=0.0)

    lower, upper = axis.get_ylim()
    assert 0.0 < lower < 0.81
    assert upper > 0.86
    assert np.all(axis.get_yticks() >= 0.0)
    assert_equal_tick_intervals(axis.get_yticks())
    plt.close(figure)


def test_dynamic_axis_keeps_meaningful_negative_observations():
    from matplotlib import pyplot as plt

    figure, axis = plt.subplots()
    values = [-0.35, -0.10, 0.20]
    axis.plot(values)
    configure_dynamic_axis(axis, "y", (values,))

    assert axis.get_ylim()[0] < -0.35
    assert axis.get_ylim()[1] > 0.20
    assert_equal_tick_intervals(axis.get_yticks())
    plt.close(figure)


def test_scientific_scale_never_adds_negative_ticks_to_nonnegative_data():
    lower, upper, ticks = scientific_scale(
        [0.0, 0.003], reserve_ratio=0.2
    )

    assert lower == 0.0
    assert upper > 0.003
    assert np.all(ticks >= 0.0)


def test_resource_usage_axes_match_observed_time_range():
    from matplotlib import pyplot as plt

    class Monitor:
        interval = 0.1
        duration = 1.25
        timestamps = [0.1, 0.6, 1.1]
        cpu_percent = [4.0, 10.0, 7.0]
        ram_gb = [0.2, 0.3, 0.25]

    stats = {
        "cpu": {"mean": 7.0},
        "ram": {"mean": 0.25},
    }
    figure, axes = plot_resource_usage(Monitor(), stats, "Pełny zbiór")

    for axis in axes:
        assert axis.get_xlim()[0] == 0.0
        assert axis.get_xlim()[1] == 1.0
        assert not axis.collections
        assert axis.lines[0].get_xdata()[0] == 0.0
        assert axis.lines[0].get_xdata()[-1] == 1.0
        assert_equal_tick_intervals(axis.get_xticks())
        assert_equal_tick_intervals(axis.get_yticks())
        assert axis.get_ylim()[0] == 0.0
        assert 0.0 in axis.get_xticks()
        assert 0.0 in axis.get_yticks()
    assert axes[0].lines[0].get_ydata()[0] == 0.0
    assert axes[1].lines[0].get_ydata()[0] == 0.0
    assert axes[0].lines[0].get_zorder() > axes[0].spines["bottom"].get_zorder()
    assert axes[0].lines[0].get_clip_on() is False
    plt.close(figure)


def test_resource_usage_plot_uses_training_window_and_rebases_time():
    from matplotlib import pyplot as plt

    class Monitor:
        interval = 0.1
        duration = 1.5
        training_start_time = 0.5
        training_end_time = 1.0
        training_duration = 0.5
        timestamps = [0.1, 0.5, 0.7, 1.0, 1.4]
        phases = ["baseline", "training", "training", "training", "post_training"]
        cpu_percent = [4.0, 20.0, 30.0, 25.0, 3.0]
        ram_gb = [1.0, 2.0, 2.1, 2.2, 3.0]

    stats = {
        "cpu": {"mean": 25.0},
        "ram": {"mean": 2.1},
    }
    figure, axes = plot_resource_usage(Monitor(), stats, "Pełny zbiór")

    assert np.allclose(axes[0].lines[0].get_xdata(), [0.0, 0.2, 0.5])
    assert np.allclose(axes[1].lines[0].get_xdata(), [0.0, 0.2, 0.5])
    assert max(axes[0].lines[0].get_ydata()) == 26.0
    assert np.isclose(max(axes[1].lines[0].get_ydata()), 1.2)
    plt.close(figure)


def test_resource_mean_matches_plotted_baseline_delta():
    from types import SimpleNamespace
    from matplotlib import pyplot as plt

    monitor = SimpleNamespace(
        training_start_time=0.5, training_end_time=1.0,
        timestamps=[0.1, 0.5, 0.7, 1.0],
        phases=["baseline", "training", "training", "training"],
        cpu_percent=[4.0, 4.0, 4.0, 4.0],
        ram_gb=[1.0, 0.8, 1.2, 0.9],
    )
    stats = {"cpu": {"mean": 4.0}, "ram": {"mean": 0.97}}
    figure, axes = plot_resource_usage(monitor, stats, "Pełny zbiór")
    mean_line = axes[1].lines[1]
    assert np.isclose(mean_line.get_ydata()[0], 0.0666666667)
    assert axes[1].texts[0].get_text() == "Średnia: 0,07 GB"
    plt.close(figure)


def test_zero_resource_mean_has_visible_axis_margin():
    from types import SimpleNamespace
    from matplotlib import pyplot as plt

    monitor = SimpleNamespace(
        training_start_time=0.5, training_end_time=1.0,
        timestamps=[0.1, 0.5, 0.7],
        phases=["baseline", "training", "training"],
        cpu_percent=[4.0, 4.0, 4.0],
        ram_gb=[1.0, 1.0, 1.0],
    )
    stats = {"cpu": {"mean": 4.0}, "ram": {"mean": 1.0}}
    figure, axes = plot_resource_usage(monitor, stats, "Pełny zbiór")
    assert axes[1].lines[1].get_ydata()[0] == 0.0
    assert axes[1].get_ylim()[0] < 0.0
    plt.close(figure)


def test_complete_resource_plot_does_not_create_synthetic_samples():
    from matplotlib import pyplot as plt

    class Monitor:
        duration = 2.0
        monitoring_duration = 2.0
        training_start_time = 0.5
        training_end_time = 1.0
        training_duration = 0.5
        timestamps = [0.1, 0.5, 0.8, 1.0, 1.4]
        phases = ["baseline", "training", "training", "training", "post_training"]
        cpu_percent = [4.0, 20.0, 25.0, 18.0, 3.0]
        ram_gb = [1.0, 2.0, 2.1, 2.0, 1.1]

    figure, axes = plot_resource_usage(
        Monitor(), {"cpu": {"mean": 20.0}, "ram": {"mean": 2.0}},
        "Pełny zbiór", include_post_training=True,
    )

    assert np.allclose(axes[0].lines[0].get_xdata(), [0.0, 0.4, 0.7, 0.9, 1.3])
    assert np.allclose(axes[0].lines[0].get_ydata(), [0.0, 16.0, 21.0, 14.0, 0.0])
    assert axes[0].get_xlim()[0] == 0.0
    assert np.isclose(axes[0].get_xlim()[1], 1.3)
    assert np.isclose(np.diff(axes[0].get_xticks())[0], 0.2)
    assert axes[0].get_ylim()[0] == 0.0
    finalization = [
        patch for patch in axes[0].patches
        if patch.get_label() == "Zapis modelu i zakończenie procesu"
    ]
    assert len(finalization) == 1
    plt.close(figure)


def test_resource_usage_can_be_saved_as_separate_cpu_and_ram_figures():
    from matplotlib import pyplot as plt

    class Monitor:
        interval = 0.1
        duration = 1.5
        monitoring_duration = 1.5
        training_start_time = 0.5
        training_end_time = 1.0
        training_duration = 0.5
        timestamps = [0.0, 0.5, 0.7, 1.0, 1.5]
        phases = ["baseline", "training", "training", "training", "post_training"]
        cpu_percent = [2.0, 20.0, 30.0, 25.0, 0.0]
        ram_gb = [1.0, 2.0, 2.1, 2.0, 0.0]

    figures = plot_resource_usage_separate(
        Monitor(),
        {"cpu": {"mean": 25.0}, "ram": {"mean": 2.1}},
        "Pełny zbiór",
        include_post_training=True,
    )

    assert len(figures) == 2
    assert all(len(figure.axes) == 1 for figure in figures)
    assert figures[0].axes[0].get_ylabel() == (
        "Zmiana wykorzystania procesora względem poziomu bazowego [%]"
    )
    assert figures[1].axes[0].get_ylabel() == (
        "Zmiana wykorzystania pamięci RAM względem poziomu bazowego [GB]"
    )
    cpu_legend = [
        text.get_text() for text in figures[0].axes[0].get_legend().get_texts()
    ]
    assert "Zmiana względem poziomu bazowego" in cpu_legend
    assert "Początek uczenia" in cpu_legend
    assert "Koniec uczenia / początek zapisu modelu" in cpu_legend
    assert "Zapis modelu i zakończenie procesu" in cpu_legend
    finalization = next(
        patch for patch in figures[0].axes[0].patches
        if patch.get_label() == "Zapis modelu i zakończenie procesu"
    )
    assert finalization.get_alpha() >= 0.1
    assert finalization.get_linewidth() > 0.0
    assert figures[0].axes[0].lines[0].get_drawstyle() == "steps-post"
    assert figures[1].axes[0].lines[0].get_drawstyle() == "default"
    start_marker = next(
        line for line in figures[0].axes[0].lines
        if line.get_label() == "Początek uczenia"
    )
    end_marker = next(
        line for line in figures[0].axes[0].lines
        if line.get_label() == "Koniec uczenia / początek zapisu modelu"
    )
    assert start_marker.get_color() == TRAINING_START_COLOR
    assert end_marker.get_color() == TRAINING_END_COLOR
    assert start_marker.get_linestyle() == TRAINING_START_STYLE
    assert end_marker.get_linestyle() == TRAINING_END_STYLE
    assert TRAINING_START_COLOR not in DATASET_COLORS.values()
    assert TRAINING_END_COLOR not in DATASET_COLORS.values()
    assert len(figures[0].axes[0].lines) == 4
    assert figures[0].axes[0].lines[0].get_zorder() > (
        figures[0].axes[0].spines["bottom"].get_zorder()
    )
    mean_line = next(
        line for line in figures[0].axes[0].lines
        if line.get_label() == "Średnia podczas uczenia"
    )
    assert np.allclose(mean_line.get_ydata(), [23.0, 23.0])
    assert mean_line.get_color() == "#228833"
    assert mean_line.get_zorder() > figures[0].axes[0].spines["bottom"].get_zorder()
    assert mean_line.get_clip_on() is False
    assert figures[0].axes[0].get_ylim()[0] == 0.0
    assert figures[0].axes[0].texts[0].get_text() == "Średnia: 23%"
    assert figures[0].axes[0].texts[0].get_ha() == "center"
    assert figures[0].axes[0].texts[0].get_fontweight() == "bold"
    assert figures[0].axes[0].texts[0].get_bbox_patch().get_linewidth() > 0.0
    assert figures[1].axes[0].texts[0].get_text() == "Średnia: 1,03 GB"
    for figure in figures:
        plt.close(figure)
