"""Create Polish and English variants of notebook figures."""

from __future__ import annotations

import copy
import re
import warnings
from pathlib import Path
from typing import Any

from matplotlib.category import StrCategoryFormatter
from matplotlib.text import Text
from matplotlib.ticker import FixedFormatter, FuncFormatter

from utils.feature_names import FEATURE_NAME_TRANSLATIONS


LANGUAGE_FOLDERS = ("pl", "eng")
POLISH_NUMBER = re.compile(
    r"^(?P<sign>[−-]?)(?P<integer>\d{1,3}(?: \d{3})*|\d+)"
    r"(?P<decimal>,\d+)?(?P<suffix>%?)$"
)
POLISH_DECIMAL_FRAGMENT = re.compile(r"(?<!\d)(\d+),(\d+)(?!\d)")
POLISH_GROUPED_NUMBER_FRAGMENT = re.compile(
    r"(?<!\d)([−-]?\d{1,3}(?: \d{3})+)(?!\d)"
)

POLISH_TO_ENGLISH = {
    **FEATURE_NAME_TRANSLATIONS,
    "Współczynnik<br>korelacji Pearsona": (
        "Pearson correlation<br>coefficient"
    ),
    "Współczynnik niezbalansowania (major/minor)": (
        "Imbalance ratio (majority/minority)"
    ),
    "Udział próbek klasy": "Within-class share",
    "Udział wartości -1": "Share of -1 values",
    "Czas trwania pomiaru": "Measurement duration",
    "Odchylenie standardowe": "Standard deviation",
    "odchylenie standardowe": "standard deviation",
    "Etykieta rzeczywista": "Actual label",
    "etykieta rzeczywista": "actual label",
    "Decyzja modelu": "Model decision",
    "Próg decyzji phishing": "Phishing decision threshold",
    "Wynik": "Result",
    "poprawny": "correct",
    "błędny": "incorrect",
    "Waga ujemna": "Negative weight",
    "Waga dodatnia": "Positive weight",
    "brak phishingu": "non-phishing",
    "Funkcja aktywacji": "Activation function",
    "Funkcja decyzyjna": "Decision function",
    "Hiperpłaszczyzna": "Hyperplane",
    "Klasyfikator losowy": "Random classifier",
    "Kombinacja liniowa": "Linear combination",
    "Mierzona charakterystyka modelu": "Measured model characteristic",
    "Regresja Logistyczna": "Logistic Regression",
    "Rozmiar warstw ukrytych": "Hidden layer sizes",
    "Liczba epok uczenia": "Number of training epochs",
    "Parametr gamma": "Gamma parameter",
    "bez balansowania": "without rebalancing",
    "bezpieczne": "non-phishing",
    "Bezpieczny": "Non-phishing",
    "Dokladnosc": "Accuracy",
    "Czulosc": "Recall",
    "Bledne": "Misclassified",
    "Rozklad": "Class distribution",
    "Zbior": "Dataset",
    "Metryka": "Metric",
    "Pelny": "Full",
    "Zoptym": "Optimized",
    "Pełny": "Full",
    "Zoptymalizowany": "Optimized",
    "Redukcja liczby cech": "Feature-count reduction",
    "Redukcja": "Reduction",
    "Poprawnie sklasyfikowane": "Correctly classified",
    "Udział wyników klasyfikacji [%]": "Share of classification outcomes [%]",
    "Wariant balansowania (średnia precyzja)": (
        "Balancing variant (average precision)"
    ),
    "Algorytm klasyfikacji i proporcja klas": (
        "Classification algorithm and class proportion"
    ),
    "Łączna liczba błędów": "Total number of errors",
    "Liczba błędów": "Number of errors",
    "Poziom bazowy": "Baseline",
    "Transformacja": "Transformation",
    "Wariant danych": "Data variant",
    "Wyjście w [0,1]": "Output in [0,1]",
    "miara F1": "F1 score",
    "Drzewa": "Trees",
    "Rozmiar": "Size",
    "cech": "features",
    "Parametry": "Parameters",
    "Warstwy": "Layers",
    "Klasa": "Class",
    "kontrolowany rozkład odpowiadający oryginalnemu korpusowi": (
        "controlled distribution matching the original corpus"
    ),
    "naturalny rozkład oryginalnego korpusu": (
        "natural distribution of the original corpus"
    ),
    "Niezmieniony zbiór testowy": "Unchanged test set",
    "Źródło niezbalansowania": "Source of imbalance",
    "syntetyczne nadpróbkowanie": "synthetic oversampling",
    "niedopróbkowanie": "undersampling",
    "nadpróbkowanie": "oversampling",
    "Warstwa wejściowa": "Input layer",
    "najważniejszych cech": "most important features",
    "Parametry modelu niedostępne": "Model parameters unavailable",
    "Wektory nośne": "Support vectors",
    "średnia głębokość": "mean depth",
    "średnia liczba liści": "mean number of leaves",
    "Głębokość": "Depth",
    "Liście": "Leaves",
    "udział klas": "class share",
    "Wynik ważony": "Weighted score",
    "Pewność błędu": "Error confidence",
    "Indeks próbki": "Sample index",
    "Prawdopodobieństwo": "Probability",
    "Klasa docelowa": "Target class",
    "Współczynnik korelacji": "Correlation coefficient",
    "Udział klasy phishing": "Phishing class prevalence",
    "pole pod krzywą ROC": "area under the ROC curve",
    "średnia precyzja": "average precision",
    "Udział próbek [%]": "Share of samples [%]",
    "Faza": "Phase",
    "próbki": "samples",
    "próbek": "samples",
    "fałsz": "false",
    "pełny": "full",
    "Wartość parametru C (C > 0; odwrotność siły regularyzacji)": (
        "C parameter value (C > 0; inverse regularization strength)"
    ),
    "Zmiana wykorzystania procesora względem poziomu bazowego [%]": (
        "Change in CPU usage relative to baseline [%]"
    ),
    "Zmiana wykorzystania pamięci RAM względem poziomu bazowego [GB]": (
        "Change in RAM usage relative to baseline [GB]"
    ),
    "Odsetek błędnie sklasyfikowanych próbek [%]": (
        "Misclassified samples [%]"
    ),
    "Łączna liczba błędnie sklasyfikowanych próbek": (
        "Total number of misclassified samples"
    ),
    "Liczba błędnie sklasyfikowanych próbek": (
        "Number of misclassified samples"
    ),
    "Wartość bezwzględna współczynnika modelu": (
        "Absolute model coefficient value"
    ),
    "Liczba drzew decyzyjnych w modelu XGBoost": (
        "Number of decision trees in the XGBoost model"
    ),
    "Udział próbek klasy w pokazanym zakresie [%]": (
        "Share of class samples in the displayed range [%]"
    ),
    "Udział próbek z wartością -1 [%]": (
        "Share of samples with value -1 [%]"
    ),
    "Współczynnik korelacji Pearsona": "Pearson correlation coefficient",
    "Przewidywane prawdopodobieństwo klasy [%]": (
        "Predicted class probability [%]"
    ),
    "Odsetek wyników fałszywie dodatnich": "False-positive rate",
    "Odsetek wyników prawdziwie dodatnich": "True-positive rate",
    "Odsetek fałszywych alarmów (FPR)": "False-positive rate (FPR)",
    "Zapis modelu i zakończenie procesu": "Model saving and process completion",
    "Koniec uczenia / początek zapisu modelu": (
        "Training end / model saving start"
    ),
    "Zbiór użyty do obliczenia miary": "Dataset used to calculate the metric",
    "Wartość miary jakości klasyfikacji": "Classification metric value",
    "Współczynnik błędu klasyfikacji": "Classification error rate",
    "Prawdopodobieństwo braku phishingu": "Non-phishing probability",
    "Prawdopodobieństwo phishingu": "Phishing probability",
    "Rzeczywista etykieta próbki": "Actual sample label",
    "Suma |wag| pierwszej warstwy": "Sum of first-layer absolute weights",
    "Maksymalna głębokość drzewa": "Maximum tree depth",
    "Liczba drzew decyzyjnych": "Number of decision trees",
    "Algorytm klasyfikacji": "Classification algorithm",
    "Miara jakości klasyfikacji": "Classification metric",
    "Binarna entropia krzyżowa": "Binary cross-entropy",
    "Pole pod krzywą ROC": "Area under the ROC curve",
    "Współczynnik regularyzacji alpha": "Alpha regularization coefficient",
    "Współczynnik uczenia": "Learning rate",
    "Jądro funkcji decyzyjnej": "Decision-function kernel",
    "Zoptymalizowany zbiór": "Optimized dataset",
    "Dokładność klasyfikacji": "Classification accuracy",
    "Liczba próbek": "Number of samples",
    "Liczba próbek w zbiorze testowym": "Number of samples in the test set",
    "Liczba cech": "Number of features",
    "Ważność cechy": "Feature importance",
    "Wartość cechy": "Feature value",
    "Wartość parametru C": "C parameter value",
    "Wartość miary": "Metric value",
    "Wartość": "Value",
    "Pełny zbiór": "Full dataset",
    "Zbiór walidacyjny": "Validation set",
    "Zbiór testowy": "Test set",
    "Zbiór uczący": "Training set",
    "Średnia podczas uczenia": "Training mean",
    "Zmiana względem poziomu bazowego": "Change relative to baseline",
    "Początek uczenia": "Training start",
    "Koniec uczenia": "Training end",
    "Macierz pomyłek": "Confusion matrix",
    "Klasa przewidziana": "Predicted class",
    "Klasa rzeczywista": "Actual class",
    "Błędnie sklasyfikowane": "Misclassified",
    "Brak phishingu": "Non-phishing",
    "Dokładność": "Accuracy",
    "Precyzja": "Precision",
    "Czułość": "Recall",
    "Specyficzność": "Specificity",
    "Miara F1": "F1 score",
    "Ważność": "Importance",
    "Cecha": "Feature",
    "Przedział": "Interval",
    "Średnia precyzja": "Average precision",
    "Średnia": "Mean",
    "Czas uczenia modelu [s]": "Model training time [s]",
    "Czas uczenia [s]": "Training time [s]",
    "Średnie wykorzystanie procesora [%]": "Mean CPU utilization [%]",
    "Średnie wykorzystanie pamięci RAM [GB]": "Mean RAM utilization [GB]",
    "Wartość miary błędu [%]": "Error metric value [%]",
    "Miara błędu": "Error metric",
    "Regresja logistyczna": "Logistic regression",
    "skala logarytmiczna": "logarithmic scale",
    "Czas [s]": "Time [s]",
    "Epoka": "Epoch",
    "Próbka": "Sample",
}


def translate_figure_text(text: str) -> str:
    """Translate known scientific plot terminology into English."""
    translated = text
    for polish, english in sorted(
        POLISH_TO_ENGLISH.items(), key=lambda item: len(item[0]), reverse=True
    ):
        translated = translated.replace(polish, english)
    return _translate_numeric_text(translated)


def _translate_numeric_text(text: str) -> str:
    """Convert a standalone Polish-formatted number into English notation."""
    match = POLISH_NUMBER.fullmatch(text.strip())
    if match is None:
        text = POLISH_GROUPED_NUMBER_FRAGMENT.sub(
            lambda item: item.group(1).replace(" ", ""), text
        )
        return POLISH_DECIMAL_FRAGMENT.sub(r"\1.\2", text)
    integer = match.group("integer").replace(" ", "")
    decimal = (match.group("decimal") or "").replace(",", ".")
    return f"{match.group('sign')}{integer}{decimal}{match.group('suffix')}"


def english_matplotlib_copy(figure: Any) -> Any:
    """Return an independent English-language Matplotlib figure."""
    translated = _copy_matplotlib_figure(figure)
    for axis in translated.axes:
        _translate_axis_formatters(axis)
    translated.canvas.draw()
    for text in translated.findobj(match=Text):
        text.set_text(translate_figure_text(text.get_text()))
    return translated


def _copy_matplotlib_figure(figure: Any) -> Any:
    """Copy a figure while isolating an upstream Python 3.14 warning."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Pickle, copy, and deepcopy support will be removed.*",
            category=DeprecationWarning,
        )
        return copy.deepcopy(figure)


def _translate_axis_formatters(axis: Any) -> None:
    """Translate category labels generated again during figure rendering."""
    _translate_axis_formatter(axis.xaxis)
    _translate_axis_formatter(axis.yaxis)


def _translate_axis_formatter(axis: Any) -> None:
    """Replace one supported formatter with an English-aware equivalent."""
    formatter = axis.get_major_formatter()
    translated = _english_formatter(formatter)
    if translated is not None:
        axis.set_major_formatter(translated)


def _english_formatter(formatter: Any) -> Any | None:
    """Return a formatter that preserves English labels after redraw."""
    if isinstance(formatter, StrCategoryFormatter):
        units = {
            translate_figure_text(label): position
            for label, position in formatter._units.items()
        }
        return StrCategoryFormatter(units)
    if isinstance(formatter, FixedFormatter):
        labels = [translate_figure_text(label) for label in formatter.seq]
        return FixedFormatter(labels, formatter.get_offset())
    if isinstance(formatter, FuncFormatter):
        return _translated_function_formatter(formatter)
    return None


def _translated_function_formatter(formatter: Any) -> FuncFormatter:
    """Wrap a dynamic formatter so every generated label is translated."""
    return FuncFormatter(
        lambda value, position: translate_figure_text(
            str(formatter(value, position))
        )
    )


def english_plotly_copy(figure: Any) -> Any:
    """Return an independent English-language Plotly figure."""
    payload = _translate_plotly_value(figure.to_plotly_json())
    translated = type(figure)(payload)
    translated.update_layout(separators=".,")
    return translated


def save_bilingual_plotly_figure(
    figure: Any,
    png_dir: str | Path,
    html_dir: str | Path,
    filename: str,
    *,
    width: int, height: int,
    scale: int = 2,
) -> dict[str, tuple[Path, Path]]:
    """Save and display Polish and English variants of a Plotly figure."""
    english_figure = english_plotly_copy(figure)
    polish_paths = _save_plotly_variant(
        figure, png_dir, html_dir, filename, "pl", width, height, scale
    )
    english_paths = _save_plotly_variant(
        english_figure, png_dir, html_dir, filename, "eng", width, height, scale
    )
    _remove_legacy_plotly_files(png_dir, html_dir, filename)
    _show_plotly_variants(figure, english_figure)
    return {"pl": polish_paths, "eng": english_paths}


def _show_plotly_variants(polish_figure: Any, english_figure: Any) -> None:
    """Display both Plotly language variants in the notebook."""
    polish_figure.show()
    english_figure.show()


def _save_plotly_variant(
    figure: Any, png_dir: str | Path, html_dir: str | Path,
    filename: str, language: str, width: int, height: int, scale: int,
) -> tuple[Path, Path]:
    """Save one Plotly language variant as PNG and interactive HTML."""
    png_path = Path(png_dir) / language / f"{filename}.png"
    html_path = Path(html_dir) / language / f"{filename}.html"
    png_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(html_path)
    figure.write_image(png_path, scale=scale, width=width, height=height)
    return png_path, html_path


def _remove_legacy_plotly_files(
    png_dir: str | Path, html_dir: str | Path, filename: str,
) -> None:
    """Remove obsolete flat Plotly files and ``en`` language variants."""
    remove_legacy_language_files(Path(png_dir) / f"{filename}.png")
    remove_legacy_language_files(Path(html_dir) / f"{filename}.html")


def _translate_plotly_value(value: Any) -> Any:
    """Recursively translate textual values in a Plotly figure payload."""
    if isinstance(value, str):
        return translate_figure_text(value)
    if isinstance(value, list):
        return [_translate_plotly_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_translate_plotly_value(item) for item in value)
    if isinstance(value, dict):
        return {key: _translate_plotly_value(item) for key, item in value.items()}
    return value


def language_output_path(output_path: str | Path, language: str) -> Path:
    """Return a path grouped inside the requested language directory."""
    path = Path(output_path)
    return path.parent / language / path.name


def path_language(output_path: str | Path) -> str | None:
    """Return the language encoded in a path, if it is explicit."""
    parts = Path(output_path).parts
    if "pl" in parts:
        return "pl"
    if "eng" in parts or "en" in parts:
        return "eng"
    return None


def remove_legacy_language_files(output_path: str | Path) -> None:
    """Remove obsolete flat and ``en``-directory variants of one artifact."""
    path = Path(output_path)
    path.unlink(missing_ok=True)
    for suffix in ("_pl", "_en", "_eng"):
        path.with_name(f"{path.stem}{suffix}{path.suffix}").unlink(missing_ok=True)
    obsolete_english = path.parent / "en" / path.name
    obsolete_english.unlink(missing_ok=True)
    _remove_empty_directory(obsolete_english.parent)


def _remove_empty_directory(directory: Path) -> None:
    """Remove an obsolete language directory only when it is empty."""
    try:
        directory.rmdir()
    except OSError:
        return
