"""Print feature ranges for full and optimized datasets."""

import os

import pandas as pd
from feature_names import get_readable_name, FEATURE_CATEGORIES

def print_header(dataset_name):
    """Print the heading for a dataset range report."""
    print(f"\n{'='*100}")
    print(f"ANALIZA ZAKRESÓW CECH - {dataset_name.upper()}")
    print(f"{'='*100}\n")

def calculate_interval(values):
    """Return valid minimum, maximum and a human-readable interval."""
    valid_values = values[values != -1]

    if len(valid_values) == 0:
        return -1, -1, "N/A (wszystkie -1)"

    min_val = valid_values.min()
    max_val = valid_values.max()
    has_unknown = (values == -1).any()

    if has_unknown:
        interval = f"[{min_val}] lub -1" if min_val == max_val else f"[{min_val}, {max_val}] lub -1"
    else:
        interval = f"[{min_val}]" if min_val == max_val else f"[{min_val}, {max_val}]"

    return min_val, max_val, interval

def print_category_header(category):
    """Print a category table heading."""
    print(f"\n{'─'*100}")
    print(f"  {category}")
    print(f"{'─'*100}")
    print(f"{'Cecha':<60} {'Min':>10} {'Max':>10} {'Przedział':>18}")
    print(f"{'─'*100}")

def analyze_category_features(df, category_features, feature_columns):
    """Print ranges for features from one category."""
    present_features = [f for f in category_features if f in feature_columns]
    if not present_features:
        return

    for feature in present_features:
        if feature not in df.columns:
            continue

        min_val, max_val, interval = calculate_interval(df[feature])
        readable_name = get_readable_name(feature)
        print(f"{readable_name:<60} {min_val:>10.2f} {max_val:>10.2f} {interval:>18}")

def analyze_feature_ranges(csv_path, dataset_name):
    """Load a dataset and print ranges grouped by feature category."""
    print_header(dataset_name)

    df = pd.read_csv(csv_path)
    feature_columns = [col for col in df.columns if col != 'phishing']

    print(f"Liczba cech: {len(feature_columns)}")
    print(f"Liczba próbek: {len(df):,}\n")

    for category, category_features in FEATURE_CATEGORIES.items():
        present_features = [f for f in category_features if f in feature_columns]
        if present_features:
            print_category_header(category)
            analyze_category_features(df, category_features, feature_columns)

    print(f"\n{'='*100}\n")

def get_feature_range(df, feature):
    """Return the minimum and maximum excluding the missing sentinel."""
    values = df[feature]
    valid = values[values != -1]
    min_val = valid.min() if len(valid) > 0 else -1
    max_val = valid.max() if len(valid) > 0 else -1
    return min_val, max_val

def print_comparison_header():
    """Print the heading for a two-dataset comparison."""
    print(f"\n{'─'*120}")
    print(f"{'Cecha':<50} {'Pełny: Min':>12} {'Pełny: Max':>12} {'Opt: Min':>12} {'Opt: Max':>12} {'Identyczne':>12}")
    print(f"{'─'*120}")

def print_removed_features(only_in_full):
    """Print features present only in the full dataset."""
    if only_in_full:
        print("Cechy usunięte w zoptymalizowanym zbiorze:")
        for feature in sorted(only_in_full):
            print(f"  • {get_readable_name(feature)}")
        print()

def compare_feature_ranges(df_full, df_optimized, common_features):
    """Print ranges of features shared by both datasets."""
    print_comparison_header()

    for feature in sorted(common_features):
        min_full, max_full = get_feature_range(df_full, feature)
        min_opt, max_opt = get_feature_range(df_optimized, feature)

        identical = "✓" if (min_full == min_opt and max_full == max_opt) else "✗"
        readable_name = get_readable_name(feature)
        if len(readable_name) > 48:
            readable_name = readable_name[:45] + "..."

        print(
            f"{readable_name:<50} {min_full:>12.2f} {max_full:>12.2f} "
            f"{min_opt:>12.2f} {max_opt:>12.2f} {identical:>12}"
        )

def _load_comparison_frames(full_csv_path, optimized_csv_path):
    """Load full and optimized dataframes for comparison."""
    return pd.read_csv(full_csv_path), pd.read_csv(optimized_csv_path)


def _compare_loaded_datasets(df_full, df_optimized):
    """Print feature counts, removed features and common ranges."""
    features_full = set(df_full.columns) - {"phishing"}
    features_optimized = set(df_optimized.columns) - {"phishing"}
    common_features = features_full & features_optimized
    only_in_full = features_full - features_optimized
    print(f"Cechy wspólne: {len(common_features)}")
    print(f"Cechy tylko w pełnym zbiorze: {len(only_in_full)}\n")
    print_removed_features(only_in_full)
    compare_feature_ranges(df_full, df_optimized, common_features)


def compare_datasets(full_csv_path, optimized_csv_path):
    """Print the ranges that differ between full and optimized datasets."""
    print(f"\n{'='*100}")
    print("PORÓWNANIE ZAKRESÓW CECH MIĘDZY ZBIORAMI")
    print(f"{'='*100}\n")

    df_full, df_optimized = _load_comparison_frames(full_csv_path, optimized_csv_path)
    _compare_loaded_datasets(df_full, df_optimized)

    print(f"{'─'*120}\n")


def main():
    """Run range reports for configured full and optimized CSV files."""
    data_dir = "../data"
    full_csv = os.path.join(data_dir, "dataset_full.csv")
    optimized_csv = os.path.join(data_dir, "dataset_optimized.csv")
    if not _require_inputs(full_csv, optimized_csv):
        return
    analyze_feature_ranges(full_csv, "Pełny zbiór danych (111 cech)")
    analyze_feature_ranges(optimized_csv, "Zoptymalizowany zbiór danych (73 cechy)")
    compare_datasets(full_csv, optimized_csv)
    print("\n✅ Analiza zakończona!")


def _require_inputs(full_csv, optimized_csv):
    """Validate that both comparison inputs exist."""
    for path in (full_csv, optimized_csv):
        if not os.path.exists(path):
            print(f"Błąd: Nie znaleziono pliku {path}")
            return False
    return True


if __name__ == "__main__":
    main()
