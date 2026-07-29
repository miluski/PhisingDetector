"""Generate balanced real and synthetic datasets used by smoke tests."""

import os

import pandas as pd

def print_generation_header(dataset_name):
    """Print the heading for one dataset generation run."""
    print(f"\n{'='*100}")
    print(f"GENEROWANIE ZBIORU TESTOWEGO - {dataset_name.upper()}")
    print(f"{'='*100}\n")

def sample_real_examples(df):
    """Return a deterministic balanced sample of real records."""
    phishing_records = df[df['phishing'] == 1]
    legitimate_records = df[df['phishing'] == 0]

    print(f"Dostępne phishingowe rekordy: {len(phishing_records):,}")
    print(f"Dostępne bezpieczne rekordy: {len(legitimate_records):,}")

    sample_phishing = phishing_records.sample(n=10000, random_state=42)
    sample_legitimate = legitimate_records.sample(n=10000, random_state=42)

    real_examples = pd.concat([sample_phishing, sample_legitimate], ignore_index=True)
    real_examples = real_examples.sample(frac=1, random_state=42).reset_index(drop=True)

    print("\n✓ Wybrano 20,000 prawdziwych przykładów (10,000 phishing + 10,000 bezpiecznych)")
    return real_examples

def calculate_synthetic_value(col, val1, val2):
    """Average two feature values while preserving labels and missing values."""
    if col == 'phishing':
        return int(val1)

    if val1 == -1 or val2 == -1:
        return -1

    avg_val = (val1 + val2) / 2

    if val1 == int(val1) and val2 == int(val2):
        return int(round(avg_val))
    return round(avg_val, 6)

def create_synthetic_record(record1, record2, columns):
    """Create one synthetic row from two records."""
    synthetic_record = {}
    for col in columns:
        synthetic_record[col] = calculate_synthetic_value(col, record1[col], record2[col])
    return synthetic_record

def _sample_class(df, label):
    """Shuffle one class deterministically for pair generation."""
    rows = df[df["phishing"] == label]
    return rows.sample(frac=1, random_state=42).reset_index(drop=True)


def _synthetic_class_records(rows, columns, count=10000):
    """Generate averaged records for one label, cycling short classes."""
    if rows.empty:
        raise ValueError("Nie można generować rekordów dla pustej klasy")
    records = []
    for index in range(count):
        first = rows.iloc[(index * 2) % len(rows)]
        second = rows.iloc[(index * 2 + 1) % len(rows)]
        records.append(create_synthetic_record(first, second, columns))
    return records


def generate_synthetic_records(df):
    """Generate 10,000 synthetic rows for each binary class."""
    columns = df.columns
    phishing = _synthetic_class_records(_sample_class(df, 1), columns)
    legitimate = _synthetic_class_records(_sample_class(df, 0), columns)
    return pd.DataFrame(phishing + legitimate)

def ensure_integer_columns(df):
    """Cast integer-like dataset columns back to integer dtype."""
    for col in df.columns:
        if col == 'phishing' or df[col].dtype in ['int64', 'int32']:
            df[col] = df[col].astype(int)
    return df

def print_generation_summary(final_df, synthetic_df, output_csv):
    """Print counts and class distribution for a generated dataset."""
    print(f"\n✅ Zapisano zbiór testowy do: {output_csv}")
    print(f"   Całkowita liczba rekordów: {len(final_df):,}")
    print("   - Prawdziwe przykłady: 20,000 (10,000 phishing, 10,000 bezpiecznych)")
    print(f"   - Syntetyczne rekordy: {len(synthetic_df):,}")

    phishing_count = final_df['phishing'].sum()
    legitimate_count = len(final_df) - phishing_count
    print(f"   Rozkład etykiet: {phishing_count:,} phishing, {legitimate_count:,} bezpiecznych")

def generate_test_dataset(input_csv, output_csv, dataset_name):
    """Generate and persist one test dataset, returning it and its source."""
    print_generation_header(dataset_name)

    df = pd.read_csv(input_csv)
    real_examples = sample_real_examples(df)
    synthetic_df = generate_synthetic_records(df)

    print(f"✓ Wygenerowano {len(synthetic_df):,} syntetycznych rekordów (średnie z par)")

    final_df = pd.concat([real_examples, synthetic_df], ignore_index=True)
    final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)
    final_df = ensure_integer_columns(final_df)
    final_df.to_csv(output_csv, index=False)

    print_generation_summary(final_df, synthetic_df, output_csv)
    return final_df, df

def print_statistics_comparison(original_df, test_df):
    """Print row and class-count comparison between two datasets."""
    print(f"{'Statystyka':<40} {'Oryginalny zbiór':>25} {'Zbiór testowy':>25}")
    print(f"{'─'*100}")
    print(f"{'Liczba rekordów':<40} {len(original_df):>25,} {len(test_df):>25,}")

    orig_phishing = original_df['phishing'].sum()
    test_phishing = test_df['phishing'].sum()
    print(f"{'Rekordy phishingowe':<40} {orig_phishing:>25,} {test_phishing:>25,}")

    orig_legitimate = len(original_df) - orig_phishing
    test_legitimate = len(test_df) - test_phishing
    print(f"{'Rekordy bezpieczne':<40} {orig_legitimate:>25,} {test_legitimate:>25,}")

    orig_phishing_pct = (orig_phishing / len(original_df)) * 100
    test_phishing_pct = (test_phishing / len(test_df)) * 100
    print(f"{'% Phishing':<40} {orig_phishing_pct:>24.2f}% {test_phishing_pct:>24.2f}%")

def get_sample_features(feature_columns):
    """Select representative features available in a dataset."""
    sample_features = [
        'url_len', 'domain_len', 'qty_dot_url',
        'qty_hyphen_url', 'qty_slash_url', 'time_response'
    ]
    return [f for f in sample_features if f in feature_columns]

def compare_feature_ranges(original_df, test_df, available_features):
    """Print source and generated ranges for representative features."""
    print(f"\n{'Cecha':<30} {'Oryg: Min':>12} {'Oryg: Max':>12} {'Test: Min':>12} {'Test: Max':>12} {'Zgodność':>10}")
    print(f"{'─'*100}")

    for feature in available_features:
        orig_vals = original_df[feature][original_df[feature] != -1]
        test_vals = test_df[feature][test_df[feature] != -1]

        if len(orig_vals) == 0 or len(test_vals) == 0:
            continue

        orig_min, orig_max = orig_vals.min(), orig_vals.max()
        test_min, test_max = test_vals.min(), test_vals.max()
        in_range = "✓" if (test_min >= orig_min and test_max <= orig_max) else "✗"

        print(f"{feature:<30} {orig_min:>12.2f} {orig_max:>12.2f} {test_min:>12.2f} {test_max:>12.2f} {in_range:>10}")

def compare_datasets(test_df, original_df, dataset_name):
    """Print statistics and range checks for one generated dataset."""
    print(f"\n{'='*100}")
    print(f"PORÓWNANIE STATYSTYK - {dataset_name.upper()}")
    print(f"{'='*100}\n")

    feature_columns = [col for col in test_df.columns if col != 'phishing']
    print_statistics_comparison(original_df, test_df)

    print(f"\n{'─'*100}")
    print("PORÓWNANIE ZAKRESÓW WYBRANYCH CECH")
    print(f"{'─'*100}")

    available_features = get_sample_features(feature_columns)
    compare_feature_ranges(original_df, test_df, available_features)
    print(f"{'─'*100}\n")

def _dataset_paths(data_dir, name):
    """Return source and destination paths for a named dataset."""
    source = os.path.join(data_dir, f"dataset_{name}_balanced.csv")
    target = os.path.join(data_dir, "test", f"test_dataset_{name}.csv")
    return source, target


def _generate_named_dataset(data_dir, name, title):
    """Generate one named dataset when its source exists."""
    source, target = _dataset_paths(data_dir, name)
    if not os.path.exists(source):
        print(f"Błąd: Nie znaleziono pliku {source}")
        return None
    return generate_test_dataset(source, target, title)


def main():
    """Generate both configured datasets and print comparisons."""
    data_dir = "../data"
    test_dir = os.path.join(data_dir, "test")
    os.makedirs(test_dir, exist_ok=True)
    full = _generate_named_dataset(data_dir, "full", "Pełny zbiór danych (111 cech)")
    optimized = _generate_named_dataset(data_dir, "optimized", "Zoptymalizowany zbiór danych (73 cechy)")
    if full:
        compare_datasets(full[0], full[1], "Pełny zbiór danych (111 cech)")
    if optimized:
        compare_datasets(optimized[0], optimized[1], "Zoptymalizowany zbiór danych (73 cechy)")

    print(f"\n{'='*100}")
    print("✅ GENEROWANIE ZAKOŃCZONE POMYŚLNIE!")
    print(f"{'='*100}\n")

if __name__ == "__main__":
    main()
