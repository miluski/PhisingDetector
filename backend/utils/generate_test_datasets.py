import pandas as pd
import numpy as np
import os

def print_generation_header(dataset_name):
    print(f"\n{'='*100}")
    print(f"GENEROWANIE ZBIORU TESTOWEGO - {dataset_name.upper()}")
    print(f"{'='*100}\n")

def sample_real_examples(df):
    phishing_records = df[df['phishing'] == 1]
    legitimate_records = df[df['phishing'] == 0]

    print(f"Dostępne phishingowe rekordy: {len(phishing_records):,}")
    print(f"Dostępne bezpieczne rekordy: {len(legitimate_records):,}")

    sample_phishing = phishing_records.sample(n=10000, random_state=42)
    sample_legitimate = legitimate_records.sample(n=10000, random_state=42)

    real_examples = pd.concat([sample_phishing, sample_legitimate], ignore_index=True)
    real_examples = real_examples.sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"\n✓ Wybrano 20,000 prawdziwych przykładów (10,000 phishing + 10,000 bezpiecznych)")
    return real_examples

def calculate_synthetic_value(col, val1, val2):
    if col == 'phishing':
        return int(val1)

    if val1 == -1 or val2 == -1:
        return -1

    avg_val = (val1 + val2) / 2

    if val1 == int(val1) and val2 == int(val2):
        return int(round(avg_val))
    return round(avg_val, 6)

def create_synthetic_record(record1, record2, columns):
    synthetic_record = {}
    for col in columns:
        synthetic_record[col] = calculate_synthetic_value(col, record1[col], record2[col])
    return synthetic_record

def generate_synthetic_records(df):
    synthetic_records = []
    phishing_records = df[df['phishing'] == 1]
    legitimate_records = df[df['phishing'] == 0]

    phishing_records = phishing_records.sample(frac=1, random_state=42).reset_index(drop=True)
    legitimate_records = legitimate_records.sample(frac=1, random_state=42).reset_index(drop=True)

    for i in range(10000):
        idx1 = i * 2
        idx2 = i * 2 + 1

        if idx2 >= len(phishing_records):
            idx1 = idx1 % len(phishing_records)
            idx2 = idx2 % len(phishing_records)

        record1 = phishing_records.iloc[idx1]
        record2 = phishing_records.iloc[idx2]
        synthetic_record = create_synthetic_record(record1, record2, df.columns)
        synthetic_records.append(synthetic_record)

    for i in range(10000):
        idx1 = i * 2
        idx2 = i * 2 + 1

        if idx2 >= len(legitimate_records):
            idx1 = idx1 % len(legitimate_records)
            idx2 = idx2 % len(legitimate_records)

        record1 = legitimate_records.iloc[idx1]
        record2 = legitimate_records.iloc[idx2]
        synthetic_record = create_synthetic_record(record1, record2, df.columns)
        synthetic_records.append(synthetic_record)

    return pd.DataFrame(synthetic_records)

def ensure_integer_columns(df):
    for col in df.columns:
        if col == 'phishing' or df[col].dtype in ['int64', 'int32']:
            df[col] = df[col].astype(int)
    return df

def print_generation_summary(final_df, synthetic_df, output_csv):
    print(f"\n✅ Zapisano zbiór testowy do: {output_csv}")
    print(f"   Całkowita liczba rekordów: {len(final_df):,}")
    print(f"   - Prawdziwe przykłady: 20,000 (10,000 phishing, 10,000 bezpiecznych)")
    print(f"   - Syntetyczne rekordy: {len(synthetic_df):,}")

    phishing_count = final_df['phishing'].sum()
    legitimate_count = len(final_df) - phishing_count
    print(f"   Rozkład etykiet: {phishing_count:,} phishing, {legitimate_count:,} bezpiecznych")

def generate_test_dataset(input_csv, output_csv, dataset_name):
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
    sample_features = [
        'url_len', 'domain_len', 'qty_dot_url',
        'qty_hyphen_url', 'qty_slash_url', 'asn_ip', 'time_response'
    ]
    return [f for f in sample_features if f in feature_columns]

def compare_feature_ranges(original_df, test_df, available_features):
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
    print(f"\n{'='*100}")
    print(f"PORÓWNANIE STATYSTYK - {dataset_name.upper()}")
    print(f"{'='*100}\n")

    feature_columns = [col for col in test_df.columns if col != 'phishing']
    print_statistics_comparison(original_df, test_df)

    print(f"\n{'─'*100}")
    print(f"PORÓWNANIE ZAKRESÓW WYBRANYCH CECH")
    print(f"{'─'*100}")

    available_features = get_sample_features(feature_columns)
    compare_feature_ranges(original_df, test_df, available_features)
    print(f"{'─'*100}\n")

def main():
    data_dir = "../data"
    test_dir = os.path.join(data_dir, "test")

    full_csv = os.path.join(data_dir, "dataset_full.csv")
    optimized_csv = os.path.join(data_dir, "dataset_optimized.csv")

    os.makedirs(test_dir, exist_ok=True)

    test_full_csv = os.path.join(test_dir, "test_dataset_full.csv")
    test_optimized_csv = os.path.join(test_dir, "test_dataset_optimized.csv")

    if not os.path.exists(full_csv):
        print(f"Błąd: Nie znaleziono pliku {full_csv}")
        return

    if not os.path.exists(optimized_csv):
        print(f"Błąd: Nie znaleziono pliku {optimized_csv}")
        return

    test_full_df, original_full_df = generate_test_dataset(
        full_csv, test_full_csv, "Pełny zbiór danych (111 cech)"
    )

    test_optimized_df, original_optimized_df = generate_test_dataset(
        optimized_csv, test_optimized_csv, "Zoptymalizowany zbiór danych (73 cechy)"
    )

    compare_datasets(test_full_df, original_full_df, "Pełny zbiór danych (111 cech)")

    compare_datasets(test_optimized_df, original_optimized_df, "Zoptymalizowany zbiór danych (73 cechy)")

    print(f"\n{'='*100}")
    print("✅ GENEROWANIE ZAKOŃCZONE POMYŚLNIE!")
    print(f"{'='*100}\n")

if __name__ == "__main__":
    main()
