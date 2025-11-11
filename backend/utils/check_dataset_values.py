import pandas as pd
import sys
sys.path.append('..')
from utils.feature_names import FEATURE_NAMES, FEATURE_CATEGORIES

df = pd.read_csv('../data/dataset_full.csv')

print("=" * 80)
print("SPRAWDZANIE WARTOŚCI W ZBIORZE DANYCH")
print("=" * 80)

print(f"\nRozmiar danych: {df.shape}")
print(f"Liczba cech: {df.shape[1] - 1}")

print("\n" + "=" * 80)
print("ANALIZA WARTOŚCI UJEMNYCH")
print("=" * 80)

negative_features = {}
for col in df.columns[:-1]:
    negative_count = (df[col] < 0).sum()
    if negative_count > 0:
        negative_features[col] = {
            'count': negative_count,
            'percentage': (negative_count / len(df)) * 100,
            'min': df[col].min(),
            'max': df[col].max(),
            'mean': df[col].mean()
        }

if negative_features:
    print(f"\n⚠️  Znaleziono {len(negative_features)} cech z wartościami ujemnymi:")
    for feature, stats in negative_features.items():
        readable_name = FEATURE_NAMES.get(feature, feature)
        print(f"\n  {feature} ({readable_name}):")
        print(f"    • Liczba wartości ujemnych: {stats['count']} ({stats['percentage']:.2f}%)")
        print(f"    • Min: {stats['min']:.2f}, Max: {stats['max']:.2f}, Mean: {stats['mean']:.2f}")
else:
    print("\n✅ Nie znaleziono wartości ujemnych")

print("\n" + "=" * 80)
print("ANALIZA ZAKRESÓW WARTOŚCI WEDŁUG KATEGORII")
print("=" * 80)

for category, features in FEATURE_CATEGORIES.items():
    print(f"\n{category}:")
    existing_features = [f for f in features if f in df.columns]

    for feature in existing_features[:5]:
        readable_name = FEATURE_NAMES.get(feature, feature)
        min_val = df[feature].min()
        max_val = df[feature].max()
        mean_val = df[feature].mean()
        unique_count = df[feature].nunique()

        print(f"  • {feature}: min={min_val:.2f}, max={max_val:.2f}, "
              f"mean={mean_val:.2f}, unique={unique_count}")

print("\n" + "=" * 80)
print("SPRAWDZANIE OCZEKIWANYCH ZAKRESÓW")
print("=" * 80)

issues = []

binary_features = ['email_in_url', 'domain_in_ip', 'server_client_domain', 'domain_spf',
                   'tls_ssl_certificate', 'tld_present_params', 'url_google_index',
                   'domain_google_index', 'url_shortened']

for feature in binary_features:
    if feature in df.columns:
        unique_vals = sorted(df[feature].unique())
        if not set(unique_vals).issubset({0, 1}):
            issues.append(f"{feature}: Powinno być 0/1, znaleziono: {unique_vals}")

count_features = [col for col in df.columns if col.startswith('qty_') and col in df.columns]
for feature in count_features:
    min_val = df[feature].min()
    if min_val < 0:
        issues.append(f"{feature}: Liczba <0 (min={min_val})")

length_features = [col for col in df.columns if 'length' in col and col in df.columns]
for feature in length_features:
    min_val = df[feature].min()
    if min_val < 0:
        issues.append(f"{feature}: Długość <0 (min={min_val})")

time_features = ['time_response', 'time_domain_activation', 'time_domain_expiration']
for feature in time_features:
    if feature in df.columns:
        min_val = df[feature].min()
        max_val = df[feature].max()
        print(f"\n{feature}:")
        print(f"  Min: {min_val}, Max: {max_val}")
        if feature in ['time_domain_activation', 'time_domain_expiration'] and min_val < 0:
            issues.append(f"{feature}: Wartość ujemna może oznaczać przeszłość/przyszłość (min={min_val})")

if issues:
    print(f"\n⚠️  Znaleziono {len(issues)} potencjalnych problemów:")
    for issue in issues:
        print(f"  • {issue}")
else:
    print("\n✅ Wszystkie wartości w oczekiwanych zakresach")
