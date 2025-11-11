
# Podsumowanie Kompleksowej Analizy Statystycznej

## 1. Przegląd Danych
- **Liczba próbek:** 87,209
- **Liczba cech (oryginalnie):** 111
- **Typ cech:** URL + DNS/WHOIS (wszystkie reprodukowalne)
- **Rozkład klas:**
    - Phishing: 30,497 (34.97%)
    - Bezpieczne: 56,712 (65.03%)

## 2. Utworzony Zbiór Danych

### Optimized (Selekcja Cech)
- **Plik:** `dataset_optimized.csv`
- **Liczba cech:** 74 (z 111)
- **Redukcja:** 33.3%
- **Typ:** Wszystkie atrybuty (URL + DNS/WHOIS)
- **Zalety:**
  - Zachowuje oryginalne cechy - łatwo zrozumieć znaczenie
  - Dobra do analizy biznesowej i wyjaśniania modelu
  - Zawiera cechy z URLFeatureAnalyzer (szybkie) + AdvancedURLAnalyzer (wolne)
  - Najwyższa dokładność modeli
- **Zastosowanie:** Modele wymagające maksymalnej dokładności

## 3. Kluczowe Cechy (Feature Selection)
**Top 5 najważniejszych zachowanych cech:**
1. `directory_length` (ważność: 0.1502)
2. `qty_dollar_directory` (ważność: 0.0563)
3. `qty_dot_file` (ważność: 0.0530)
4. `time_domain_activation` (ważność: 0.0504)
5. `qty_slash_directory` (ważność: 0.0366)

## 4. Rekomendacje

### Wybór Zbioru Danych:
- **Dataset Full** → Maksymalna informacja (111 cech)
- **Dataset Optimized** → Najlepszy stosunek jakości do wydajności (74 cech)

### Zalecane Modele:
- Random Forest, XGBoost, Logistic Regression (dobre do interpretacji)
- SVM, Neural Networks (dla lepszej dokładności)

## 5. Informacja o Reprodukowalności Cech
Wszystkie cechy są reprodukowalne przy użyciu:
- **URLFeatureAnalyzer** - szybka ekstrakcja z URL (89 cech URL-only)
- **AdvancedURLAnalyzer** - wolna ekstrakcja z DNS/WHOIS/HTTP (11 cech):
  - time_response, domain_spf, asn_ip
  - time_domain_activation, time_domain_expiration
  - qty_ip_resolved, qty_nameservers, qty_mx_servers
  - ttl_hostname, url_google_index, domain_google_index

## 6. Następne Kroki
1. Przetestować oba zbiory danych z różnymi modelami
2. Porównać dokładność i czas trenowania
3. Wybrać najlepsze podejście dla aplikacji

## 7. Zapisane Artefakty
- **Dane oryginalne:** `../data/dataset_full.csv` (111 cech)
- **Dane zoptymalizowane:** `../data/dataset_optimized.csv` (74 cech)
- **Wizualizacje PNG:** `../static/statistical_analysis/png/`
- **Wizualizacje HTML:** `../static/statistical_analysis/html/`
- **Raporty CSV:** `../static/statistical_analysis/csv/`
- **Dokumentacja:** `../static/statistical_analysis/docs/`
