# Instrukcja uruchomienia

## Wymagania
- Python w wersji 3.8+
- Node.js w wersji 18+
- Angular CLI (Command Line Interface)

## Uruchomienie backendu (Flask API)

1. Na początku nalezy przejść do folderu backend:
```bash
cd backend
```

2. Później należy zainstalować wszystkie niezbędne pakiety:
```bash
pip install -r requirements.txt
```

3. Ostatnim krokiem jest uruchomienie serwera:
```bash
python app.py
```

API będzie dostępne pod adresem: http://localhost:5001

## Uruchomienie frontendu (Angular)

1. Należy przejść do folderu frontend:
```bash
cd frontend
```

2. Należy zainstalować pakiety za pomocą menadżera pakietów, np. npm:
```bash
npm install
```

3. Należy uruchomić aplikację komendą:
```bash
ng serve
```

Strona internetowa będzie dostępna pod adresem: http://localhost:4200

## Punkty końcowe API

### Predykcja pojedynczych modeli:
- `POST /api/predict/logistic-regression` - Model wyuczony za pomocą algorytmu regresji logistycznej
- `POST /api/predict/random-forest` - Model wyuczony za pomocą algorytmu Random Forest
- `POST /api/predict/xgboost` - Model wyuczony za pomocą algorytmu XGBoost
- `POST /api/predict/svm` - Model wyuczony za pomocą algorytmu maszyny wektorów nośnych
- `POST /api/predict/mlp` - Model wyuczony za pomocą algorytmów stosowanych w uczeniu perceptronu wielowarstwowego

### Predykcja zespołowa:
- `POST /api/predict/all` - Wszystkie modele wyuczone za pomocą wyżej wymienionych algorytmów

### Wykresy i wizualizacje:
- `GET /api/plots/{filename}` - Pobranie konkretnego obrazka wykresu statystyk danego modelu

## Przykład ciała żądania do endpointu predykcji:

```json
{
  "sender": "suspicious@example.com",
  "content": "Urgent! Click this link to claim your prize!",
  "receiver": "victim@example.com"
}
```

### Zbiór danych

https://gregavrbancic.github.io/Phishing-Dataset/

### Usunięte kolumny

time_response - czas odpowiedzi DNS
domain_spf - sprawdzenie rekordu SPF
asn_ip - sprawdzenie ASN dla adresu IP
time_domain_activation - data utworzenia domeny (WHOIS)
time_domain_expiration - data wygaśnięcia domeny (WHOIS)
qty_ip_resolved - liczba rozwiązań DNS (IP)
qty_nameservers - liczba serwerów nazw (DNS)
qty_mx_servers - liczba serwerów pocztowych (MX)
ttl_hostname - wartości TTL DNS
url_google_index - status indeksowania URL w Google
domain_google_index - status indeksowania domeny w Google