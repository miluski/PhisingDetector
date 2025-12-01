# PhishingDetector - Secure ML-based Phishing Detection System

## Table of Contents

- [Project Overview](#project-overview)
- [System Architecture](#system-architecture)
- [Security Features](#security-features)
- [Frontend GUI](#frontend-gui)
- [Dataset](#dataset)
- [Feature Extraction](#feature-extraction)
- [Installation & Setup](#installation--setup)
- [API Endpoints](#api-endpoints)
- [Project Structure](#project-structure)
- [Machine Learning Models](#machine-learning-models)
- [Development](#development)

---

## Project Overview

PhishingDetector is an advanced secure platform for detecting phishing in emails and SMS using machine learning algorithms with Docker-based isolation. The project consists of:

- **Backend (Flask API)**: RESTful API with containerized phishing detection
- **Frontend (Angular)**: Comprehensive GUI for safe email/SMS analysis
- **ML Models**: 5 trained models (Random Forest, XGBoost, MLP, SVM, Logistic Regression)
- **Docker Sandbox**: Isolated execution environment for security

### Key Features

- ✅ **Secure Containerized Analysis**: Docker isolation with no network access
- ✅ **Email & SMS Support**: Extract URLs from email (sender/receiver/content/attachments) and SMS
- ✅ **Multi-Source URL Extraction**: PDF, DOCX, TXT attachments supported
- ✅ **ML Ensemble Predictions**: Use single or multiple models with voting
- ✅ **Real-time Visualizations**: 4 chart types (probabilities, voting, feature importance, ensemble)
- ✅ **Extract 111 features** (full mode) or **74 features** (optimized mode)
- ✅ **HTTPS** with automated certificate generation
- ✅ **SOLID principles** throughout codebase
- ✅ **All methods ≤20 lines** - clean, maintainable code

---

## Security Features

### Docker Container Isolation

Every phishing analysis runs in an **isolated Docker container** with:

- **No Network Access**: `network_mode='none'` prevents external communication
- **Resource Limits**: 2GB RAM, 2 CPU cores maximum
- **Read-Only Volumes**: Models and data mounted as read-only
- **Automatic Cleanup**: Containers removed after analysis
- **Sandboxed Execution**: Feature extraction and ML prediction isolated from host

### Architecture Flow

```
User → Angular Frontend → Flask API → Docker Container → ML Analysis → Results
                                      ↓
                              [Isolated Sandbox]
                              - URL Feature Extraction
                              - ML Model Predictions
                              - No Network Access
                              - Resource Limited
```

---

## Frontend GUI

### Comprehensive Web Interface

The Angular frontend provides a **secure, user-friendly interface** for phishing detection:

#### Features:

1. **Dual Analysis Mode**

   - 📧 **Email Analysis**: Full email inspection with sender, receiver, content, attachments
   - 💬 **SMS Analysis**: Quick SMS message scanning

2. **Smart Input Handling**

   - Email address fields for sender/receiver
   - Multi-line content editor
   - File upload for attachments (PDF, DOCX, TXT)
   - Drag-and-drop support

3. **ML Model Selection**

   - Select individual models or use "Select All"
   - Visual checkboxes with model descriptions
   - Real-time accuracy indicators:
     - Random Forest (96%)
     - XGBoost (97%)
     - MLP (95%)
     - SVM (94%)
     - Logistic Regression (92%)

4. **Analysis Mode Toggle**

   - **Optimized** (74 features, faster)
   - **Full** (111 features, comprehensive)

5. **Rich Results Display**

   - 🚨 **Threat Indicator**: Clear visual alert (SAFE vs PHISHING)
   - 📊 **Summary Card**: URLs found, models used, analysis mode
   - 🎯 **Per-URL Analysis**: Individual results for each detected URL
   - 📈 **Ensemble Voting**: Consensus from multiple models
   - 🔍 **Model Details**: Confidence scores, probabilities, predictions

6. **Interactive Visualizations**

   - **Probability Chart**: Bar chart comparing Safe vs Phishing across models
   - **Voting Chart**: Pie chart showing Random Forest tree voting
   - **Feature Importance**: Top 10 most important features
   - **Ensemble Voting**: Model consensus visualization
   - All charts displayed as high-quality PNG images

7. **Safety Features**
   - No direct URL submission (extracts from email/SMS)
   - Client-side file validation
   - Error handling with clear messages
   - Automatic cleanup after analysis

### UI Components

**Location**: `frontend/src/app/components/phishing-detector/`

- `phishing-detector.component.ts` - Main logic (150 lines)
- `phishing-detector.component.html` - Template (200+ lines)
- `phishing-detector.component.css` - Styled UI (600+ lines)

**Services**: `frontend/src/app/services/phishing-detection.service.ts`

**Models**: `frontend/src/app/models/phishing-analysis.model.ts`

---

## System Architecture

### Design Principles

The system follows **SOLID** principles:

1. **Single Responsibility Principle**: Each class has one responsibility

   - `CharacterCounter` - only character counting
   - `DNSResolver` - only DNS operations
   - `SSLValidator` - only certificate validation

2. **Open/Closed Principle**: Extensibility without modification

   - New extractors can be added without changing existing code
   - Configuration driven by config files

3. **Liskov Substitution**: Component interchangeability

   - All extractors have consistent interfaces
   - Analyzers can be swapped without affecting the system

4. **Interface Segregation**: Small, focused interfaces

   - No class depends on methods it doesn't use

5. **Dependency Inversion**: Dependencies on abstractions
   - `URLFeatureService` orchestrates without knowledge of implementation details

### Package Structure

```
backend/
├── extractors/              # Feature extraction (SRP)
│   ├── lexical/            # Lexical analysis
│   │   ├── character_counter.py
│   │   ├── vowel_counter.py
│   │   ├── tld_counter.py
│   │   ├── email_detector.py
│   │   ├── ip_address_detector.py
│   │   ├── keyword_detector.py
│   │   ├── url_shortener_detector.py
│   │   ├── url_part_extractor.py
│   │   ├── path_part_extractor.py
│   │   └── params_extractor.py
│   ├── network/            # Network analysis
│   │   ├── response_timer.py
│   │   ├── dns_resolver.py
│   │   ├── whois_extractor.py
│   │   └── ssl_validator.py
│   └── content/            # HTML content analysis
│       ├── external_resource_analyzer.py
│       ├── anchor_analyzer.py
│       ├── form_handler.py
│       ├── popup_detector.py
│       └── html_content_extractor.py
├── validators/             # Input validation (OCP)
│   ├── url_validator.py
│   └── mode_validator.py
├── services/               # Business logic
│   └── url_feature_service.py
├── config/                 # Configuration
│   ├── feature_config.py
│   ├── certificate_path_resolver.py
│   └── server_runner.py
├── app/                    # Flask application
│   ├── __init__.py
│   └── routes.py
├── models/                 # Trained ML models
│   ├── logistic_regression/
│   ├── random_forest/
│   ├── xgboost/
│   ├── svm/
│   └── mlp/
├── data/                   # Datasets
└── certs/                  # HTTPS certificates
```

### Code Properties

- ✅ **All methods ≤ 20 lines**
- ✅ **Each class in separate file**
- ✅ **No comments** (code is self-documenting)
- ✅ **Clean naming** and single level of abstraction

---

## Dataset

### Source

Dataset from scientific research:

- **Source**: [Phishing Website Features Dataset](https://gregavrbancic.github.io/Phishing-Dataset/)
- **Publication**: Mohammad, R. M., Thabtah, F., & McCluskey, L. (2014). "An assessment of features related to phishing websites using an automated technique"

### Characteristics

**Size:**

- `dataset_full.csv`: 88,647 rows × 111 features
- `dataset_optimized.csv`: 87,211 rows × 74 features

**Class Distribution:**

- Phishing: 30,647 instances (34.6%)
- Legitimate: 58,000 instances (65.4%)

**Feature Types:**

1. **Lexical (48 features)**: Character counting in URL parts
2. **Network (13 features)**: DNS, WHOIS, SSL/TLS
3. **Content (5 features)**: HTML analysis (full mode only)
4. **Special (8 features)**: Email patterns, IP, URL shorteners

### Dataset Modes

#### Optimized Mode (74 features)

Optimized feature set selected based on importance analysis:

**URL (10 features):**

- Character counts: `.`, `-`, `_`, `/`, `?`, `=`, `@`, `&`
- Additional: `qty_tld_url`, `length_url`

**Domain (4 features):**

- Character counts: `.`, `-`
- Additional: `qty_vowels_domain`, `domain_length`

**Directory (18 features):**

- All 17 special characters + `directory_length`

**File (17 features):**

- All 17 special characters (without `$`) + `file_length`

**Params (12 features):**

- Selected: `.`, `-`, `_`, `/`, `?`, `@`, `&`, ` `, `*`, `%`
- Additional: `params_length`, `qty_params`

**Network (13 features):**

- `time_response`, `domain_spf`, `asn_ip`
- `time_domain_activation`, `time_domain_expiration`
- `qty_ip_resolved`, `qty_nameservers`, `qty_mx_servers`
- `ttl_hostname`, `tls_ssl_certificate`, `qty_redirects`

**Special (2 features):**

- `email_in_url`, `url_shortened`

#### Full Mode (111 features)

Complete set contains all optimized features plus:

**Additional URL/Domain characters:**

- `!`, ` `, `~`, `,`, `+`, `*`, `#`, `$`, `%`

**Additional Domain features:**

- `domain_in_ip`, `server_client_domain`

**Additional Params features:**

- `tld_present_params` + all special characters

**Cechy HTML Content (5):**

- `request_url`: % external resources
- `url_of_anchor`: % suspicious links
- `sfh_handler`: Form analysis
- `popup_window`: Popup detection
- `web_traffic`: Traffic ranking

**Google features:**

- `url_google_index`, `domain_google_index`

---

## Feature Extraction

### Extraction Process

Feature extraction occurs in 4 phases:

```
URL Input → Parsing → Feature Extraction → Feature Vector
              ↓            ↓                    ↓
         urlparse()   Lexical/Network/    Dict[str, int]
         tldextract   Content Analysis    (74 or 111 features)
```

### Feature Categories

#### 1. Lexical Features

**Counted characters:**
`.` (dot), `-` (hyphen), `_` (underline), `/` (slash), `?` (questionmark),
`=` (equal), `@` (at), `&` (and), `!` (exclamation), ` ` (space),
`~` (tilde), `,` (comma), `+` (plus), `*` (asterisk), `#` (hashtag),
`$` (dollar), `%` (percent)

**URL Parts:**

```
https://sub.example.com:8080/dir/file.php?id=1&name=test
│      │    │       │   │    │   │        │              │
│      │    │       │   │    │   │        │              └─ params
│      │    │       │   │    │   │        └──────────────── query
│      │    │       │   │    │   └─────────────────────── file
│      │    │       │   │    └─────────────────────────── directory
│      │    │       │   └──────────────────────────────── port
│      │    │       └──────────────────────────────────── TLD
│      │    └──────────────────────────────────────────── domain
│      └───────────────────────────────────────────────── subdomain
└──────────────────────────────────────────────────────── scheme
```

**Implementation:**

- `CharacterCounter`: Counts character occurrences
- `VowelCounter`: Counts vowels in domain
- `TLDCounter`: Detects TLD patterns in text

#### 2. Network Features

**DNS Features (DNSResolver):**

```python
qty_ip_resolved     # Liczba adresów IP
qty_nameservers     # Liczba serwerów NS
qty_ip_resolved     # Number of IP addresses
qty_nameservers     # Number of NS servers
qty_mx_servers      # Number of MX servers
ttl_hostname        # TTL for hostname
domain_spf          # SPF record presence
```

**WHOIS Features (WHOISExtractor):**

```python
time_domain_activation   # Days since domain creation
time_domain_expiration   # Days until domain expiration
```

**Connection Features:**

```python
time_response          # Response time (s)
qty_redirects          # Number of HTTP redirects
tls_ssl_certificate    # Valid SSL certificate
```

**Value interpretation:**

- Success: numeric value (≥ 0)
- Error/timeout: `-1`

#### 3. Content Features (Full Mode)

**External Resources (ExternalResourceAnalyzer):**

```
Percentage = (external_objects / total_objects) × 100

Classification:
  < 22%  → 1  (Legitimate)
  22-61% → 0  (Suspicious)
  ≥ 61%  → -1 (Phishing)
```

**Anchor Analysis (AnchorAnalyzer):**

```
Percentage = (suspicious_anchors / total_anchors) × 100

Suspicious if href:
  - In ['#', 'javascript:void(0)', '']
  - Starts with 'javascript:'
  - External domain

Classification:
  < 31%  → 1  (Legitimate)
  31-67% → 0  (Suspicious)
  ≥ 67%  → -1 (Phishing)
```

**Form Handler (FormHandler):**

```
form action analysis:
  Empty/'about:blank' → -1 (Phishing)
  External domain     → 0  (Suspicious)
  Same domain         → 1  (Legitimate)
```

**Popup Detector (PopupDetector):**

```
Searches for: window.open, prompt(), alert(), confirm()
Present → -1 (Phishing)
Absent  → 1  (Legitimate)
```

#### 4. Pattern Detectors

**Email Detection:**

```regex
Pattern: [a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}
```

**IP Address Detection:**

```regex
Pattern: ^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$
```

**URL Shortener Detection:**

```
Domains: bit.ly, tinyurl.com, goo.gl, ow.ly, t.co, is.gd, ...
```

**Keyword Detection:**

```
Keywords: server, client, host, admin, user
```

### Extraction Example

**Input URL:**

```
https://secure-login.example-bank.com/verify/account.php?token=abc123&redirect=home
```

**Extracted features (example):**

```json
{
  "qty_dot_url": 3,
  "qty_hyphen_domain": 2,
  "length_url": 83,
  "domain_length": 27,
  "directory_length": 7,
  "file_length": 11,
  "params_length": 31,
  "qty_params": 2,
  "time_response": 0.234,
  "tls_ssl_certificate": 1,
  "qty_redirects": 0,
  "email_in_url": 0
}
```

### Error Handling

All network operations are protected:

- **Timeout**: 5 seconds (configurable)
- **DNS/WHOIS Error**: returns `-1`
- **HTTP Error**: returns `-1`
- **No Certificate**: returns `0`

---

## Installation & Setup

### Requirements

- Python 3.8+
- Node.js 18+
- Angular CLI

### Backend Setup (Flask API)

1. Navigate to the backend folder:

```bash
cd backend
```

2. Install required packages:

```bash
pip install -r requirements.txt
```

3. Generate HTTPS certificates (required for secure connection):

**Linux/macOS:**

```bash
./generate_certs.sh
```

**Windows (Git Bash):**

```bash
bash generate_certs.sh
```

**Windows (Command Prompt):**

```cmd
generate_certs.bat
```

4. Start the server:

```bash
python run.py
```

API will be available at: **https://localhost:5001** (HTTPS)

**Note:** Browsers will display a certificate warning (self-signed). This is normal in development.

### Frontend Setup (Angular)

1. Navigate to the frontend folder:

```bash
cd frontend
```

2. Install packages using npm:

```bash
npm install
```

3. Start the application:

```bash
ng serve
```

The website will be available at: http://localhost:4200

---

## API Endpoints

### 1. Email Phishing Analysis (NEW)

`POST /api/analyze-email` - Analyze email for phishing with Docker isolation

**Request Example:**

```json
{
  "sender": "support@paypal-secure.com",
  "receiver": "user@example.com",
  "content": "Your account has been suspended. Click here: http://bit.ly/restore",
  "attachments": [
    {
      "filename": "invoice.pdf",
      "content": "base64_encoded_content..."
    }
  ],
  "models": ["random_forest", "xgboost", "mlp"],
  "mode": "optimized"
}
```

**Response:**

```json
{
  "success": true,
  "is_phishing": true,
  "urls_found": 2,
  "urls": ["http://bit.ly/restore", "https://paypal-secure.com"],
  "analysis": [
    {
      "url": "http://bit.ly/restore",
      "success": true,
      "is_phishing": true,
      "predictions": [
        {
          "model": "random_forest",
          "prediction": 1,
          "prediction_label": "Phishing",
          "confidence": 0.94,
          "probabilities": {
            "safe": 0.06,
            "phishing": 0.94
          }
        }
      ],
      "ensemble": {
        "ensemble_prediction": 1,
        "ensemble_label": "Phishing",
        "confidence": 0.92,
        "models_voting": {
          "phishing": 3,
          "safe": 0,
          "total": 3
        }
      },
      "visualizations": {
        "charts": {
          "probabilities": "base64_png...",
          "voting": "base64_png...",
          "feature_importance": "base64_png...",
          "ensemble_voting": "base64_png..."
        }
      }
    }
  ]
}
```

### 2. SMS Phishing Analysis (NEW)

`POST /api/analyze-sms` - Analyze SMS message for phishing

**Request Example:**

```json
{
  "content": "URGENT: Your bank account will be closed. Verify now: http://tinyurl.com/bank123",
  "models": ["random_forest"],
  "mode": "optimized"
}
```

**Response:** Same structure as email analysis

### 3. URL Feature Extraction

`POST /api/extract-features` - Extract features from URL for phishing detection

**Request Example:**

```json
{
  "url": "https://example.com/page?id=1",
  "mode": "optimized"
}
```

**Response:**

```json
{
  "success": true,
  "url": "https://example.com/page?id=1",
  "mode": "optimized",
  "features": { ... },
  "feature_count": 74
}
```

### 4. Health Check

- `GET /api/health` - Server status
- `GET /` - API information

**curl Examples:**

```bash
# Analyze Email
curl -X POST https://localhost:5001/api/analyze-email \
  -H "Content-Type: application/json" \
  -d '{
    "sender": "admin@suspicious.com",
    "content": "Click here: http://phishing-site.com",
    "models": ["random_forest"],
    "mode": "optimized"
  }' \
  -k

# Analyze SMS
curl -X POST https://localhost:5001/api/analyze-sms \
  -H "Content-Type: application/json" \
  -d '{
    "content": "Win $1000! Click: http://scam.com",
    "models": ["xgboost", "random_forest"],
    "mode": "full"
  }' \
  -k
```

---

## Project Structure

### Backend (Flask + Python)

```
backend/
├── app/
│   ├── __init__.py          # Inicjalizacja Flask
│   └── routes.py            # Endpointy API
├── extractors/
│   ├── lexical/             # Analiza leksykalna (10 klas)
│   ├── network/             # Analiza sieciowa (4 klasy)
│   └── content/             # Analiza HTML (5 klas)
├── validators/              # Walidatory wejścia (2 klasy)
├── services/                # Logika biznesowa (1 klasa)
├── config/                  # Configuration (3 classes)
├── models/                  # Trained ML models
│   ├── logistic_regression/
│   ├── random_forest/
│   ├── xgboost/
│   ├── svm/
│   └── mlp/
├── data/
│   ├── dataset_full.csv
│   ├── dataset_optimized.csv
│   └── test/
├── notebooks/               # Jupyter notebooks (analysis)
├── certs/                   # HTTPS certificates
├── run.py                   # Entry point
├── requirements.txt
└── generate_certs.sh        # Certificate generator
```

### Frontend (Angular)

```
frontend/
├── src/
│   ├── app/
│   │   ├── components/      # UI components
│   │   ├── services/        # HTTP services
│   │   └── models/          # TypeScript models
│   ├── assets/              # Static resources
│   └── environments/        # Environment configs
├── angular.json
├── package.json
└── tsconfig.json
```

### Backend Classes (Each in separate file)

**Extractors (19 classes):**

1. `CharacterCounter` - Character counting
2. `VowelCounter` - Vowel counting
3. `TLDCounter` - TLD counting
4. `EmailDetector` - Email detection
5. `IPAddressDetector` - IP detection
6. `KeywordDetector` - Keyword detection
7. `URLShortenerDetector` - URL shortener detection
8. `URLPartExtractor` - URL/domain feature extraction
9. `PathPartExtractor` - Directory/file feature extraction
10. `ParamsExtractor` - Parameter feature extraction
11. `ResponseTimer` - Response time measurement
12. `DNSResolver` - DNS operations
13. `WHOISExtractor` - WHOIS data extraction
14. `SSLValidator` - Certificate validation
15. `ExternalResourceAnalyzer` - External resource analysis
16. `AnchorAnalyzer` - Anchor link analysis
17. `FormHandler` - Form analysis
18. `PopupDetector` - Popup detection
19. `HTMLContentExtractor` - HTML analysis orchestration

**Services (1 class):** 20. `URLFeatureService` - Main extraction service

**Validators (2 classes):** 21. `URLValidator` - URL validation 22. `ModeValidator` - Mode validation

**Config (3 classes):** 23. `FeatureSetConfig` - Feature set configuration 24. `CertificatePathResolver` - Certificate path resolution 25. `ServerRunner` - Server runner

---

## Machine Learning Models

### Algorithms Used

The system contains 5 trained models:

#### 1. Random Forest

- **Accuracy**: ~96%
- **Type**: Ensemble (bagging)
- **Parameters**: 100 trees, max_depth=None
- **Advantages**: Overfitting resistance, interpretable
- **File**: `models/random_forest/model.pkl`

#### 2. XGBoost

- **Accuracy**: ~97%
- **Type**: Gradient boosting
- **Parameters**: learning_rate=0.1, max_depth=6
- **Advantages**: Highest accuracy, optimization
- **File**: `models/xgboost/model.pkl`

#### 3. Multi-Layer Perceptron (MLP)

- **Accuracy**: ~95%
- **Type**: Neural network
- **Architecture**: (100, 50) hidden layers
- **Advantages**: Non-linear relationship learning
- **File**: `models/mlp/model.pkl`

#### 4. Support Vector Machine (SVM)

- **Accuracy**: ~94%
- **Type**: Kernel-based
- **Kernel**: RBF
- **Advantages**: Good generalization
- **File**: `models/svm/model.pkl`

#### 5. Logistic Regression

- **Accuracy**: ~92%
- **Type**: Linear model
- **Parameters**: max_iter=1000
- **Advantages**: Fast prediction, baseline
- **File**: `models/logistic_regression/model.pkl`

### Training Process

**Preprocessing:**

1. Numerical feature normalization
2. Train/test split (80/20)
3. Cross-validation (5-fold)

**Metrics:**

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC

**Feature Importance:**
Top 10 most important features:

1. `length_url`
2. `qty_hyphen_domain`
3. `time_domain_activation`
4. `tls_ssl_certificate`
5. `qty_redirects`
6. `qty_dot_domain`
7. `qty_params`
8. `directory_length`
9. `domain_length`
10. `qty_slash_url`

### Notebooks

Analysis and training process documented in Jupyter notebooks:

1. `0_comprehensive_statistical_analysis.ipynb` - Statistical analysis
2. `1_logistic_regression_phishing_detection.ipynb` - Logistic Regression
3. `2_random_forest_phishing_detection.ipynb` - Random Forest
4. `3_xgboost_phishing_detection.ipynb` - XGBoost
5. `4_svm_phishing_detection.ipynb` - SVM
6. `5_mlp_phishing_detection.ipynb` - MLP
7. `6_comprehensive_model_testing.ipynb` - Comparative tests

---

## Development

### Running Tests

```bash
cd backend
python -m pytest tests/ -v
```

### API Testing Example

**Python:**

```python
import requests

response = requests.post(
    'https://localhost:5001/api/extract-features',
    json={"url": "https://example.com", "mode": "optimized"},
    verify=False
)
print(response.json())
```

**JavaScript:**

```javascript
fetch('https://localhost:5001/api/extract-features', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ url: 'https://example.com', mode: 'optimized' }),
})
  .then((res) => res.json())
  .then((data) => console.log(data));
```

### Performance Benchmarks

**Optimized mode (74 features):**

- Average time: 2-5 seconds
- DNS/WHOIS: ~1-2s
- Lexical analysis: <0.1s
- Content analysis: N/A

**Full mode (111 features):**

- Average time: 3-8 seconds
- HTML parsing: +1-3s
- Network operations: ~2-4s

---

## Security & HTTPS

### Certificate Generation

System automatically generates self-signed certificates for development:

**Linux/macOS:**

```bash
cd backend
./generate_certs.sh
```

**Windows:**

```cmd
cd backend
generate_certs.bat
```

**Certificate Details:**

- Algorithm: RSA 4096-bit
- Validity: 365 days
- Format: PEM
- Location: `backend/certs/`

### SSL Configuration

Server automatically detects certificates:

1. Checks environment variables: `SSL_CERT_FILE`, `SSL_KEY_FILE`
2. Checks directory: `backend/certs/`
3. If no certificates: runs in HTTP mode (dev only)

**Production:**
Use certificates from trusted CA (Let's Encrypt, Certbot):

```bash
export SSL_CERT_FILE=/path/to/cert.pem
export SSL_KEY_FILE=/path/to/key.pem
python run.py
```

### Input Validation

All endpoints protected by:

- `URLValidator`: Validates URL scheme
- `ModeValidator`: Validates allowed modes
- Flask validation: JSON verification

---

## Troubleshooting

### Common Issues

**1. Import errors:**

```bash
pip install -r requirements.txt --force-reinstall
```

**2. SSL certificate warnings:**

- Expected in development with self-signed certificates
- Use `-k` flag in curl or `verify=False` in requests

**3. DNS/WHOIS timeouts:**

- Check internet connection
- Some domains may block automated queries
- Results return `-1` on timeout

**4. Port already in use:**

```bash
lsof -ti:5001 | xargs kill -9
```

## Dataset Source

https://gregavrbancic.github.io/Phishing-Dataset/
