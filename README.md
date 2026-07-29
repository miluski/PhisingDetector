# PhishingDetector - Secure ML-based Phishing Detection System

## Table of Contents

- [Project Overview](#project-overview)
- [System Architecture](#system-architecture)
- [Security Features](#security-features)
- [Frontend GUI](#frontend-gui)
- [Dataset](#dataset)
- [Current URL Corpus Dataset](#current-url-corpus-dataset)
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
- ✅ **Canonical feature modes**: 110 (`full`), 76 (`optimized`) and 48 (`selected`)
  (`optimized_v2` is accepted only as a legacy request alias; new files use `selected`)
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
   - Accuracy indicators loaded from the current benchmark results rather than
     hard-coded frontend values

4. **Analysis Mode Toggle**

   - **Optimized** (76 features)
   - **Full** (110 features)
   - **Selected** (48 features, selection learned only from the URL-corpus training split)

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
│   └── content/            # URL extraction from email/SMS/attachments
│       └── url_extractor.py
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
├── models/                 # Local trained binaries (ignored by Git)
│   ├── logistic_regression/
│   ├── random_forest/
│   ├── xgboost/
│   ├── svm/
│   └── mlp/
├── data/                   # Datasets and test fixtures
├── archive/                # Reversible archive of old generated versions
└── certs/                  # HTTPS certificates
```

Generated experiment artifacts are stored in the project-level `results/`
directory, outside the backend package.

### Code Properties

- ✅ **All methods ≤ 20 lines**
- ✅ **Each class in separate file**
- ✅ **Comments only for non-obvious decisions, fallbacks and limitations**
- ✅ **Clean naming** and single level of abstraction

---

## Dataset

The project keeps the original research dataset and a newer URL corpus dataset.
The newer corpus is stored in `backend/data/url_corpus/` and is the preferred
dataset for experiments that need fresher and more diverse URLs.

### Original Dataset Source

Dataset from scientific research:

- **Source**: [Phishing Website Features Dataset](https://gregavrbancic.github.io/Phishing-Dataset/)
- **Publication**: Mohammad, R. M., Thabtah, F., & McCluskey, L. (2014). "An assessment of features related to phishing websites using an automated technique"

### Characteristics

**Original dataset size:**

- `dataset_full.csv`: 88,647 rows × 111 feature columns
- `dataset_optimized.csv`: 88,647 rows × 71 features

**Class Distribution:**

- Phishing: 30,647 instances (34.6%)
- Legitimate: 58,000 instances (65.4%)

**Feature Types:**

1. **Lexical**: Character counting in URL parts
2. **Network**: DNS, WHOIS and SSL/TLS-related values
3. **Special**: Email patterns, IP/domain patterns and URL shorteners

### Original-dataset schema (legacy reference)

The CSV files currently stored for the original research dataset contain:

- `dataset_full.csv`: 88,647 records and 111 feature columns;
- `dataset_optimized.csv`: 88,647 records and 71 feature columns.

These files preserve the historical Grabarica schema, including legacy network
fields such as `asn_ip`, Google-index columns and WHOIS date fields. They are a
reference point for the thesis and are separate from the active URL-corpus
contract, which uses 110 (`full`), 76 (`optimized`) or 48 (`selected`) features
with explicit missing-value indicators.

---

## Current URL Corpus Dataset

The current reproducible URL-corpus iteration is `quality_v4_repaired`. The
original and intermediate iterations described below are retained only as
historical research documentation and are archived locally.

### Canonical local artifacts

- `backend/data/url_corpus/quality_v4_repaired_splits/` — raw train,
  validation and test CSV files;
- `backend/data/url_corpus/quality_v4_repaired_features/` — extracted `full`,
  `optimized` and `selected` feature CSV files;
- `backend/data/url_corpus/external_test_large_no_domain_overlap_splits/` —
  independent external test with 37,790 URLs;
- `backend/config/feature_manifests/url_corpus/quality_v4/selected.json` —
  canonical 48-feature `selected` manifest;
- `backend/models/staged_url_corpus_v4/` — staged models and retraining report.

The repaired splits contain:

| Split | Rows | Safe | Phishing |
|---|---:|---:|---:|
| Train | 108,762 | 54,381 | 54,381 |
| Validation | 21,676 | 10,838 | 10,838 |
| Test | 21,000 | 10,500 | 10,500 |

The feature files contain 110 prediction columns in `full`, 76 in
`optimized` and 48 in `selected`, in addition to URL, label and audit metadata.
The `selected` manifest is fitted on the training split only and then reused
unchanged for validation and test.

Registered-domain overlap between the three splits is zero. Each split also
contains the audited benign URL categories (`about`, `account`, `contact`,
`docs`, `help`, `login`, `payment`, `playlist`, `pricing`, `products`,
`secure`, `status`, `support`, `verify`, `watch`).

### Current retraining workflow

Run commands from the repository root:

```bash
# Validate schemas, category coverage and domain disjointness without training.
./.venv/bin/python backend/scripts/retrain_url_corpus_all_models.py \
  --modes full optimized selected \
  --minimum-category-samples 100 --check-only

# Train into the staged directory; production models are not overwritten.
./.venv/bin/python backend/scripts/retrain_url_corpus_all_models.py \
  --modes full optimized selected

# Recheck grouped, external and fixed real-world cases before promotion.
./.venv/bin/python backend/scripts/recheck_url_corpus_models.py
```

The first external recheck is long because it extracts live network features
for 37,790 URLs in three modes. A compatible completed cache makes subsequent
rechecks much shorter. The external report must contain
`"promotion_ready": true` before staged models can replace production models.

### Sources and feature policy

The corpus combines Phishing.Database, Tranco and PhreshPhish sources. Raw
source files and extracted CSVs are generated artifacts and are not committed
to Git. The pipeline is reproducible from the scripts, feature policy and
manifests. See [`docs/URL_CORPUS_DATA_SOURCES.md`](docs/URL_CORPUS_DATA_SOURCES.md)
for provenance and [`docs/RETRAINING_URL_CORPUS.md`](docs/RETRAINING_URL_CORPUS.md)
for the complete procedure. Figure construction and thesis interpretation are
documented in
[`docs/THESIS_FIGURES_AND_EVALUATION.md`](docs/THESIS_FIGURES_AND_EVALUATION.md).

## Historical URL Corpus Versions

### Location

Main files:

- `backend/data/url_corpus/raw_urls.csv` - merged raw URL list.
- `backend/data/url_corpus/raw_urls_quality.csv` - recommended quality-first raw URL list.
- `backend/data/url_corpus/raw_urls_phishing_db_tranco.csv` - Phishing.Database + Tranco source component.
- `backend/data/url_corpus/raw_urls_phreshphish.csv` - PhreshPhish source component.
- `backend/data/url_corpus/raw_urls_phreshphish_extra.csv` - additional PhreshPhish source component.
- `backend/data/url_corpus/raw_urls_phreshphish_external_test_large_no_domain_overlap.csv` - external PhreshPhish test URLs after removing domains present in the large quality corpus.
- `backend/data/url_corpus/dataset_full.csv` - extracted full feature dataset.
- `backend/data/url_corpus/quality_large_splits/dataset_optimized_{train,val,test}.csv` - optimized predefined feature splits.

Detailed source notes:

- `docs/URL_CORPUS_DATA_SOURCES.md`

### Sources

The URL corpus combines licensed and research-oriented sources:

- **Phishing.Database** - active phishing URLs, MIT license.
- **Tranco** - ranked popular domains used as legitimate URL seeds.
- **PhreshPhish** - Hugging Face phishing/benign URL dataset, `cc-by-4.0`.

### Rebuild Commands

Import PhreshPhish sample:

```bash
cd backend
../.venv/bin/python scripts/import_phreshphish_urls.py \
  --per-class-limit 50000 \
  --output data/url_corpus/raw_urls_phreshphish.csv \
  --dedupe-existing data/url_corpus/raw_urls_phishing_db_tranco.csv
```

Merge raw URL sources:

```bash
cd backend
../.venv/bin/python scripts/merge_raw_url_sources.py \
  --input data/url_corpus/raw_urls_phishing_db_tranco.csv data/url_corpus/raw_urls_phreshphish.csv data/url_corpus/raw_urls_phreshphish_extra.csv \
  --output data/url_corpus/raw_urls.csv
```

Build the recommended quality-first raw corpus:

```bash
cd backend
../.venv/bin/python scripts/build_quality_url_corpus.py \
  --input data/url_corpus/raw_urls_phreshphish.csv data/url_corpus/raw_urls_phreshphish_extra.csv data/url_corpus/raw_urls_phishing_db_tranco.csv \
  --output data/url_corpus/raw_urls_quality.csv \
  --per-class-limit 150000 \
  --minimum-benign-roots 40000 \
  --exclude-benign-root-domains-seen-in-phishing \
  --seed 42
```

Do not use `--require-hard-benign` for the final training corpus. That profile
removes short legitimate home pages such as `https://gmail.com/` and teaches
the model that short root URLs are phishing. It remains available only for
focused ablation experiments.

Migration of existing domain-isolated splits:

```bash
cd backend
../.venv/bin/python scripts/build_benign_root_hard_negatives.py \
  --input data/url_corpus/raw_urls_phishing_db_tranco.csv \
  --phishing-input data/url_corpus/raw_urls_quality.csv \
  --output data/url_corpus/benign_roots_tranco.csv

../.venv/bin/python scripts/inject_benign_roots_into_splits.py \
  --source-dir data/url_corpus/quality_large_splits \
  --benign-roots data/url_corpus/benign_roots_tranco.csv \
  --output-dir data/url_corpus/quality_corrected_splits \
  --root-limit 39000 \
  --seed 42
```

The migration rejects roots whose registered domain appears in phishing data,
keeps every split balanced, and requires zero registered-domain overlap.

Build full dataset with live network extraction:

```bash
cd backend
../.venv/bin/python scripts/build_url_corpus_dataset.py \
  --input data/url_corpus/raw_urls.csv \
  --mode full \
  --output data/url_corpus/dataset_full.csv \
  --workers 4 \
  --network-mode live \
  --missing-policy keep \
  --timeout 20 \
  --progress-interval 1000 \
  --resume-temp
```

Important extraction settings:

- `--network-mode live` runs DNS, WHOIS, HTTP response timing and SSL checks.
- `--missing-policy keep` preserves `-1` for unavailable network values.
- `--resume-temp` resumes long extraction from `dataset_full.csv.tmp` and `dataset_full.csv.tmp.done`.
- Final `dataset_full.csv` is replaced only after successful extraction.

Do not use deterministic network defaults for training. They create artificial
signals. Current policy uses live extraction and preserves unavailable values
as `-1`, with `missing_dns`, `missing_ssl`, `missing_ttl` and `missing_http`
indicators shared by training, external test and application inference;
`missing_whois` is retained only as an audit field while WHOIS coverage is zero.

### Leakage-Resistant Training Splits

For realistic phishing experiments, split the raw URLs before feature extraction.
Do not rely on a random split of `dataset_full.csv`, because repeated domains
and URL templates can leak between training and testing.

Create balanced train/validation/test splits with domain and template isolation:

```bash
cd backend
../.venv/bin/python scripts/build_url_corpus_splits.py \
  --input data/url_corpus/raw_urls.csv \
  --output-dir data/url_corpus/splits \
  --group-by registered_domain \
  --dedupe template \
  --balance-per-split \
  --val-size 0.15 \
  --test-size 0.15 \
  --seed 42
```

Current split report:

- Template deduplication removed `44703` near-duplicate URL patterns.
- Train: `81766` rows, balanced `40883` / `40883`.
- Validation: `16638` rows, balanced `8319` / `8319`.
- Test: `18302` rows, balanced `9151` / `9151`.
- Domain overlap between train/validation/test: `0`.
- URL template overlap between train/validation/test: `0`.

Build feature datasets from the split raw files:

```bash
cd backend
../.venv/bin/python scripts/build_url_corpus_dataset.py \
  --input data/url_corpus/splits/raw_train.csv \
  --mode full \
  --output data/url_corpus/splits/dataset_full_train.csv \
  --workers 4 \
  --network-mode live \
  --missing-policy keep \
  --timeout 20 \
  --failure-policy keep-empty \
  --resume-temp
```

Repeat the command for `raw_val.csv` and `raw_test.csv`, changing the output to
`dataset_full_val.csv` and `dataset_full_test.csv`. Use these three files in
training/evaluation instead of `train_test_split` on the full CSV.

For audits, add `--include-metadata` to `build_url_corpus_dataset.py`. It keeps
`source_index` and `url` next to the extracted features, but those columns must
be removed before model training.

To keep every raw row, including repeated URL templates and class imbalance,
create an additional no-skip split:

```bash
cd backend
../.venv/bin/python scripts/build_url_corpus_splits.py \
  --input data/url_corpus/raw_urls.csv \
  --output-dir data/url_corpus/splits_no_skip \
  --group-by registered_domain \
  --dedupe none \
  --val-size 0.15 \
  --test-size 0.15 \
  --seed 42
```

Then use `--failure-policy keep-empty` during extraction so failed live network
lookups do not remove rows.

### Recommended Quality-First Workflow

For phishing detection that should behave better on real email/SMS URLs, prefer
the quality split over a random split of `dataset_full.csv`.

Create the quality train/validation/test split:

```bash
cd backend
../.venv/bin/python scripts/build_url_corpus_splits.py \
  --input data/url_corpus/raw_urls_quality.csv \
  --output-dir data/url_corpus/quality_large_splits \
  --group-by registered_domain \
  --dedupe exact \
  --balance-per-split \
  --val-size 0.15 \
  --test-size 0.15 \
  --seed 42
```

Create the external benchmark with no domain overlap:

```bash
cd backend
../.venv/bin/python scripts/import_phreshphish_urls.py \
  --split test \
  --per-class-limit 50000 \
  --output data/url_corpus/raw_urls_phreshphish_external_test.csv \
  --dedupe-existing data/url_corpus/raw_urls_quality.csv

../.venv/bin/python scripts/filter_raw_url_by_domain.py \
  --input data/url_corpus/raw_urls_phreshphish_external_test.csv \
  --output data/url_corpus/raw_urls_phreshphish_external_test_large_no_domain_overlap.csv \
  --exclude-domains-from data/url_corpus/quality_large_splits/raw_train.csv data/url_corpus/quality_large_splits/raw_val.csv data/url_corpus/quality_large_splits/raw_test.csv

../.venv/bin/python scripts/build_url_corpus_splits.py \
  --input data/url_corpus/raw_urls_phreshphish_external_test_large_no_domain_overlap.csv \
  --output-dir data/url_corpus/external_test_large_no_domain_overlap_splits \
  --group-by registered_domain \
  --dedupe exact \
  --balance-per-split \
  --val-size 0 \
  --test-size 0.999 \
  --seed 43
```

Shard raw split files before live extraction. Domain-hash sharding keeps the
same registered domains in the same process, which improves cache reuse for
DNS/WHOIS and avoids extra network noise:

```bash
cd backend
../.venv/bin/python scripts/shard_raw_url_csv.py \
  --input data/url_corpus/quality_large_splits/raw_train.csv \
  --output-dir data/url_corpus/quality_large_splits/train_shards \
  --prefix raw_train \
  --shards 8 \
  --strategy domain-hash
```

Recommended live extraction profile:

- Quality profile: `4` train shards x `2` workers = `8` live lookups at once,
  `--timeout 30`.
- Faster profile: `4` train shards x `3` workers = `12` live lookups at once,
  `--timeout 20`.
- Avoid pushing far above `24-32` total live lookups; it usually creates more
  timeouts and less reliable network features.

Example shard extraction command:

```bash
cd backend
../.venv/bin/python scripts/build_url_corpus_dataset.py \
  --input data/url_corpus/quality_large_splits/train_shards/raw_train_shard_00.csv \
  --mode full \
  --output data/url_corpus/quality_large_splits/train_shards/dataset_full_train_shard_00.csv \
  --workers 2 \
  --network-mode live \
  --missing-policy keep \
  --failure-policy keep-empty \
  --timeout 30 \
  --progress-interval 1000 \
  --resume-temp \
  --include-metadata
```

After all shard CSVs are complete, merge them with:

```bash
cd backend
../.venv/bin/python scripts/merge_dataset_shards.py \
  --input data/url_corpus/quality_large_splits/train_shards/dataset_full_train_shard_00.csv data/url_corpus/quality_large_splits/train_shards/dataset_full_train_shard_01.csv data/url_corpus/quality_large_splits/train_shards/dataset_full_train_shard_02.csv data/url_corpus/quality_large_splits/train_shards/dataset_full_train_shard_03.csv data/url_corpus/quality_large_splits/train_shards/dataset_full_train_shard_04.csv data/url_corpus/quality_large_splits/train_shards/dataset_full_train_shard_05.csv data/url_corpus/quality_large_splits/train_shards/dataset_full_train_shard_06.csv data/url_corpus/quality_large_splits/train_shards/dataset_full_train_shard_07.csv \
  --output data/url_corpus/quality_large_splits/dataset_full_train.csv
```

Quality audit:

- `backend/data/url_corpus/quality_large_audit_report.json` verifies 50/50 class
  balance after deduplication.
- It also verifies `duplicate_url_rows = 0` and no missing/extra/duplicated
  records between each source split and its shards.

---

## Feature Extraction

The same `URLFeatureService` is used for corpus construction and application
inference. Under the current policy it produces 110 `full` features, 76
`optimized` features or the 48-feature `selected` manifest. Network
failures are represented consistently with missing indicators; `asn_ip` is
not used as a synthetic constant and raw WHOIS values are not model features.

### Extraction Process

Feature extraction occurs in three steps:

```
URL Input → Parsing → Feature Extraction → Feature Vector
              ↓            ↓                    ↓
         urlparse()   Lexical/Network     Dict[str, int]
                                             (48, 76 or 110 features)
         tldextract   Analysis            (48, 76 or 110 features)
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
qty_ip_resolved     # Number of IP addresses
qty_nameservers     # Number of NS servers
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

#### 3. Pattern Detectors

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

### 1. Email Phishing Analysis

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

### 2. SMS Phishing Analysis

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
  "feature_count": 76
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
│   ├── __init__.py          # Flask initialization
│   └── routes.py            # API endpoints
├── extractors/
│   ├── lexical/             # Lexical analysis (10 classes)
│   ├── network/             # Network analysis (4 classes)
│   └── content/             # URL extraction from email/SMS/attachments
├── validators/              # Input validators (2 classes)
├── services/                # Business logic (1 class)
├── config/                  # Configuration (3 classes)
├── models/                  # Local trained binaries (ignored by Git)
│   ├── logistic_regression/
│   ├── random_forest/
│   ├── xgboost/
│   ├── svm/
│   └── mlp/
├── data/
│   ├── dataset_full.csv
│   ├── dataset_optimized.csv
│   ├── url_corpus/
│   │   ├── raw_urls.csv
│   │   ├── dataset_full.csv
│   │   ├── quality_v4_repaired_splits/
│   │   └── quality_v4_repaired_features/
│   └── test/
├── notebooks/               # Tracked notebook sources and generators
├── certs/                   # HTTPS certificates
├── run.py                   # Entry point
├── requirements.txt
└── generate_certs.sh        # Certificate generator
```

Project-level generated artifacts:

```text
results/
├── original/
├── url_corpus/
└── notebook_runs/
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

**Extractors:**

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
15. `URLExtractor` - URL extraction from email, SMS and attachments

**Services:** `URLFeatureService` - Main extraction service

**Validators:** `URLValidator` - URL validation, `ModeValidator` - mode validation

**Config:** `FeatureSetConfig` - feature set configuration, `CertificatePathResolver` - certificate path resolution, `ServerRunner` - server runner

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

**Optimized mode (76 features):**

- Average time: 2-5 seconds
- DNS/WHOIS: ~1-2s
- Lexical analysis: <0.1s

**Full mode (110 features):**

- Average time: 3-8 seconds
- Network operations: ~2-4s

**URL corpus live extraction:**

- Long-running batch process.
- Uses DNS, WHOIS, HTTP response timing and SSL checks.
- Preserves `-1` for unavailable live network values when `--missing-policy keep` is used.

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
