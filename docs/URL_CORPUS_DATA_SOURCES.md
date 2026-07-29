# URL Corpus Data Sources

Data inventory date: 2026-06-20. Hard-negative inventory updated: 2026-06-23.

## Project files

| File | Role | Records | Labels |
|---|---|---:|---|
| `raw_urls_phishing_db_tranco.csv` | Phishing.Database positives and Tranco benign roots | 100,000 | 50,000 / 50,000 |
| `raw_urls_phreshphish.csv` | PhreshPhish sample | 100,000 | 50,000 / 50,000 |
| `raw_urls_phreshphish_extra.csv` | Additional deduplicated PhreshPhish training sample | 200,000 | 100,000 / 100,000 |
| `raw_urls.csv` | Merged URL-deduplicated source corpus | 400,000 | 200,000 / 200,000 |
| `raw_urls_quality_v4.csv` | Quality-first corpus before domain splits | 300,000 | 150,000 / 150,000 |
| `external_test_large_no_domain_overlap_splits/raw_test.csv` | Independent external test | 37,790 | 18,895 / 18,895 |

Raw files use the following format:

```csv
url,label
https://example.test/,0
https://phishing.example.test/login,1
```

`label=1` means phishing; `label=0` means legitimate or benign.

## PhreshPhish

- Dataset: <https://huggingface.co/datasets/phreshphish/phreshphish>
- Paper: <https://arxiv.org/abs/2507.10854>
- License: Creative Commons Attribution 4.0 International (`cc-by-4.0`).
- Source size: 666,315 records; 498,255 train and 168,060 test records.
- Used columns: `url` and `label`.
- Mapping: `phish -> 1`, `benign -> 0`.
- Local importer: `backend/scripts/import_phreshphish_urls.py`.

Example commands:

```bash
cd backend
../.venv/bin/python scripts/import_phreshphish_urls.py \
  --per-class-limit 50000 \
  --output data/url_corpus/raw_urls_phreshphish.csv \
  --dedupe-existing data/url_corpus/raw_urls_phishing_db_tranco.csv
```

## Phishing.Database

- Repository: <https://github.com/Phishing-Database/Phishing.Database>
- Feed: `phishing-links-ACTIVE.txt`.
- License: MIT.
- Each active-feed URL receives `label=1`.
- Downloader: `backend/scripts/download_feeds.py`.

## Tranco

- Website: <https://tranco-list.eu/>.
- Feed: the latest standard `top-1m.csv.zip` list.
- Role: popular domains used as benign seeds.
- Mapping: `https://{domain}/` receives `label=0`.
- Attribution and component licenses are documented on the Tranco website.

Root domains alone are too easy as benign examples. They are expanded with
realistic legal paths such as `login`, `account`, `billing`, `support`,
`verify` and `payment` to create hard negatives.

## Additional candidate sources

These sources may extend the corpus or provide independent evaluation data.
License attribution and domain/template isolation are required before import.

### Mixed URL, SMS, email and HTML dataset

- <https://huggingface.co/datasets/ealvaradob/phishing-dataset>
- Apache 2.0.
- Use only URL/website records for URL-only experiments; keep SMS and email
  records in separate message experiments.

### Mendeley web-page phishing dataset

- <https://data.mendeley.com/datasets/c2gw7fy2j4/3>
- DOI: <https://doi.org/10.17632/c2gw7fy2j4.3>
- CC BY 4.0; 11,430 URLs and 87 attributes.
- Best treated as a historical external benchmark because it was collected in
  2020/2021.

### Mendeley SMS phishing dataset

- <https://data.mendeley.com/datasets/f45bkkt8pr/1>
- DOI: <https://doi.org/10.17632/f45bkkt8pr.1>
- CC BY 4.0; 5,971 SMS messages.
- Use for the SMS module, not as URL-only training data.

## Source merging

`raw_urls.csv` is created by full-URL deduplication while preserving the first
label encountered:

```bash
cd backend
../.venv/bin/python scripts/merge_raw_url_sources.py \
  --input data/url_corpus/raw_urls_phishing_db_tranco.csv \
         data/url_corpus/raw_urls_phreshphish.csv \
         data/url_corpus/raw_urls_phreshphish_extra.csv \
  --output data/url_corpus/raw_urls.csv
```

## Quality-first corpus and domain splits

The active release uses `raw_urls_quality_v4.csv`, trusted benign roots and
synthetic lexical hard negatives. It removes benign roots whose registered
domain appears in phishing data, balances both labels and isolates domains
between train, validation and test.

The active repaired splits are:

| Split | Records | Phishing | Benign |
|---|---:|---:|---:|
| Train | 108,762 | 54,381 | 54,381 |
| Validation | 21,676 | 10,838 | 10,838 |
| Test | 21,000 | 10,500 | 10,500 |

The external test contains 37,790 records and has zero registered-domain
overlap with the internal splits.

```bash
cd backend
../.venv/bin/python scripts/build_url_corpus_splits.py \
  --input data/url_corpus/raw_urls_quality_v4.csv \
  --output-dir data/url_corpus/quality_v4_splits \
  --group-by registered_domain --dedupe template \
  --balance-per-split --min-hard-benign-per-split 1000 --seed 42
```

The repaired release is checked with:

```bash
./.venv/bin/python scripts/retrain_url_corpus_all_models.py \
  --feature-dir data/url_corpus/quality_v4_repaired_features \
  --modes full optimized selected \
  --minimum-category-samples 100 --check-only
```

## Feature extraction contract

Feature extraction uses the same `URLFeatureService` for dataset construction,
external recheck and application inference. The active policy is `live + keep`:

- lexical features contain measured values;
- `0` means the checked property is absent;
- `-1` means unavailable, timed out or not reliably measurable;
- missing indicators identify DNS, SSL, TTL and HTTP failures;
- `missing_whois` is an audit field while WHOIS is disabled;
- artificial `asn_ip`, Google-index and fixed network values are forbidden.

Extract a split with:

```bash
cd backend
../.venv/bin/python scripts/build_url_corpus_dataset.py \
  --input data/url_corpus/quality_v4_repaired_splits/raw_train.csv \
  --mode full --output data/url_corpus/quality_v4_repaired_features/dataset_full_train.csv \
  --workers 4 --network-mode live --missing-policy keep \
  --timeout 15 --failure-policy keep-empty --progress-interval 1000 --resume-temp
```

`--resume-temp` allows interrupted long-running extraction to continue. The
`selected` manifest is fitted on training data only and reused unchanged for
validation and test:

`backend/config/feature_manifests/url_corpus/quality_v4/selected.json`.

## WHOIS and RDAP

The WHOIS extractor tries RDAP over HTTPS first and falls back to the WHOIS
library. WHOIS dates are disabled in the default policy to keep current models
reproducible. Enabling `URL_WHOIS_ENABLED=1` changes the schema and requires a
new feature extraction and complete retraining; existing models must not be
mixed with that policy.
